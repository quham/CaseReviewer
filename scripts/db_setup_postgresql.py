# PDF to PostgreSQL Pipeline with pgvector Embeddings
# Step-by-step script for processing NSPCC case reviews

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import requests
from datetime import datetime

# Required installations:
# pip install pypdf2 langchain-text-splitters scikit-learn python-dotenv requests sentence-transformers psycopg2-binary
# pip install torch --index-url https://download.pytorch.org/whl/cpu

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch

# Fix NumPy compatibility issues
try:
    import numpy as np
    if hasattr(np, '_ARRAY_API'):
        # Force NumPy 1.x compatibility mode
        np._ARRAY_API = None
except ImportError:
    pass

# Load environment variables
load_dotenv()

class PDFToPostgreSQLProcessor:
    def __init__(self):
        """Initialize the processor with lightweight embeddings and PostgreSQL connection"""
        self.setup_environment()
        self.setup_embeddings()
        self.setup_postgresql()
        self.setup_text_splitter()
        self.setup_openrouter()
    
    def setup_environment(self):
        """Setup environment variables"""
        # PostgreSQL Configuration
        self.pg_host = os.getenv('POSTGRES_HOST', 'localhost')
        self.pg_port = os.getenv('POSTGRES_PORT', '5432')
        self.pg_database = os.getenv('POSTGRES_DATABASE', 'nspcc_cases')
        self.pg_user = os.getenv('POSTGRES_USER', 'postgres')
        self.pg_password = os.getenv('POSTGRES_PASSWORD')
        
        # OpenRouter Configuration (optional)
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.openrouter_base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        
        if not self.pg_password:
            raise ValueError("POSTGRES_PASSWORD not found in environment variables")
        
        # Make OpenRouter optional for testing
        if not self.openrouter_api_key:
            print("⚠️ Warning: OPENROUTER_API_KEY not found. LLM extraction will be disabled.")
            self.llm_enabled = False
        else:
            self.llm_enabled = True
    
    def setup_openrouter(self):
        """Initialize OpenRouter connection for LLM calls"""
        if not hasattr(self, 'llm_enabled') or not self.llm_enabled:
            print("⚠️ OpenRouter disabled - LLM extraction will not be available")
            return
            
        print("Setting up OpenRouter connection...")
        self.openrouter_headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "NSPCC Case Review Processor"
        }
        print("OpenRouter connection initialized successfully")
    
    def setup_embeddings(self):
        """Initialize lightweight embedding model for testing"""
        print("Setting up lightweight embedding model for testing...")
        try:
            # Use a much smaller, faster model for testing
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            print("✅ Lightweight model loaded successfully")
            print(f"Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"❌ Error loading lightweight model: {e}")
            print("Trying alternative lightweight model...")
            try:
                self.embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
                print("✅ Alternative lightweight model loaded successfully")
                print(f"Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
            except Exception as e2:
                print(f"❌ Error loading alternative model: {e2}")
                raise Exception("Could not load any embedding model")
    
    def setup_postgresql(self):
        """Initialize PostgreSQL connection and create tables if needed"""
        print("Setting up PostgreSQL connection...")
        
        try:
            # Connect to PostgreSQL
            self.conn = psycopg2.connect(
                host=self.pg_host,
                port=self.pg_port,
                database=self.pg_database,
                user=self.pg_user,
                password=self.pg_password
            )
            self.conn.autocommit = True
            
            print(f"✅ Connected to PostgreSQL at {self.pg_host}:{self.pg_port}")
            
            # Create tables and extensions
            self.create_database_schema()
            
        except Exception as e:
            print(f"❌ Error connecting to PostgreSQL: {e}")
            print("Make sure PostgreSQL is running and credentials are correct")
            raise
    
    def create_database_schema(self):
        """Create the necessary tables and extensions"""
        with self.conn.cursor() as cursor:
            # Enable pgvector extension
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                print("✅ pgvector extension enabled")
            except Exception as e:
                print(f"⚠️ Could not enable pgvector extension: {e}")
                print("Make sure pgvector is installed in your PostgreSQL instance")
                raise
            
            # Get embedding dimension
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            # Create case_reviews table
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS case_reviews (
                    id SERIAL PRIMARY KEY,
                    source_file TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    full_text TEXT,
                    embedding VECTOR({embedding_dim}),
                    structured_data JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_case_reviews_embedding 
                ON case_reviews USING ivfflat (embedding vector_cosine_ops);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_case_reviews_structured_data 
                ON case_reviews USING gin (structured_data);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_case_reviews_source_file 
                ON case_reviews (source_file);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_case_reviews_file_hash 
                ON case_reviews (file_hash);
            """)
            
            print("✅ Database schema created successfully")
    
    def setup_text_splitter(self):
        """Setup RecursiveCharacterTextSplitter with optimal settings for case reviews"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=150,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks (preferred)
                "\n",    # Line breaks
                ".",     # Sentence breaks
                " ",     # Word breaks
                ""       # Character breaks (last resort)
            ],
            keep_separator=True
        )
        print("✅ Text splitter configured")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        print(f"Extracting text from: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        print(f"⚠️ Error extracting text from page {page_num + 1}: {e}")
                        continue
                
                print(f"Extracted {len(text)} characters from {len(pdf_reader.pages)} pages")
                return text
                
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            raise
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text content"""
        try:
            # Truncate text if too long (sentence-transformers limit)
            max_chars = 8000  # Conservative limit for sentence-transformers
            if len(text) > max_chars:
                print(f"⚠️ Truncating text from {len(text)} to {max_chars} characters")
                text = text[:max_chars]
            
            # Create embedding with proper error handling
            embedding = self.embedding_model.encode(
                text,
                convert_to_numpy=False,  # Don't convert to numpy to avoid compatibility issues
                show_progress_bar=False
            )
            
            # Convert to list and ensure it's not all zeros
            embedding_list = embedding.tolist()
            
            # Check if embedding is valid (not all zeros)
            if all(x == 0.0 for x in embedding_list):
                raise ValueError("Generated embedding contains only zeros")
            
            return embedding_list
            
        except Exception as e:
            print(f"❌ Error creating embedding: {str(e)}")
            # Create a simple fallback embedding using text hash
            print("Creating fallback embedding...")
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            # Create a simple hash-based embedding as fallback
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            fallback_embedding = []
            
            for i in range(embedding_dim):
                # Use hash to generate pseudo-random but consistent values
                hash_val = int(text_hash[i*2:(i+1)*2], 16) / 255.0
                fallback_embedding.append(hash_val)
            
            print("✅ Fallback embedding created")
            return fallback_embedding
    
    def extract_structured_information(self, pdf_text: str, pdf_filename: str) -> Dict[str, Any]:
        """Extract structured information from PDF text using OpenRouter LLM"""
        print(f"Extracting structured information from {pdf_filename}...")
        
        # Check if LLM is enabled
        if not hasattr(self, 'llm_enabled') or not self.llm_enabled:
            print("⚠️ LLM extraction disabled - using fallback data")
            return self._create_fallback_structured_data(pdf_filename, pdf_text)
        
        # Create a comprehensive prompt for information extraction
        prompt = f"""
        You are an expert social worker and case review analyst. Analyze the following NSPCC case review document and extract key information in a structured format.

        Document: {pdf_filename}

        Please provide the following information in JSON format:

        1. **Clear Summary** (3-4 sentences): A concise but comprehensive overview of the case
        2. **Agencies Involved**: All agencies, organizations, services mentioned
        3. **Timeline of Key Events**: Chronological sequence of significant events
        4. **Recommendations**: All recommendations from the case review
        5. **Risk Factors**: All risk factors and warning signs identified
        6. **Outcomes**: What happened to the child/family and key lessons learned

        Return ONLY valid JSON with this exact structure:
        {{
            "summary": "clear case overview",
            "agencies": ["agency1", "agency2"],
            "timeline": [
                {{
                    "date": "date or period",
                    "event": "what happened",
                    "type": "missed_opportunity|critical_incident|concern_raised|positive_practice|other",
                    "impact": "significance of this event"
                }}
            ],
            "recommendations": ["recommendation 1", "recommendation 2"],
            "risk_factors": ["risk factor 1", "warning sign 2"],
            "outcomes": "outcomes and lessons learned"
        }}

        Document text:
        {pdf_text[:12000]}
        """
        
        try:
            # Call OpenRouter API
            response = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=self.openrouter_headers,
                json={
                    "model": "meta-llama/llama-3.3-70b-instruct:free",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 2000
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                try:
                    # Extract JSON content from response
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_content = content[start_idx:end_idx]
                        structured_data = json.loads(json_content)
                        print("✅ Structured information extracted successfully")
                        return structured_data
                    else:
                        raise ValueError("No JSON content found in response")
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON parsing failed: {e}")
                    return self._create_fallback_structured_data(pdf_filename, pdf_text)
                    
            else:
                print(f"❌ OpenRouter API error: {response.status_code}")
                print(f"Response: {response.text}")
                return self._create_fallback_structured_data(pdf_filename, pdf_text)
                
        except Exception as e:
            print(f"❌ Error calling OpenRouter API: {str(e)}")
            return self._create_fallback_structured_data(pdf_filename, pdf_text)
    
    def _create_fallback_structured_data(self, pdf_filename: str, pdf_text: str = "") -> Dict[str, Any]:
        """Create fallback structured data when LLM extraction fails"""
        # Try to extract some basic information from the text
        agencies = []
        recommendations = []
        
        if pdf_text:
            # Simple keyword extraction for agencies
            agency_keywords = ["social services", "police", "school", "health visitor", "gp", "hospital", "council"]
            for keyword in agency_keywords:
                if keyword.lower() in pdf_text.lower():
                    agencies.append(keyword.title())
            
            # Look for recommendation sections
            if "recommend" in pdf_text.lower():
                recommendations.append("Recommendations found in document - requires manual review")
        
        return {
            "summary": f"Case review document: {pdf_filename}. Manual analysis required for detailed information.",
            "agencies": agencies if agencies else ["Manual extraction required"],
            "timeline": [
                {
                    "date": "Unknown",
                    "event": "Document processed - manual timeline extraction required",
                    "type": "other",
                    "impact": "Information extraction requires manual review"
                }
            ],
            "recommendations": recommendations if recommendations else ["Manual extraction required"],
            "risk_factors": ["Manual extraction required"],
            "outcomes": "Manual analysis required for outcomes and lessons learned"
        }
    
    def save_to_database(self, pdf_path: str, text: str, embedding: List[float], structured_info: Dict[str, Any]) -> int:
        """Save PDF data to PostgreSQL database"""
        print("Saving to PostgreSQL database...")
        
        try:
            # Generate file hash for uniqueness
            file_hash = hashlib.md5(Path(pdf_path).name.encode()).hexdigest()
            
            with self.conn.cursor() as cursor:
                # Check if file already exists
                cursor.execute("SELECT id FROM case_reviews WHERE file_hash = %s", (file_hash,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record
                    cursor.execute("""
                        UPDATE case_reviews 
                        SET full_text = %s, embedding = %s, structured_data = %s, updated_at = NOW()
                        WHERE file_hash = %s
                        RETURNING id
                    """, (text, embedding, Json(structured_info), file_hash))
                    record_id = cursor.fetchone()[0]
                    print(f"✅ Updated existing record with ID: {record_id}")
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO case_reviews (source_file, file_hash, full_text, embedding, structured_data)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """, (Path(pdf_path).name, file_hash, text, embedding, Json(structured_info)))
                    record_id = cursor.fetchone()[0]
                    print(f"✅ Inserted new record with ID: {record_id}")
                
                return record_id
                
        except Exception as e:
            print(f"❌ Error saving to database: {e}")
            raise
    
    def save_structured_info_to_file(self, structured_info: Dict[str, Any], pdf_path: str):
        """Save structured information to a JSON file for backup"""
        output_dir = Path("extracted_data")
        output_dir.mkdir(exist_ok=True)
        
        filename = Path(pdf_path).stem + "_structured_info.json"
        output_path = output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_info, f, indent=2, ensure_ascii=False)
            print(f"✅ Structured info saved to: {output_path}")
        except Exception as e:
            print(f"⚠️ Could not save structured info to file: {e}")
    
    def process_pdf(self, pdf_path: str) -> int:
        """Complete pipeline to process a single PDF"""
        print(f"\n{'='*50}")
        print(f"Processing PDF: {pdf_path}")
        print(f"{'='*50}")
        
        try:
            # Step 1: Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # Step 2: Extract structured information using LLM
            print("Extracting structured information...")
            structured_info = self.extract_structured_information(text, Path(pdf_path).name)
            
            # Step 3: Create embedding
            print("Creating embedding...")
            embedding = self.create_embedding(text)
            print("✅ Embedding created successfully")
            
            # Step 4: Save to database
            record_id = self.save_to_database(pdf_path, text, embedding, structured_info)
            
            # Step 5: Save structured information to file (backup)
            self.save_structured_info_to_file(structured_info, pdf_path)
            
            print(f"✅ Successfully processed {Path(pdf_path).name}")
            print(f"📊 Summary: {structured_info.get('summary', 'N/A')[:100]}...")
            print(f"🗂️ Timeline events: {len(structured_info.get('timeline', []))}")
            print(f"🏢 Agencies involved: {len(structured_info.get('agencies', []))}")
            
            return record_id
            
        except Exception as e:
            print(f"❌ Error processing {Path(pdf_path).name}: {str(e)}")
            raise
    
    def search_similar_cases(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar cases using vector similarity"""
        print(f"\nSearching for: '{query_text[:100]}...'")
        
        try:
            # Create embedding for query
            query_embedding = self.create_embedding(query_text)
            
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Search using cosine similarity
                cursor.execute("""
                    SELECT 
                        id,
                        source_file,
                        structured_data,
                        created_at,
                        1 - (embedding <=> %s::vector) as similarity_score
                    FROM case_reviews
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, top_k))
                
                results = cursor.fetchall()
                
                # Convert to list of dictionaries
                formatted_results = []
                for row in results:
                    formatted_results.append({
                        'id': row['id'],
                        'source_file': row['source_file'],
                        'similarity_score': float(row['similarity_score']),
                        'structured_data': row['structured_data'],
                        'created_at': row['created_at']
                    })
                
                print(f"✅ Found {len(results)} similar cases")
                return formatted_results
                
        except Exception as e:
            print(f"❌ Error searching cases: {e}")
            return []
    
    def search_by_metadata(self, filters: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search cases by metadata criteria"""
        print(f"\nSearching by metadata: {filters}")
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                where_conditions = []
                params = []
                
                # Build dynamic WHERE clause based on filters
                if 'agencies' in filters:
                    where_conditions.append("structured_data->'agencies' ? %s")
                    params.append(filters['agencies'])
                
                if 'risk_factors' in filters:
                    where_conditions.append("structured_data->'risk_factors' ? %s")
                    params.append(filters['risk_factors'])
                
                if 'timeline_type' in filters:
                    where_conditions.append("structured_data->'timeline' @> %s")
                    params.append(json.dumps([{"type": filters['timeline_type']}]))
                
                # Build and execute query
                where_clause = " AND ".join(where_conditions) if where_conditions else "TRUE"
                params.append(top_k)
                
                cursor.execute(f"""
                    SELECT id, source_file, structured_data, created_at
                    FROM case_reviews
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s
                """, params)
                
                results = cursor.fetchall()
                
                # Convert to list of dictionaries
                formatted_results = []
                for row in results:
                    formatted_results.append({
                        'id': row['id'],
                        'source_file': row['source_file'],
                        'structured_data': row['structured_data'],
                        'created_at': row['created_at']
                    })
                
                print(f"✅ Found {len(results)} matching cases")
                return formatted_results
                
        except Exception as e:
            print(f"❌ Error searching by metadata: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM case_reviews")
                total_cases = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM case_reviews WHERE embedding IS NOT NULL")
                cases_with_embeddings = cursor.fetchone()[0]
                
                cursor.execute("SELECT AVG(LENGTH(full_text)) FROM case_reviews WHERE full_text IS NOT NULL")
                avg_text_length = cursor.fetchone()[0]
                
                return {
                    'total_cases': total_cases,
                    'cases_with_embeddings': cases_with_embeddings,
                    'average_text_length': int(avg_text_length) if avg_text_length else 0
                }
                
        except Exception as e:
            print(f"❌ Error getting database stats: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
            print("✅ Database connection closed")


def main():
    """Main function to process PDFs"""
    
    # Example PDF path - update this to your PDF location
    pdf_path = "/Users/quhamadefila/Desktop/childrens_sw/nspcc_case_reviews/2023EnfieldAndreCSPR.pdf"
    
    # Initialize processor
    try:
        processor = PDFToPostgreSQLProcessor()
        
        # Process single PDF
        if os.path.exists(pdf_path):
            record_id = processor.process_pdf(pdf_path)
            print(f"\n🎉 PDF processed successfully! Record ID: {record_id}")
        else:
            print(f"❌ PDF file not found: {pdf_path}")
            
        # Show database stats
        stats = processor.get_database_stats()
        print(f"\n📊 Database Stats:")
        print(f"   Total cases: {stats.get('total_cases', 0)}")
        print(f"   Cases with embeddings: {stats.get('cases_with_embeddings', 0)}")
        print(f"   Average text length: {stats.get('average_text_length', 0)} characters")
        
        # Example search
        print(f"\n🔍 Example search:")
        results = processor.search_similar_cases("missed opportunities in child protection", top_k=3)
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['source_file']} (similarity: {result['similarity_score']:.3f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'processor' in locals():
            processor.close()


if __name__ == "__main__":
    main()