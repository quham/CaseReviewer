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
from urllib.parse import urlparse

# Required installations:
# pip install pypdf2 langchain-text-splitters scikit-learn python-dotenv requests sentence-transformers psycopg2-binary
# pip install torch --index-url https://download.pytorch.org/whl/cpu

import PyPDF2
# from langchain_text_splitters import RecursiveCharacterTextSplitter
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
        # self.setup_text_splitter()  # Commented out - not needed for full PDF embeddings
        self.setup_openrouter()
        self.setup_retry_system()
    
    def setup_environment(self):
        """Setup environment variables"""
        # Neon PostgreSQL Configuration
        self.database_url = os.getenv('DATABASE_URL')
        
        if not self.database_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        # Parse connection string for display
        parsed = urlparse(self.database_url)
        self.pg_host = parsed.hostname
        self.pg_port = parsed.port or 5432
        self.pg_database = parsed.path[1:] if parsed.path else 'default'
        self.pg_user = parsed.username
        
        # OpenRouter Configuration (optional)
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.openrouter_base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        
        # Make OpenRouter optional for testing
        if not self.openrouter_api_key:
            print("⚠️ Warning: OPENROUTER_API_KEY not found. LLM extraction will be disabled.")
            self.llm_enabled = False
        else:
            self.llm_enabled = True
        
        print(f"✅ Database configuration loaded:")
        print(f"   Host: {self.pg_host}")
        print(f"   Port: {self.pg_port}")
        print(f"   Database: {self.pg_database}")
        print(f"   User: {self.pg_user}")
    
    def setup_embeddings(self):
        """Initialize the Qwen3-Embedding-8B embedding model"""
        print("Setting up Qwen3-Embedding-8B embedding model...")
        import psutil
        import gc

        def print_memory_usage():
            process = psutil.Process()
            memory_info = process.memory_info()
            print(f"   📊 Memory usage: {memory_info.rss / 1024 / 1024 / 1024:.2f} GB")
            
        print_memory_usage()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_memory_usage()
        
        try:
            # Use the Qwen3-Embedding-8B model from Hugging Face
            model_name = "Qwen/Qwen3-Embedding-8B"
            
            print(f"🔄 Loading {model_name}...")
            self.embedding_model = SentenceTransformer(model_name)
            print_memory_usage()
            
            # Get model information
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"✅ Embedding model loaded successfully!")
            print(f"   Model: {model_name}")
            print(f"   Embedding dimension: {embedding_dim}")
            print(f"   Context length: 32k tokens")
            print(f"   Multilingual support: 100+ languages")
            
            # Test the model with a simple sentence
            test_text = "This is a test sentence for the embedding model."
            test_embedding = self.embedding_model.encode(test_text)
            print(f"   Test embedding created: {len(test_embedding)} dimensions")
            
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise Exception("Failed to initialize embedding model - cannot proceed without embeddings")
    
    def setup_retry_system(self):
        """Setup the retry system for failed embeddings"""
        self.retry_file = Path("embedding_retry_list.json")
        self.retry_data = self.load_retry_data()
        print("✅ Retry system initialized")
    
    def load_retry_data(self) -> Dict[str, Any]:
        """Load existing retry data from file"""
        if self.retry_file.exists():
            try:
                with open(self.retry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load retry file: {e}")
                return {"failed_pdfs": [], "retry_attempts": {}}
        return {"failed_pdfs": [], "retry_attempts": {}}
    
    def save_retry_data(self):
        """Save retry data to file"""
        try:
            with open(self.retry_file, 'w', encoding='utf-8') as f:
                json.dump(self.retry_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Could not save retry file: {e}")
    
    def add_to_retry_list(self, pdf_path: str, error_message: str, structured_info: Dict[str, Any] = None):
        """Add a PDF to the retry list for embedding generation"""
        pdf_name = Path(pdf_path).name
        pdf_hash = hashlib.md5(pdf_name.encode()).hexdigest()
        
        retry_entry = {
            "pdf_name": pdf_name,
            "pdf_path": pdf_path,
            "pdf_hash": pdf_hash,
            "error_message": error_message,
            "added_date": datetime.now().isoformat(),
            "structured_info": structured_info,
            "retry_count": self.retry_data["retry_attempts"].get(pdf_hash, 0)
        }
        
        # Check if already in retry list
        existing_index = None
        for i, entry in enumerate(self.retry_data["failed_pdfs"]):
            if entry["pdf_hash"] == pdf_hash:
                existing_index = i
                break
        
        if existing_index is not None:
            # Update existing entry
            self.retry_data["failed_pdfs"][existing_index] = retry_entry
            print(f"📝 Updated retry entry for: {pdf_name}")
        else:
            # Add new entry
            self.retry_data["failed_pdfs"].append(retry_entry)
            print(f"📝 Added to retry list: {pdf_name}")
        
        self.save_retry_data()
    
    def get_retry_list(self) -> List[Dict[str, Any]]:
        """Get the current retry list"""
        return self.retry_data["failed_pdfs"].copy()
    
    def clear_retry_entry(self, pdf_hash: str):
        """Remove a PDF from the retry list after successful processing"""
        self.retry_data["failed_pdfs"] = [
            entry for entry in self.retry_data["failed_pdfs"] 
            if entry["pdf_hash"] != pdf_hash
        ]
        self.save_retry_data()
    
    def retry_failed_embeddings(self, max_retries: int = 3) -> List[str]:
        """Retry generating embeddings for all failed PDFs"""
        print(f"\n🔄 Retrying failed embeddings...")
        print(f"📋 Found {len(self.retry_data['failed_pdfs'])} PDFs in retry list")
        
        successful_retries = []
        failed_retries = []
        
        for entry in self.retry_data["failed_pdfs"]:
            pdf_path = entry["pdf_path"]
            pdf_hash = entry["pdf_hash"]
            retry_count = entry["retry_count"]
            
            if retry_count >= max_retries:
                print(f"⚠️ Skipping {entry['pdf_name']} - max retries exceeded ({retry_count})")
                failed_retries.append(entry['pdf_name'])
                continue
            
            print(f"\n🔄 Retrying embedding for: {entry['pdf_name']} (attempt {retry_count + 1}/{max_retries})")
            
            try:
                # Try to create embedding using the full PDF text
                # We need to re-extract the text since we don't store it in structured_info
                pdf_path = entry['pdf_path']
                if os.path.exists(pdf_path):
                    text = self.extract_text_from_pdf(pdf_path)
                    embedding = self.create_embedding(text)
                else:
                    raise Exception(f"PDF file not found: {pdf_path}") 
                
                # Update the database with the new embedding
                self.update_embedding_in_database(pdf_hash, embedding)
                
                # Remove from retry list
                self.clear_retry_entry(pdf_hash)
                
                # Update retry count
                self.retry_data["retry_attempts"][pdf_hash] = retry_count + 1
                
                print(f"✅ Successfully retried embedding for: {entry['pdf_name']}")
                successful_retries.append(entry['pdf_name'])
                
            except Exception as e:
                print(f"❌ Retry failed for {entry['pdf_name']}: {e}")
                # Update retry count
                self.retry_data["retry_attempts"][pdf_hash] = retry_count + 1
                entry["retry_count"] = retry_count + 1
                entry["error_message"] = str(e)
                entry["last_retry_date"] = datetime.now().isoformat()
        
        # Save updated retry data
        self.save_retry_data()
        
        print(f"\n📊 Retry Results:")
        print(f"   ✅ Successful: {len(successful_retries)}")
        print(f"   ❌ Failed: {len(failed_retries)}")
        print(f"   📋 Remaining in retry list: {len(self.retry_data['failed_pdfs'])}")
        
        return successful_retries
    
    def update_embedding_in_database(self, pdf_hash: str, embedding: List[float]):
        """Update the embedding for a specific PDF in the database"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE case_reviews 
                    SET embedding = %s, updated_at = NOW()
                    WHERE file_hash = %s
                """, (embedding, pdf_hash))
                
                if cursor.rowcount == 0:
                    print(f"⚠️ No record found for hash: {pdf_hash}")
                else:
                    print(f"✅ Updated embedding for PDF hash: {pdf_hash}")
                
                self.conn.commit()
                
        except Exception as e:
            print(f"❌ Error updating embedding in database: {e}")
            raise
    
    def setup_postgresql(self):
        """Initialize PostgreSQL connection with pgvector support"""
        print("Setting up PostgreSQL connection...")
        
        try:
            # Connect to PostgreSQL
            self.conn = psycopg2.connect(self.database_url)
            
            # Enable pgvector extension
            with self.conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.conn.commit()
            
            # Validate database schema
            self.validate_database_schema()
            
            print("✅ PostgreSQL connection established successfully")
            print("✅ pgvector extension enabled")
            
        except Exception as e:
            print(f"❌ Error connecting to PostgreSQL: {e}")
            raise
    
    def validate_database_schema(self):
        """Validate that required tables and columns exist"""
        print("Validating database schema...")
        
        try:
            with self.conn.cursor() as cursor:
                # Check if case_reviews table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'case_reviews'
                    )
                """)
                
                if not cursor.fetchone()[0]:
                    raise Exception("Table 'case_reviews' does not exist. Please run database migrations first.")
                
                # Check if timeline_events table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'timeline_events'
                    )
                """)
                
                if not cursor.fetchone()[0]:
                    raise Exception("Table 'timeline_events' does not exist. Please run database migrations first.")
                
                # Check if users table exists (for stats)
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'users'
                    )
                """)
                
                if not cursor.fetchone()[0]:
                    print("⚠️ Warning: Table 'users' does not exist. User statistics will be unavailable.")
                
                print("✅ Database schema validation passed")
                
        except Exception as e:
            print(f"❌ Database schema validation failed: {e}")
            raise
    
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
        
        # Simple list of free models to try in order
        self.models = [
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.3-70b-instruct:free", 
            "qwen/qwen3-8b:free",
            "anthropic/claude-3-haiku:free",
            "mistralai/mistral-7b-instruct:free"
        ]
        
        print(f"✅ OpenRouter connection initialized successfully")
        print(f"📋 Will try {len(self.models)} models in order")
    
    def try_models_in_order(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Try each model in order until one works"""
        for i, model in enumerate(self.models, 1):
            print(f"🔄 Trying model {i}/{len(self.models)}: {model}")
            
            try:
                response = requests.post(
                    f"{self.openrouter_base_url}/chat/completions",
                    headers=self.openrouter_headers,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 2000
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    # Try to extract JSON
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        json_content = content[start_idx:end_idx]
                        structured_data = json.loads(json_content)
                        print(f"✅ Success with {model}")
                        return structured_data
                    else:
                        print(f"⚠️ {model}: No valid JSON in response")
                        continue
                        
                else:
                    print(f"❌ {model}: API error {response.status_code}")
                    continue
                    
            except Exception as e:
                print(f"❌ {model}: Error - {str(e)}")
                continue
        
        print("❌ All models failed")
        return None
    
    def add_model(self, model_name: str):
        """Add a new model to the list"""
        if model_name not in self.models:
            self.models.append(model_name)
            print(f"✅ Added {model_name}")
        else:
            print(f"⚠️ {model_name} already exists")
    
    def remove_model(self, model_name: str):
        """Remove a model from the list"""
        if model_name in self.models:
            self.models.remove(model_name)
            print(f"✅ Removed {model_name}")
        else:
            print(f"⚠️ {model_name} not found")
    
    def get_models(self):
        """Get current list of models"""
        return self.models.copy()
    
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
        """Create embedding for full PDF text content"""
        try:
            # Qwen3-Embedding-8B can handle up to 32k tokens
            # Approximate: 1 token ≈ 4 characters, so 32k tokens ≈ 128k characters
            max_chars = 128000  # Increased limit for full PDFs
            
            if len(text) > max_chars:
                print(f"⚠️ PDF text ({len(text)} chars) exceeds 32k token limit")
                print(f"   Truncating to {max_chars} characters for embedding")
                text = text[:max_chars]
            else:
                print(f"✅ Processing full PDF text: {len(text)} characters")
            
            # Create embedding for the full document
            embedding = self.embedding_model.encode(
                text,
                convert_to_numpy=False,
                show_progress_bar=False
            )
            
            # Convert to list and ensure it's not all zeros
            embedding_list = embedding.tolist()
            
            # Check if embedding is valid (not all zeros)
            if all(x == 0.0 for x in embedding_list):
                raise ValueError("Generated embedding contains only zeros")
            
            print(f"✅ Created embedding for full PDF: {len(embedding_list)} dimensions")
            return embedding_list
            
        except Exception as e:
            print(f"❌ Error creating embedding: {str(e)}")
            raise Exception(f"Embedding generation failed: {str(e)}")
    
    def extract_structured_information(self, pdf_text: str, pdf_filename: str) -> Dict[str, Any]:
        """Extract structured information from PDF text using OpenRouter LLM"""
        print(f"Extracting structured information from {pdf_filename}...")
        
        # Check if LLM is enabled
        if not hasattr(self, 'llm_enabled') or not self.llm_enabled:
            print("❌ LLM extraction disabled - cannot process PDF")
            raise Exception("LLM extraction disabled - cannot process PDF")
        
        # Create a comprehensive prompt for information extraction aligned with the schema
        prompt = f"""
        You are an expert social worker and case review analyst. Analyze the following NSPCC case review document and extract key information in a structured format.

        Document: {pdf_filename}

        Please provide the following information in JSON format that matches the database schema:

        1. **title**: A concise title for the case review
        2. **summary**: A clear summary (3-4 sentences) of the case
        3. **child_age**: Approximate age of the child (if mentioned)
        5. **risk_types**: Array of risk types identified (e.g., ["neglect", "abuse", "domestic_violence"])
        6. **outcome**: What happened to the child/family
        7. **review_date**: Date of the review (if mentioned)
        8. **agencies**: Array of agencies involved (e.g., ["social_services", "police", "school"])
        9. **warning_signs_early**: Array of early warning signs identified
        10. **risk_factors**: Array of risk factors identified
        11. **barriers**: Array of barriers to effective intervention
        12. **relationship_model**: Object with family structure, professional network, support systems, and power dynamics
        13. **timeline**: Array of key events with dates, types, and impact

        Return ONLY valid JSON with this exact structure:
        {{
            "title": "Case title",
            "summary": "clear case overview",
            "child_age": 5,
            "risk_types": ["neglect", "abuse"],
            "outcome": "outcomes and lessons learned",
            "review_date": "2023-01-01",
            "agencies": ["social_services", "police"],
            "warning_signs_early": ["warning sign 1", "warning sign 2"],
            "risk_factors": ["risk factor 1", "risk factor 2"],
            "barriers": ["barrier 1", "barrier 2"],
            "relationship_model": {{
                "familyStructure": "description",
                "professionalNetwork": "description",
                "supportSystems": "description",
                "powerDynamics": "description"
            }},
            "timeline": [
                {{
                    "date": "date or period",
                    "description": "what is the event and what happened",
                    "type": "missed_opportunity|critical_incident|concern_raised|positive_practice|other",
                    "impact": "significance of this event",
                }}
            ]
        }}

        Document text:
        {pdf_text}
        """
        
        # Try models in order until one works
        structured_data = self.try_models_in_order(prompt)
        
        if structured_data:
            print("✅ Structured information extracted successfully using LLM")
            return structured_data
        else:
            print("❌ All LLM models failed - cannot process PDF")
            raise Exception("All LLM models failed - cannot extract structured information")
    

    
    def save_to_database(self, pdf_path: str, text: str, embedding: List[float], structured_info: Dict[str, Any]) -> str:
        """Save PDF data to PostgreSQL database using the existing schema"""
        print("Saving to PostgreSQL database...")
        
        try:
            # Generate file hash for uniqueness
            file_hash = hashlib.md5(Path(pdf_path).name.encode()).hexdigest()
            
            # Start transaction
            self.conn.autocommit = False
            
            with self.conn.cursor() as cursor:
                # Check if file already exists
                cursor.execute("SELECT id FROM case_reviews WHERE file_hash = %s", (file_hash,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record
                    cursor.execute("""
                        UPDATE case_reviews 
                        SET title = %s, summary = %s, child_age = %s, risk_types = %s,
                            outcome = %s, review_date = %s, agencies = %s, warning_signs_early = %s,
                            risk_factors = %s, barriers = %s, relationship_model = %s, 
                            embedding = %s
                        WHERE file_hash = %s
                        RETURNING id
                    """, (
                        structured_info.get('title'),
                        structured_info.get('summary'),
                        structured_info.get('child_age'),
                        Json(structured_info.get('risk_types', [])),
                        structured_info.get('outcome'),
                        structured_info.get('review_date'),
                        Json(structured_info.get('agencies', [])),
                        Json(structured_info.get('warning_signs_early', [])),
                        Json(structured_info.get('risk_factors', [])),
                        Json(structured_info.get('barriers', [])),
                        Json(structured_info.get('relationship_model')),
                        embedding,
                        file_hash
                    ))
                    record_id = cursor.fetchone()[0]
                    print(f"✅ Updated existing record with ID: {record_id}")
                    
                    # Save timeline events if they exist (for update case)
                    timeline_events = structured_info.get('timeline', [])
                    if timeline_events and isinstance(timeline_events, list):
                        # Delete existing timeline events for this case
                        cursor.execute("DELETE FROM timeline_events WHERE case_review_id = %s", (record_id,))
                        
                        # Insert new timeline events
                        for event in timeline_events:
                            if isinstance(event, dict):
                                cursor.execute("""
                                    INSERT INTO timeline_events (
                                        case_review_id, event_date, event_type, description, 
                                        impact
                                    )
                                    VALUES (%s, %s, %s, %s, %s)
                                """, (
                                    record_id,
                                    event.get('date'),
                                    event.get('type', 'other'),
                                    event.get('description', ''),
                                    event.get('impact', ''),
                                  
                                ))
                        
                        print(f"✅ Updated {len(timeline_events)} timeline events")
                else:
                    # Insert new record
                    cursor.execute("""
                        INSERT INTO case_reviews (
                            title, summary, child_age, risk_types, outcome, review_date,
                            agencies, warning_signs_early, risk_factors, barriers, relationship_model,
                            embedding, source_file, file_hash
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        structured_info.get('title'),
                        structured_info.get('summary'),
                        structured_info.get('child_age'),
                        Json(structured_info.get('risk_types', [])),
                        structured_info.get('outcome'),
                        structured_info.get('review_date'),
                        Json(structured_info.get('agencies', [])),
                        Json(structured_info.get('warning_signs_early', [])),
                        Json(structured_info.get('risk_factors', [])),
                        Json(structured_info.get('barriers', [])),
                        Json(structured_info.get('relationship_model')),
                        embedding,
                        Path(pdf_path).name,
                        file_hash
                    ))
                    record_id = cursor.fetchone()[0]
                    print(f"✅ Inserted new record with ID: {record_id}")
                
                # Save timeline events if they exist (METHOD 1)
                timeline_events = structured_info.get('timeline', [])
                if timeline_events and isinstance(timeline_events, list):
                    # Delete existing timeline events for this case (in case of update)
                    cursor.execute("DELETE FROM timeline_events WHERE case_review_id = %s", (record_id,))
                    
                    # Insert new timeline events
                    for event in timeline_events:
                        if isinstance(event, dict):
                            cursor.execute("""
                                INSERT INTO timeline_events (
                                    case_review_id, event_date, event_type, description, 
                                    impact
                                )
                                VALUES (%s, %s, %s, %s, %s)
                            """, (
                                record_id,
                                event.get('date'),
                                event.get('type', 'other'),
                                event.get('description', ''),
                                event.get('impact', '')
                            ))
                    
                    print(f"✅ Saved {len(timeline_events)} timeline events (METHOD 1)")
                
                # Commit transaction for METHOD 1
                self.conn.commit()
                self.conn.autocommit = True
                
                return record_id
                
        except Exception as e:
            # Rollback transaction on error for METHOD 1
            self.conn.rollback()
            self.conn.autocommit = True
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
    
    def process_pdf(self, pdf_path: str) -> str:
        """Complete pipeline to process a single PDF with full content embedding"""
        print(f"\n{'='*50}")
        print(f"Processing PDF: {pdf_path}")
        print(f"{'='*50}")
        
        try:
            # Step 1: Extract text from PDF (entire document)
            text = self.extract_text_from_pdf(pdf_path)
            print(f"📄 Extracted {len(text)} characters from entire PDF")
            
            # Step 2: Extract structured information using LLM
            print("Extracting structured information...")
            structured_info = self.extract_structured_information(text, Path(pdf_path).name)
            
            # Step 3: Create embedding for full PDF content
            print("Creating embedding for full PDF content...")
            try:
                embedding = self.create_embedding(text)
                print("✅ Full PDF embedding created successfully")
            except Exception as embedding_error:
                print(f"❌ Embedding failed: {embedding_error}")
                print("📝 Adding to retry list for later embedding generation...")
                
                # Add to retry list with structured info and PDF path for later embedding
                self.add_to_retry_list(pdf_path, str(embedding_error), structured_info)
                
                # Save structured info to database without embedding
                record_id = self.save_to_database_without_embedding(pdf_path, text, structured_info)
                
                print(f"✅ PDF processed and saved to database (ID: {record_id})")
                print(f"📝 Full PDF embedding will be retried later using the original PDF file. Check retry list: {self.retry_file}")
                
                return record_id
            
            # Step 4: Save to database with full PDF content embedding
            record_id = self.save_to_database(pdf_path, text, embedding, structured_info)
            
            # Step 5: Save structured information to file (backup)
            self.save_structured_info_to_file(structured_info, pdf_path)
            
            print(f"✅ Successfully processed {Path(pdf_path).name}")
            print(f"📊 Summary: {structured_info.get('summary', 'N/A')[:100]}...")
            print(f"🏢 Agencies involved: {len(structured_info.get('agencies', []))}")
            print(f"⚠️ Risk types: {len(structured_info.get('risk_types', []))}")
            print(f"🔗 Full PDF embedding stored: {len(embedding)} dimensions")
            
            return record_id
            
        except Exception as e:
            print(f"❌ Error processing {Path(pdf_path).name}: {str(e)}")
            raise
    
    def save_to_database_without_embedding(self, pdf_path: str, text: str, structured_info: Dict[str, Any]) -> str:
        """Save PDF data to database without embedding (for retry cases) - METHOD 2"""
        print("Saving to PostgreSQL database (without embedding)...")
        
        try:
            # Generate file hash for uniqueness
            file_hash = hashlib.md5(Path(pdf_path).name.encode()).hexdigest()
            
            # Start transaction for METHOD 2
            self.conn.autocommit = False
            
            with self.conn.cursor() as cursor:
                # Check if file already exists
                cursor.execute("SELECT id FROM case_reviews WHERE file_hash = %s", (file_hash,))
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing record (without embedding)
                    cursor.execute("""
                        UPDATE case_reviews 
                        SET title = %s, summary = %s, child_age = %s, risk_types = %s,
                            outcome = %s, review_date = %s, agencies = %s, warning_signs_early = %s,
                            risk_factors = %s, barriers = %s, relationship_model = %s
                        WHERE file_hash = %s
                        RETURNING id
                    """, (
                        structured_info.get('title'),
                        structured_info.get('summary'),
                        structured_info.get('child_age'),
                        Json(structured_info.get('risk_types', [])),
                        structured_info.get('outcome'),
                        structured_info.get('review_date'),
                        Json(structured_info.get('agencies', [])),
                        Json(structured_info.get('warning_signs_early', [])),
                        Json(structured_info.get('risk_factors', [])),
                        Json(structured_info.get('barriers', [])),
                        Json(structured_info.get('relationship_model')),
                        file_hash
                    ))
                    record_id = cursor.fetchone()[0]
                    print(f"✅ Updated existing record with ID: {record_id}")
                else:
                    # Insert new record (without embedding)
                    cursor.execute("""
                        INSERT INTO case_reviews (
                            title, summary, child_age, risk_types, outcome, review_date,
                            agencies, warning_signs_early, risk_factors, barriers, relationship_model,
                            source_file, file_hash
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        structured_info.get('title'),
                        structured_info.get('summary'),
                        structured_info.get('child_age'),
                        Json(structured_info.get('risk_types', [])),
                        structured_info.get('outcome'),
                        structured_info.get('review_date'),
                        Json(structured_info.get('agencies', [])),
                        Json(structured_info.get('warning_signs_early', [])),
                        Json(structured_info.get('risk_factors', [])),
                        Json(structured_info.get('barriers', [])),
                        Json(structured_info.get('relationship_model')),
                        Path(pdf_path).name,
                        file_hash
                    ))
                    record_id = cursor.fetchone()[0]
                    print(f"✅ Inserted new record with ID: {record_id}")
                
                # Save timeline events if they exist (METHOD 2)
                timeline_events = structured_info.get('timeline', [])
                if timeline_events and isinstance(timeline_events, list):
                    # Delete existing timeline events for this case (in case of update)
                    cursor.execute("DELETE FROM timeline_events WHERE case_review_id = %s", (record_id,))
                    
                    # Insert new timeline events
                    for event in timeline_events:
                        if isinstance(event, dict):
                            cursor.execute("""
                                INSERT INTO timeline_events (
                                    case_review_id, event_date, event_type, description, 
                                    impact
                                )
                                VALUES (%s, %s, %s, %s, %s)
                            """, (
                                record_id,
                                event.get('date'),
                                event.get('type', 'other'),
                                event.get('description', ''),
                                event.get('impact', '')
                            ))
                    
                    print(f"✅ Saved {len(timeline_events)} timeline events (METHOD 2)")
                
                # Commit transaction for METHOD 2
                self.conn.commit()
                self.conn.autocommit = True
                
                return record_id
                
        except Exception as e:
            # Rollback transaction on error for METHOD 2
            self.conn.rollback()
            self.conn.autocommit = True
            print(f"❌ Error saving to database: {e}")
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
                        title,
                        summary,
                        source_file,
                        risk_types,
                        agencies,
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
                        'title': row['title'],
                        'summary': row['summary'],
                        'source_file': row['source_file'],
                        'risk_types': row['risk_types'],
                        'agencies': row['agencies'],
                        'similarity_score': float(row['similarity_score']),
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
                    where_conditions.append("agencies ? %s")
                    params.append(filters['agencies'])
                
                if 'risk_types' in filters:
                    where_conditions.append("risk_types ? %s")
                    params.append(filters['risk_types'])
                
                if 'child_age' in filters:
                    where_conditions.append("child_age = %s")
                    params.append(filters['child_age'])
                
                # Build and execute query
                where_clause = " AND ".join(where_conditions) if where_conditions else "TRUE"
                params.append(top_k)
                
                cursor.execute(f"""
                    SELECT id, title, summary, source_file, risk_types, agencies, created_at
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
                        'title': row['title'],
                        'summary': row['summary'],
                        'source_file': row['source_file'],
                        'risk_types': row['risk_types'],
                        'agencies': row['agencies'],
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
                
                cursor.execute("SELECT AVG(LENGTH(summary)) FROM case_reviews WHERE summary IS NOT NULL")
                avg_text_length = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM timeline_events")
                total_timeline_events = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM users")
                total_users = cursor.fetchone()[0]
                
                return {
                    'total_cases': total_cases,
                    'cases_with_embeddings': cases_with_embeddings,
                    'average_text_length': int(avg_text_length) if avg_text_length else 0,
                    'total_timeline_events': total_timeline_events,
                    'total_users': total_users
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
    """Main function to process PDFs with retry system demonstration"""
    
    # Example PDF path - update this to your PDF location
    # pdf_path = "/Users/quhamadefila/Desktop/childrens_sw/nspcc_case_reviews/2023EnfieldAndreCSPR.pdf"
    pdf_path = input("Enter the path to your PDF file: ").strip()
    
    if not pdf_path:
        print("❌ No PDF path provided. Exiting.")
        return
    
    # Initialize processor
    try:
        processor = PDFToPostgreSQLProcessor()
        
        # Demonstrate retry system functionality
        print("\n" + "="*60)
        print("🔄 RETRY SYSTEM DEMONSTRATION")
        print("="*60)
        
        # Show current retry list
        retry_list = processor.get_retry_list()
        print(f"\n📋 Current Retry List ({len(retry_list)} PDFs):")
        if retry_list:
            for i, entry in enumerate(retry_list, 1):
                print(f"   {i}. {entry['pdf_name']} (attempts: {entry['retry_count']})")
        else:
            print("   No PDFs in retry list")
        
        # Demonstrate retry functionality
        if retry_list:
            print(f"\n🔄 Retrying failed embeddings...")
            successful_retries = processor.retry_failed_embeddings(max_retries=3)
            
            if successful_retries:
                print(f"\n✅ Successfully retried {len(successful_retries)} embeddings")
            else:
                print(f"\n⚠️ No embeddings were successfully retried")
        
        print("\n" + "="*60)
        print("📄 PDF PROCESSING")
        print("="*60)
        
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
        print(f"   Total timeline events: {stats.get('total_timeline_events', 0)}")
        print(f"   Total users: {stats.get('total_users', 0)}")
        
        # Show final retry list status
        final_retry_list = processor.get_retry_list()
        print(f"\n📋 Final Retry List Status:")
        print(f"   PDFs waiting for embedding: {len(final_retry_list)}")
        if final_retry_list:
            print(f"   Retry file location: {processor.retry_file}")
            print(f"   You can run retry_failed_embeddings() later to process these")
        
        # Example search
        print(f"\n🔍 Example search:")
        results = processor.search_similar_cases("missed opportunities in child protection", top_k=3)
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['title']} (similarity: {result['similarity_score']:.3f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'processor' in locals():
            processor.close()


if __name__ == "__main__":
    main()