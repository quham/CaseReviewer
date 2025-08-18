# PDF to Pinecone Pipeline with Google Gemini Embeddings
# Step-by-step script for processing NSPCC case reviews

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import hashlib
import requests
from datetime import datetime

# Required installations:
# pip install pypdf2 langchain-text-splitters google-genai scikit-learn pinecone python-dotenv requests

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PDFToPineconeProcessor:
    def __init__(self):
        """Initialize the processor with Google Gemini embeddings and Pinecone connection"""
        self.setup_environment()
        self.setup_gemini_client()
        self.setup_pinecone()
        self.setup_text_splitter()
        self.setup_openrouter()
    
    def setup_environment(self):
        """Setup environment variables"""
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'nspcc')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.openrouter_base_url = os.getenv('OPENROUTER_BASE_URL', 'https://openrouter.ai/api/v1')
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    def setup_openrouter(self):
        """Initialize OpenRouter connection for LLM calls"""
        print("Setting up OpenRouter connection...")
        self.openrouter_headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",  # Update with your repo
            "X-Title": "NSPCC Case Review Processor"
        }
        print("OpenRouter connection initialized successfully")
    
    def setup_gemini_client(self):
        """Initialize Google Gemini client for embeddings"""
        print("Setting up Google Gemini embeddings...")
        # genai.configure(api_key=self.gemini_api_key)
        self.client = genai.Client()
        print("Gemini client initialized successfully")
    
    def setup_pinecone(self):
        """Initialize Pinecone connection and create index if needed"""
        print("Setting up Pinecone connection...")
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Check if index exists, create if not
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=3072,  # Gemini embeddings default to 768 dimensions
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to index: {self.index_name}")
    
    def setup_text_splitter(self):
        """Setup RecursiveCharacterTextSplitter with optimal settings for case reviews"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,  # Based on our earlier analysis
            chunk_overlap=150,  # 15% overlap
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
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        print(f"Extracting text from: {pdf_path}")
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                except Exception as e:
                    print(f"Error extracting page {page_num + 1}: {str(e)}")
                    continue
        
        print(f"Extracted {len(text)} characters from {len(pdf_reader.pages)} pages")
        return text
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using Google Gemini API"""
        print(f"Creating embeddings for {len(texts)} text chunks...")
        
        # Process in batches to avoid API limits
        batch_size = 10  # Gemini API can handle multiple texts at once
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                result = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=3072
                    )
                )
                
                # Convert embeddings to lists
                batch_embeddings = [list(e.values) for e in result.embeddings]
                all_embeddings.extend(batch_embeddings)
                
                print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                print(f"Error creating embeddings for batch {i//batch_size + 1}: {str(e)}")
                # Fallback: create empty embeddings for failed batch
                all_embeddings.extend([[0.0] * 3072] * len(batch))
        
        print(f"Generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def create_query_embedding(self, query_text: str) -> List[float]:
        """Create embedding for search queries using RETRIEVAL_QUERY task type"""
        print(f"Creating query embedding for: '{query_text[:100]}...'")
        
        try:
            result = self.client.models.embed_content(
                model="gemini-embedding-001",
                contents=query_text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=3072
                )
            )
            
            # Return the first (and only) embedding as a list
            query_embedding = list(result.embeddings[0].values)
            print("Query embedding created successfully")
            return query_embedding
            
        except Exception as e:
            print(f"Error creating query embedding: {str(e)}")
            # Fallback: return zero vector
            return [0.0] * 3072
    
    def generate_chunk_id(self, text: str, source: str, chunk_index: int) -> str:
        """Generate unique ID for each chunk"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{Path(source).stem}_{chunk_index}_{content_hash}"
    
    def extract_structured_information(self, pdf_text: str, pdf_filename: str) -> Dict[str, Any]:
        """Extract structured information from PDF text using OpenRouter LLM"""
        print(f"Extracting structured information from {pdf_filename}...")
        
        # Create a comprehensive prompt for information extraction
        prompt = f"""
        You are an expert social worker and case review analyst. Analyze the following NSPCC case review document and extract key information in a structured format.

        Document: {pdf_filename}

        Please provide the following information in JSON format:

        1. **Executive Summary** (2-3 sentences): A concise overview of the case and its key findings

        2. **Agencies Involved**: List all agencies, organizations, and services mentioned in the case review

        3. **Key Recommendations**: Extract all recommendations made in the case review

        4. **Timeline of Key Events**: Chronological list of significant events including:
           - Missed opportunities
           - Incidents and concerns
           - What went wrong
           - What could have been done better
           - Positive events and good practice
           - Dates (if available)

        5. **Risk Factors Identified**: Any risk factors mentioned in the case

        6. **Outcomes**: What happened to the child/family and any lessons learned

        Return ONLY valid JSON with this exact structure:
        {{
            "summary": "executive summary here",
            "agencies": ["agency1", "agency2"],
            "recommendations": ["rec1", "rec2"],
            "timeline": [
                {{
                    "date": "date if available",
                    "event": "description of event",
                    "type": "missed_opportunity|incident|concern|positive|improvement"
                }}
            ],
            "risk_factors": ["factor1", "factor2"],
            "outcomes": "outcomes description"
        }}

        Document text to analyze:
        {pdf_text[:8000]}  # Limit text length for API efficiency
        """
        
        try:
            # Call OpenRouter API
            response = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=self.openrouter_headers,
                json={
                    "model": "meta-llama/llama-3.3-70b-instruct:free",  # High-quality model for analysis
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.1,  # Low temperature for consistent, factual output
                    "max_tokens": 2000
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Try to extract JSON from the response
                try:
                    # Find JSON content in the response
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
                    print(f"Raw response: {content[:500]}...")
                    # Return fallback structured data
                    return self._create_fallback_structured_data(pdf_filename)
                    
            else:
                print(f"❌ OpenRouter API error: {response.status_code}")
                print(f"Response: {response.text}")
                return self._create_fallback_structured_data(pdf_filename)
                
        except Exception as e:
            print(f"❌ Error calling OpenRouter API: {str(e)}")
            return self._create_fallback_structured_data(pdf_filename)
    
    def _create_fallback_structured_data(self, pdf_filename: str) -> Dict[str, Any]:
        """Create fallback structured data when LLM extraction fails"""
        return {
            "summary": f"Case review document: {pdf_filename}",
            "agencies": ["Information extraction failed"],
            "recommendations": ["Information extraction failed"],
            "timeline": [
                {
                    "date": "Unknown",
                    "event": "Information extraction failed",
                    "type": "unknown"
                }
            ],
            "risk_factors": ["Information extraction failed"],
            "outcomes": "Information extraction failed"
        }
    
    def prepare_metadata(self, pdf_path: str, chunk_text: str, chunk_index: int, 
                        structured_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare enhanced metadata for each chunk"""
        metadata = {
            "source_file": Path(pdf_path).name,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text[:1000],  # Store first 1000 chars for reference
            "text_length": len(chunk_text),
            "file_type": "nspcc_case_review",
            "processing_date": datetime.now().isoformat()
        }
        
        # Add structured information if available
        if structured_info:
            metadata.update({
                "case_summary": structured_info.get("summary", ""),
                "agencies_involved": structured_info.get("agencies", []),
                "key_recommendations": structured_info.get("recommendations", []),
                "risk_factors": structured_info.get("risk_factors", []),
                "case_outcomes": structured_info.get("outcomes", ""),
                "has_structured_data": True
            })
        else:
            metadata["has_structured_data"] = False
        
        return metadata
    
    def upsert_to_pinecone(self, embeddings: List[List[float]], 
                          texts: List[str], pdf_path: str, structured_info: Dict[str, Any] = None) -> None:
        """Upload embeddings to Pinecone"""
        print("Uploading to Pinecone...")
        
        vectors = []
        for i, (embedding, text) in enumerate(zip(embeddings, texts)):
            chunk_id = self.generate_chunk_id(text, pdf_path, i)
            metadata = self.prepare_metadata(pdf_path, text, i, structured_info)
            
            vectors.append({
                'id': chunk_id,
                'values': embedding,
                'metadata': metadata
            })
        
        # Upload in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"Successfully uploaded {len(vectors)} vectors to Pinecone")
    
    def process_pdf(self, pdf_path: str) -> None:
        """Complete pipeline to process a single PDF"""
        print(f"\n{'='*50}")
        print(f"Processing PDF: {pdf_path}")
        print(f"{'='*50}")
        
        # Step 1: Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Extract structured information using LLM
        print("Extracting structured information using LLM...")
        structured_info = self.extract_structured_information(text, Path(pdf_path).name)
        print("Structured info: ", structured_info)
        
        # Step 3: Split text into chunks
        print("Splitting text into chunks...")
        chunks = self.text_splitter.split_text(text)
        print(f"Created {len(chunks)} chunks")
        
        # Step 4: Create embeddings
        embeddings = self.create_embeddings(chunks)
        
        # Step 5: Upload to Pinecone with structured information
        self.upsert_to_pinecone(embeddings, chunks, pdf_path, structured_info)
        
        # Step 6: Save structured information to file
        self.save_structured_info_to_file(structured_info, pdf_path)
        
        print(f"✅ Successfully processed {Path(pdf_path).name}")
        print(f"📊 Extracted summary: {structured_info.get('summary', 'N/A')[:100]}...")
    
    def process_directory(self, directory_path: str) -> None:
        """Process all PDFs in a directory"""
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_path in pdf_files:
            try:
                self.process_pdf(str(pdf_path))
            except Exception as e:
                print(f"❌ Error processing {pdf_path.name}: {str(e)}")
                continue
    
    def query_similar_documents(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Test function to query similar documents"""
        print(f"\nQuerying: '{query_text[:100]}...'")
        
        # Create embedding for query using RETRIEVAL_QUERY task type
        query_embedding = self.create_query_embedding(query_text)
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        print(results)
        
        print(f"Found {len(results.matches)} similar documents:")
        for i, match in enumerate(results.matches):
            print(f"{i+1}. Score: {match.score:.3f} - {match.metadata['source_file']}")
        
        return results.matches
    
    def save_structured_info_to_file(self, structured_info: Dict[str, Any], pdf_path: str) -> None:
        """Save extracted structured information to a JSON file"""
        output_dir = Path("extracted_data")
        output_dir.mkdir(exist_ok=True)
        
        filename = Path(pdf_path).stem
        output_file = output_dir / f"{filename}_structured_info.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(structured_info, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Structured information saved to: {output_file}")
    
    def query_by_structured_criteria(self, criteria: Dict[str, Any], top_k: int = 5) -> List[Dict]:
        """Query documents based on structured criteria (agencies, recommendations, etc.)"""
        print(f"\nQuerying by structured criteria: {criteria}")
        
        # Create a text query from the criteria
        query_parts = []
        if criteria.get('agencies'):
            query_parts.append(f"agencies: {', '.join(criteria['agencies'])}")
        if criteria.get('recommendations'):
            query_parts.append(f"recommendations: {', '.join(criteria['recommendations'])}")
        if criteria.get('risk_factors'):
            query_parts.append(f"risk factors: {', '.join(criteria['risk_factors'])}")
        
        query_text = " | ".join(query_parts) if query_parts else str(criteria)
        
        return self.query_similar_documents(query_text, top_k)
    
    def display_structured_info(self, structured_info: Dict[str, Any]) -> None:
        """Display extracted structured information in a readable format"""
        print(f"\n{'='*60}")
        print("📋 EXTRACTED STRUCTURED INFORMATION")
        print(f"{'='*60}")
        
        print(f"\n📝 Executive Summary:")
        print(f"   {structured_info.get('summary', 'N/A')}")
        
        print(f"\n🏢 Agencies Involved:")
        agencies = structured_info.get('agencies', [])
        if agencies:
            for agency in agencies:
                print(f"   • {agency}")
        else:
            print("   None identified")
        
        print(f"\n💡 Key Recommendations:")
        recommendations = structured_info.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print("   None identified")
        
        print(f"\n⚠️ Risk Factors:")
        risk_factors = structured_info.get('risk_factors', [])
        if risk_factors:
            for factor in risk_factors:
                print(f"   • {factor}")
        else:
            print("   None identified")
        
        print(f"\n📅 Timeline of Key Events:")
        timeline = structured_info.get('timeline', [])
        if timeline:
            for event in timeline:
                date = event.get('date', 'Unknown date')
                event_desc = event.get('event', 'No description')
                event_type = event.get('type', 'unknown')
                print(f"   • [{date}] {event_desc} ({event_type})")
        else:
            print("   None identified")
        
        print(f"\n🎯 Outcomes:")
        print(f"   {structured_info.get('outcomes', 'N/A')}")
        
        print(f"\n{'='*60}")


# Usage Scripts
def main():
    """Main execution function"""
    
    # Initialize processor
    processor = PDFToPineconeProcessor()
    
    # Option 1: Process single PDF file
    # pdf_path = "/Users/quhamadefila/Desktop/childrens_sw/nspcc_case_reviews/2025LambethMaraCSPRSummaryReport.pdf"
    pdf_path = "/Users/quhamadefila/Desktop/childrens_sw/nspcc_case_reviews/2023EnfieldAndreCSPR.pdf"
    processor.process_pdf(pdf_path)
    
    # Option 2: Process all PDFs in directory
    # pdf_directory = "./nspcc_pdfs/"  # Change to your PDF directory
    # processor.process_directory(pdf_directory)
    
    # Option 3: Test query functionality
    # test_query = "Mara is from a minority ethnic background and has Autism"
    # processor.query_similar_documents(test_query)
    # test_query = "James is from a minority ethnic background and has Autism"
    # processor.query_similar_documents(test_query)
    
    # Option 4: Test structured querying
    # processor.query_by_structured_criteria({
    #     "agencies": ["Social Services", "Police"],
    #     "risk_factors": ["domestic violence"]
    # })
    
    # Option 5: Test structured information extraction on sample text
    # print("\n🧪 Testing structured information extraction...")
    # sample_text = """
    # This case review examines the circumstances surrounding the death of Child A, aged 3 years, 
    # who died from injuries sustained in a domestic violence incident. The review identified 
    # multiple missed opportunities for intervention by Social Services, Police, and Health services. 
    # Key recommendations include improved inter-agency communication, mandatory domestic violence 
    # training for all frontline staff, and enhanced risk assessment procedures. The timeline shows 
    # escalating concerns over 6 months prior to the incident, with 12 separate reports to agencies 
    # that were not adequately followed up.
    # """
    
    # sample_structured_info = processor.extract_structured_information(sample_text, "Sample_Case_Review")
    # processor.display_structured_info(sample_structured_info)


if __name__ == "__main__":
    main()