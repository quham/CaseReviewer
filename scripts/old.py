# PDF to Pinecone Pipeline with Qwen-8B Embeddings
# Step-by-step script for processing NSPCC case reviews

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import hashlib

# Required installations:
# pip install pypdf2 langchain-text-splitters transformers torch pinecone-client python-dotenv

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch
import pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class PDFToPineconeProcessor:
    def __init__(self):
        """Initialize the processor with Qwen model and Pinecone connection"""
        self.setup_environment()
        self.setup_qwen_model()
        self.setup_pinecone()
        self.setup_text_splitter()
    
    def setup_environment(self):
        """Setup environment variables"""
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1-aws')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'nspcc-case-reviews')
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    def setup_qwen_model(self):
        """Initialize Qwen3-Embedding-8B model"""
        print("Loading Qwen3-Embedding-8B model...")
        model_name = "Qwen/Qwen3-Embedding-8B"
        
        # Initialize tokenizer with left padding for proper embedding generation
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(model_name)
        
        # Get EOD token ID for proper tokenization
        self.eod_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        self.max_length = 8192  # Qwen3-Embedding-8B supports up to 32k, but 8k is more practical
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Qwen3-Embedding-8B loaded on device: {self.device}")
        
        # Optional: Enable flash_attention_2 for better performance if available
        # Uncomment if you have flash_attention_2 installed:
        # self.model = AutoModel.from_pretrained(
        #     model_name, 
        #     attn_implementation="flash_attention_2", 
        #     torch_dtype=torch.float16
        # ).cuda()
    
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
                dimension=4096,  # Adjust based on Qwen-8B embedding dimension
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
        """Create embeddings using Qwen model"""
        print(f"Creating embeddings for {len(texts)} text chunks...")
        embeddings = []
        
        batch_size = 8  # Adjust based on GPU memory
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                max_length=512,  # Adjust based on model limits
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding or mean pooling
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings.extend(batch_embeddings.cpu().numpy().tolist())
        
        print(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def generate_chunk_id(self, text: str, source: str, chunk_index: int) -> str:
        """Generate unique ID for each chunk"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        return f"{Path(source).stem}_{chunk_index}_{content_hash}"
    
    def prepare_metadata(self, pdf_path: str, chunk_text: str, chunk_index: int) -> Dict[str, Any]:
        """Prepare metadata for each chunk"""
        return {
            "source_file": Path(pdf_path).name,
            "source_path": pdf_path,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text[:1000],  # Store first 1000 chars for reference
            "text_length": len(chunk_text),
            "file_type": "nspcc_case_review"
        }
    
    def upsert_to_pinecone(self, embeddings: List[List[float]], 
                          texts: List[str], pdf_path: str) -> None:
        """Upload embeddings to Pinecone"""
        print("Uploading to Pinecone...")
        
        vectors = []
        for i, (embedding, text) in enumerate(zip(embeddings, texts)):
            chunk_id = self.generate_chunk_id(text, pdf_path, i)
            metadata = self.prepare_metadata(pdf_path, text, i)
            
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
        
        # Step 2: Split text into chunks
        print("Splitting text into chunks...")
        chunks = self.text_splitter.split_text(text)
        print(f"Created {len(chunks)} chunks")
        
        # Step 3: Create embeddings
        embeddings = self.create_embeddings(chunks)
        
        # Step 4: Upload to Pinecone
        self.upsert_to_pinecone(embeddings, chunks, pdf_path)
        
        print(f"✅ Successfully processed {Path(pdf_path).name}")
    
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
        
        # Create embedding for query
        query_embedding = self.create_embeddings([query_text])[0]
        
        # Search Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        print(f"Found {len(results.matches)} similar documents:")
        for i, match in enumerate(results.matches):
            print(f"{i+1}. Score: {match.score:.3f} - {match.metadata['source_file']}")
        
        return results.matches


# Usage Scripts
def main():
    """Main execution function"""
    
    # Initialize processor
    processor = PDFToPineconeProcessor()
    
    # Option 1: Process single PDF file
    # pdf_path = "/path/to/your/nspcc_case_review.pdf"
    # processor.process_pdf(pdf_path)
    
    # Option 2: Process all PDFs in directory
    pdf_directory = "./nspcc_pdfs/"  # Change to your PDF directory
    processor.process_directory(pdf_directory)
    
    # Option 3: Test query functionality
    test_query = "domestic violence and child protection concerns"
    processor.query_similar_documents(test_query)


if __name__ == "__main__":
    main()