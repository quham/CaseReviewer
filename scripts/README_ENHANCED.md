# Enhanced PDF to PostgreSQL Pipeline with Qwen3-Embedding-8B and LLM Information Extraction

This enhanced version of the script now includes intelligent information extraction using OpenRouter LLM calls and the state-of-the-art Qwen3-Embedding-8B model for creating high-quality embeddings from NSPCC case review PDFs.

## New Features

### 🧠 LLM-Powered Information Extraction
- **Executive Summary**: Concise overview of each case
- **Agencies Involved**: All organizations and services mentioned
- **Key Recommendations**: Specific recommendations from the case review
- **Timeline of Key Events**: Chronological events including:
  - Missed opportunities
  - Incidents and concerns
  - What went wrong
  - What could have been done better
  - Positive events and good practice
- **Risk Factors**: Identified risk factors
- **Outcomes**: Case outcomes and lessons learned

### 🚀 Advanced Embedding Model
- **Qwen3-Embedding-8B**: State-of-the-art multilingual embedding model
- **4096 dimensions**: High-quality vector representations
- **32k context length**: Handles long documents effectively
- **100+ languages**: Multilingual support for diverse case materials
- **Fallback support**: Automatic fallback to reliable alternative models

### 💾 Enhanced Data Storage
- Structured information stored in PostgreSQL with pgvector
- JSON files saved locally for each processed PDF
- Rich metadata for better search and analysis
- Vector similarity search for finding related cases

### 🔍 Advanced Querying
- Query by structured criteria (agencies, recommendations, risk factors)
- Semantic search with enhanced context
- Timeline-based analysis

## Environment Variables Required

```bash
# PostgreSQL Configuration
DATABASE_URL=postgresql://username:password@host:port/database

# OpenRouter Configuration (for LLM calls)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## Installation

```bash
# Install from requirements file
pip install -r requirements.txt

# Or install manually
pip install pypdf2 langchain-text-splitters scikit-learn python-dotenv requests sentence-transformers>=2.7.0 transformers>=4.51.0 torch huggingface-hub psycopg2-binary
```

## Usage

### Process a Single PDF
```python
processor = PDFToPostgreSQLProcessor()
processor.process_pdf("path/to/case_review.pdf")
```

### Process Multiple PDFs
```python
processor.process_directory("./nspcc_pdfs/")
```

### Query by Structured Criteria
```python
# Find cases involving specific agencies
results = processor.query_by_structured_criteria({
    "agencies": ["Social Services", "Police"],
    "risk_factors": ["domestic violence"]
})
```

### Display Extracted Information
```python
structured_info = processor.extract_structured_information(text, filename)
processor.display_structured_info(structured_info)
```

## Output Structure

### PostgreSQL Database Schema
Each case review includes:
- Case summary and content
- Agencies involved
- Risk factors and types
- Timeline of events
- Vector embeddings for similarity search
- Processing timestamp

### Local JSON Files
Structured information saved to `extracted_data/` directory:
- `{filename}_structured_info.json`

## Models Used

### LLM Model (via OpenRouter)
- **Model**: Multiple models tried in order (Gemini, Llama, Qwen, Claude, Mistral)
- **Temperature**: 0.1 (for consistent, factual output)
- **Max Tokens**: 2000
- **Task Type**: Structured information extraction

### Embedding Model
- **Primary**: `Qwen/Qwen3-Embedding-8B`
- **Dimensions**: 4096
- **Context Length**: 32k tokens
- **Languages**: 100+ languages supported
- **Fallback**: `sentence-transformers/all-MiniLM-L6-v2` (1024 dimensions)

## Testing the Embedding Model

Before running the main pipeline, test the Qwen3-Embedding-8B model:

```bash
# Test the embedding model
python test_qwen_embedding.py

# This will verify:
# - Model can be loaded successfully
# - Embeddings are created correctly
# - Multilingual support works
# - Fallback model is available if needed
```

## Error Handling
- Graceful fallback when LLM extraction fails
- Detailed logging of extraction process
- Continues processing even if individual extractions fail
- Automatic fallback to alternative embedding models

## Benefits for Social Workers
1. **Quick Case Overview**: Get summaries without reading entire documents
2. **Pattern Recognition**: Identify common themes across cases
3. **Agency Accountability**: Track which organizations were involved
4. **Learning from Failures**: Focus on missed opportunities and improvements
5. **Evidence-Based Practice**: Access to structured case data for research

## Performance Considerations
- LLM calls add processing time (~30-60 seconds per PDF)
- Text limited to 8000 characters for API efficiency
- Batch processing for multiple PDFs
- Vector embeddings stored in PostgreSQL with pgvector for fast similarity search
- Qwen3-Embedding-8B provides high-quality 4096-dimensional embeddings


