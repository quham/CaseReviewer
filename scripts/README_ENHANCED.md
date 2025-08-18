# Enhanced PDF to Pinecone Pipeline with LLM Information Extraction

This enhanced version of the script now includes intelligent information extraction using OpenRouter LLM calls to extract structured data from NSPCC case review PDFs.

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

### 💾 Enhanced Data Storage
- Structured information stored in Pinecone metadata
- JSON files saved locally for each processed PDF
- Rich metadata for better search and analysis

### 🔍 Advanced Querying
- Query by structured criteria (agencies, recommendations, risk factors)
- Semantic search with enhanced context
- Timeline-based analysis

## Environment Variables Required

```bash
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=nspcc

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# OpenRouter Configuration (for LLM calls)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## Installation

```bash
pip install pypdf2 langchain-text-splitters scikit-learn pinecone python-dotenv requests sentence-transformers torch

```

## Usage

### Process a Single PDF
```python
processor = PDFToPineconeProcessor()
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

### Pinecone Metadata
Each chunk now includes:
- Case summary
- Agencies involved
- Key recommendations
- Risk factors
- Case outcomes
- Processing timestamp

### Local JSON Files
Structured information saved to `extracted_data/` directory:
- `{filename}_structured_info.json`

## LLM Model Used
- **Model**: `anthropic/claude-3.5-sonnet` (via OpenRouter)
- **Temperature**: 0.1 (for consistent, factual output)
- **Max Tokens**: 2000
- **Task Type**: Structured information extraction

## Error Handling
- Graceful fallback when LLM extraction fails
- Detailed logging of extraction process
- Continues processing even if individual extractions fail

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
- Structured data cached in Pinecone for fast retrieval


