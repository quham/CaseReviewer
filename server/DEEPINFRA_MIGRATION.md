# Migration from Hugging Face to DeepInfra API

## Overview

This document outlines the migration from Hugging Face API to DeepInfra API for the CaseReviewer server. The migration replaces the Hugging Face Inference API with DeepInfra's OpenAI-compatible API for generating embeddings.

## Changes Made

### 1. Dependencies Updated

**Before (requirements.txt):**
```
huggingface-hub>=0.20.0
```

**After (requirements.txt):**
```
openai>=1.0.0
```

### 2. Environment Variables

**Before (.env):**
```env
HF_TOKEN=your_huggingface_token_here
```

**After (.env):**
```env
DEEPINFRA_TOKEN=your_deepinfra_token_here
```

### 3. Code Changes

#### Client Initialization

**Before (main.py):**
```python
from huggingface_hub import InferenceClient

hf_client = InferenceClient(
    model="Qwen/Qwen3-Embedding-8B",
    token=hf_token,
)
```

**After (main.py):**
```python
from openai import OpenAI

deepinfra_client = OpenAI(
    api_key=deepinfra_token,
    base_url="https://api.deepinfra.com/v1/openai",
)
```

#### Embedding Creation

**Before (main.py):**
```python
result = hf_client.feature_extraction(
    text,
    model="Qwen/Qwen3-Embedding-8B",
)
embedding = result.tolist()
```

**After (main.py):**
```python
result = deepinfra_client.embeddings.create(
    model="Qwen/Qwen3-Embedding-8B",
    input=text,
    encoding_format="float"
)
embedding = result.data[0].embedding
```

### 4. Files Modified

- `server/main.py` - Core API client and embedding function
- `server/requirements.txt` - Dependencies
- `server/env.template` - Environment variables template
- `server/start_server.py` - Startup script
- `server/test_server.py` - Test script
- `server/README.md` - Documentation
- `server/TODO.txt` - Task list

### 5. New Files Created

- `server/test_deepinfra.py` - DeepInfra-specific test script
- `server/DEEPINFRA_MIGRATION.md` - This migration guide

## Benefits of DeepInfra

1. **OpenAI Compatibility**: Uses the standard OpenAI client library
2. **Better Reliability**: More stable API endpoints
3. **Cost Effective**: Competitive pricing for embeddings
4. **Same Model**: Still uses Qwen3-Embedding-8B model
5. **Better Error Handling**: More consistent error responses

## Setup Instructions

### 1. Install Dependencies
```bash
cd server
pip install -r requirements.txt
```

### 2. Set Environment Variables
Create a `.env` file in the server directory:
```env
DATABASE_URL=postgresql://username:password@localhost:5432/casereviewer
DEEPINFRA_TOKEN=your_deepinfra_token_here
JWT_SECRET=your_jwt_secret_here
PORT=5000
```

### 3. Test Integration
```bash
python test_deepinfra.py
```

### 4. Start Server
```bash
python start_server.py
```

## Testing

### Test DeepInfra Integration
```bash
python test_deepinfra.py
```

### Test Server Endpoints
```bash
python test_server.py
```

## Troubleshooting

### Common Issues

1. **DEEPINFRA_TOKEN not found**
   - Ensure the token is set in your `.env` file
   - Check that the file is in the correct directory

2. **API rate limits**
   - DeepInfra has rate limits; check your plan
   - Implement retry logic if needed

3. **Model not found**
   - Ensure you're using the correct model name: `Qwen/Qwen3-Embedding-8B`
   - Check DeepInfra's model availability

### Fallback Behavior

If DeepInfra API fails, the system will:
1. Log the error
2. Use a hash-based fallback embedding
3. Continue processing with reduced functionality

## Migration Checklist

- [x] Update dependencies
- [x] Replace Hugging Face client with OpenAI client
- [x] Update environment variables
- [x] Modify embedding function
- [x] Update startup scripts
- [x] Update test scripts
- [x] Update documentation
- [x] Test integration
- [x] Verify fallback behavior

## Rollback Plan

If you need to rollback to Hugging Face:

1. Revert `requirements.txt` to include `huggingface-hub>=0.20.0`
2. Revert `main.py` to use `InferenceClient`
3. Change environment variable back to `HF_TOKEN`
4. Update all references accordingly

## Support

For DeepInfra API issues:
- Check [DeepInfra Documentation](https://deepinfra.com/docs)
- Review API status at [DeepInfra Status](https://status.deepinfra.com)
- Contact DeepInfra support if needed

For CaseReviewer issues:
- Check the server logs for detailed error messages
- Review the test scripts for configuration issues
- Ensure all environment variables are properly set

