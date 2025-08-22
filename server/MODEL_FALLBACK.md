# AI Model Fallback Mechanism

## Overview

The CaseReviewer server now includes an intelligent model fallback mechanism that automatically tries multiple AI models in order of preference when generating insights and recommendations. This ensures higher reliability and better performance by not relying on a single model.

## How It Works

### 1. Model Priority Order

The system tries models in this specific order:

1. **meta-llama/llama-3.3-70b-instruct:free** (Primary)
2. **deepseek/deepseek-chat-v3-0324:free**
3. **deepseek/deepseek-r1-0528:free**
4. **google/gemini-2.0-flash-exp:free**
5. **anthropic/claude-3-haiku:free**
6. **qwen/qwen3-8b:free**
7. **deepseek/deepseek-r1-0528-qwen3-8b:free**
8. **mistralai/mistral-7b-instruct:free**

### 2. Automatic Fallback

When a request is made:
- The system starts with the first model in the list
- If that model fails (API error, timeout, etc.), it automatically tries the next one
- This continues until a successful response is received
- If all models fail, the system falls back to basic pattern analysis

### 3. Response Tracking

Each successful response includes metadata showing:
- Which model was actually used
- Timestamp of generation
- Source of the insights

## Configuration

### Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your_api_key_here

# Optional (defaults shown)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
```

### Model Selection

The `OPENROUTER_MODEL` environment variable sets the primary model, but the fallback list is hardcoded in the service for optimal reliability.

## API Endpoints

### Get Model Status
```
GET /api/models/status
```
Returns current model configuration and status.

### Test All Models
```
GET /api/models/test
```
Tests all models and returns their working status.

## Testing

Run the test script to verify the fallback mechanism:

```bash
cd server
python test_model_fallback.py
```

This will:
1. Test each model individually
2. Verify the fallback mechanism works
3. Show which models are working/failing
4. Generate sample recommendations

## Benefits

1. **Higher Reliability**: If one model is down, others can take over
2. **Better Performance**: Some models may be faster for certain types of requests
3. **Cost Optimization**: Free models are tried first
4. **Transparency**: Users can see which model generated their insights
5. **Automatic Recovery**: No manual intervention needed when models fail

## Monitoring

The system logs all model attempts:
- 🔄 When trying a new model
- ✅ When a model succeeds
- ⚠️ When a model fails
- ❌ When all models fail

## Fallback Behavior

If all AI models fail, the system provides:
- Basic pattern analysis from case data
- Risk factor identification
- Agency coordination recommendations
- Professional development suggestions

This ensures users always get some value, even when AI services are unavailable.
