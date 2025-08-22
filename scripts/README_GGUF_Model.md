# Qwen3-Embedding-8B-Q8_0.gguf Model Integration

## Overview

This script has been updated to use the **Qwen3-Embedding-8B-Q8_0.gguf** model from the [JonathanMiddleton/Qwen3-Embedding-8B-GGUF](https://huggingface.co/JonathanMiddleton/Qwen3-Embedding-8B-GGUF) repository instead of the original Hugging Face model.

## Key Changes

### 1. Model Switch
- **From**: `Qwen/Qwen3-Embedding-8B` (Hugging Face)
- **To**: `JonathanMiddleton/Qwen3-Embedding-8B-GGUF` (GGUF format)

### 2. Library Change
- **From**: `sentence-transformers` library
- **To**: `llama-cpp-python` library (for GGUF support)

### 3. Model Specifications
- **Model**: Qwen3-Embedding-8B-Q8_0.gguf
- **Precision**: Q8_0 (8-bit quantization)
- **Size**: ~8.6 GB (vs 15.1 GB for FP16)
- **Quality**: ≈ +0.02 MTEB delta vs FP16 (excellent quality retention)
- **Embedding Dimensions**: 8192
- **Context Length**: 32k tokens

## Benefits of the New Model

1. **Smaller Size**: 8.6 GB vs 15.1 GB (43% reduction)
2. **Faster Loading**: Quantized model loads faster
3. **Lower Memory Usage**: Reduced RAM requirements
4. **Maintained Quality**: Minimal quality loss (+0.02 MTEB delta)
5. **Better CPU Performance**: Optimized for CPU inference

## Installation

### Option 1: Use the Installation Script
```bash
cd scripts
./install_dependencies.sh
```

### Option 2: Manual Installation
```bash
# Activate virtual environment
source venv_py311/bin/activate

# Install dependencies
pip install pypdf2 langchain-text-splitters scikit-learn python-dotenv requests llama-cpp-python huggingface-hub psycopg2-binary torch psutil
```

## Setup Steps

1. **Install Dependencies**: Run the installation script or manually install packages
2. **Login to Hugging Face**: 
   ```bash
   huggingface-cli login
   ```
3. **Run the Script**: 
   ```bash
   python db_setup_postgresql.py
   ```

## First Run

On the first run, the script will:
1. Download the ~8.6 GB Qwen3-Embedding-8B-Q8_0.gguf model
2. Cache it in the `./models` directory
3. Initialize the llama-cpp-python embedding engine

## Model Download Details

- **Repository**: JonathanMiddleton/Qwen3-Embedding-8B-GGUF
- **File**: Qwen3-Embedding-8B-Q8_0.gguf
- **Size**: 8.6 GB
- **License**: Apache-2.0
- **Source**: Converted from Qwen/Qwen3-Embedding-8B

## Performance Notes

- **CPU Usage**: Optimized for CPU inference with multi-threading
- **GPU Support**: Automatic GPU layer detection for CUDA/MPS
- **Memory**: Lower memory footprint compared to FP16 version
- **Speed**: Faster inference due to quantization

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Ensure you're logged into Hugging Face: `huggingface-cli login`
   - Check internet connection and available disk space

2. **llama-cpp-python Installation Issues**
   - On macOS: `pip install llama-cpp-python --no-cache-dir`
   - On Linux: May need to install build dependencies

3. **Memory Issues**
   - The Q8_0 model uses ~8.6 GB vs 15.1 GB, but still requires sufficient RAM
   - Consider using CPU-only mode if GPU memory is limited

### Support

For issues with the GGUF model specifically, refer to:
- [JonathanMiddleton/Qwen3-Embedding-8B-GGUF](https://huggingface.co/JonathanMiddleton/Qwen3-Embedding-8B-GGUF)
- [llama-cpp-python documentation](https://github.com/abetlen/llama-cpp-python)

## Migration Notes

If you were previously using the sentence-transformers version:
- Embeddings will be compatible (same 8192 dimensions)
- Database schema remains unchanged
- Existing embeddings can be regenerated if needed
- Performance should improve due to quantization
