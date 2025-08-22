#!/bin/bash

# Installation script for PDF to PostgreSQL Pipeline with Qwen3-Embedding-8B-Q8_0.gguf
echo "🚀 Installing dependencies for PDF to PostgreSQL Pipeline..."

# Check if Python virtual environment exists
if [ ! -d "venv_py311" ]; then
    echo "📦 Creating Python virtual environment..."
    python3.11 -m venv venv_py311
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv_py311/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install pypdf2 langchain-text-splitters scikit-learn python-dotenv requests psycopg2-binary

# Install embedding model dependencies
echo "🤖 Installing embedding model dependencies..."
pip install llama-cpp-python huggingface-hub

# Install optional PyTorch (CPU version for compatibility)
echo "🔥 Installing PyTorch (CPU version)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install system utilities
echo "🔧 Installing system utilities..."
pip install psutil

echo "✅ All dependencies installed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Activate the virtual environment: source venv_py311/bin/activate"
echo "2. Login to Hugging Face: huggingface-cli login"
echo "3. Run the script: python db_setup_postgresql.py"
echo ""
echo "💡 Note: The first run will download the ~8.6 GB Qwen3-Embedding-8B-Q8_0.gguf model"
