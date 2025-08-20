#!/bin/bash

# Setup script for CaseReviewer with Qwen3-Embedding-8B
# This script sets up the Python environment and installs dependencies

echo "🚀 Setting up CaseReviewer environment with Qwen3-Embedding-8B..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
if [ -z "$python_version" ]; then
    echo "❌ Python 3.8+ is required but not found"
    echo "Please install Python 3.8 or higher and try again"
    exit 1
fi

echo "✅ Python $python_version found"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not found"
    echo "Please install pip3 and try again"
    exit 1
fi

echo "✅ pip3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔄 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "🔄 Installing dependencies..."
echo "   This may take several minutes for the first time..."

# Install PyTorch first (CPU version for compatibility)
echo "   Installing PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "   Installing other dependencies..."
pip install -r requirements.txt

echo "✅ Dependencies installed successfully"

# Test the installation
echo "🧪 Testing the installation..."

# Test sentence-transformers
python3 -c "from sentence_transformers import SentenceTransformer; print('✅ sentence-transformers imported successfully')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ sentence-transformers working correctly"
else
    echo "❌ sentence-transformers test failed"
    exit 1
fi

# Test transformers
python3 -c "from transformers import AutoTokenizer; print('✅ transformers imported successfully')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ transformers working correctly"
else
    echo "❌ transformers test failed"
    exit 1
fi

echo ""
echo "🎉 Environment setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. Set up your environment variables (DATABASE_URL, OPENROUTER_API_KEY)"
echo "   2. Test the embedding model: python test_qwen_embedding.py"
echo "   3. Run the main pipeline: python db_setup_postgresql.py"
echo ""
echo "💡 Note: The first time you run the script, it will download the"
echo "   Qwen3-Embedding-8B model (~7.57GB) which may take some time."
echo ""
echo "🔧 If you encounter any issues, check the requirements.txt file"
echo "   and ensure all dependencies are properly installed."
