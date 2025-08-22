#!/bin/bash

# CaseReviewer Python Server Startup Script
# This script sets up and starts the Python server

echo "🚀 Starting CaseReviewer Python Server..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8 or higher is required. Current version: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if requirements are installed
if ! python3 -c "import fastapi, uvicorn" &> /dev/null; then
    echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Check environment file
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Please copy env.template to .env and configure your settings."
    echo "   cp env.template .env"
    exit 1
fi

# Start the server
echo "🚀 Starting server..."
python3 start_server.py
