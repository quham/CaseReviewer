#!/usr/bin/env python3
"""
Startup script for CaseReviewer Python Server
Handles environment setup and server initialization
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["DATABASE_URL"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these in your .env file or environment")
        sys.exit(1)
    
    logger.info("✅ Environment variables configured")

def install_dependencies():
    """Install Python dependencies if needed"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        logger.error("requirements.txt not found")
        sys.exit(1)
    
    try:
        logger.info("Installing Python dependencies...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True)
        logger.info("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        logger.error("Please install manually: pip install -r requirements.txt")
        sys.exit(1)

def check_database_connection():
    """Test database connection"""
    try:
        import psycopg2
        from dotenv import load_dotenv
        
        load_dotenv()
        database_url = os.getenv("DATABASE_URL")
        
        if not database_url:
            logger.error("DATABASE_URL not found")
            return False
        
        # Test connection
        conn = psycopg2.connect(database_url)
        conn.close()
        logger.info("✅ Database connection successful")
        return True
        
    except ImportError:
        logger.error("psycopg2 not available. Please install dependencies first.")
        return False
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    try:
        logger.info("Starting CaseReviewer Python Server...")
        
        # Set default port if not specified
        port = os.getenv("PORT", "5000")
        
        # Start server with uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app",
            "--host", "0.0.0.0",
            "--port", port,
            "--reload"
        ], check=True)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    logger.info("🚀 Starting CaseReviewer Python Server Setup...")
    
    # Check prerequisites
    check_python_version()
    check_environment()
    
    # Install dependencies if needed
    try:
        import fastapi
        import uvicorn
        logger.info("✅ Dependencies already installed")
    except ImportError:
        logger.info("Installing dependencies...")
        install_dependencies()
    
    # Check database connection
    if not check_database_connection():
        logger.error("❌ Database connection failed. Please check your configuration.")
        sys.exit(1)
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
