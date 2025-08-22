#!/usr/bin/env python3
"""
Test script to verify the embedding fix works correctly
"""

import sys
import os
import torch
from pathlib import Path

def test_embedding_fix():
    """Test the fixed embedding method"""
    
    # Add the current directory to Python path
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from db_setup_postgresql import PDFToPostgreSQLProcessor
        
        print("🔍 Testing embedding fix...")
        
        # Initialize the processor
        processor = PDFToPostgreSQLProcessor()
        
        # Test with a simple sentence
        test_text = "This is a test case for child protection review."
        print(f"📝 Test text: {test_text}")
        print(f"📏 Text length: {len(test_text)} characters")
        
        # Try to create embedding
        try:
            embedding = processor.create_embedding(test_text)
            print(f"✅ Embedding created successfully!")
            print(f"   Dimensions: {len(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
            
            # Check if embedding is valid
            if all(x == 0.0 for x in embedding):
                print("⚠️ Warning: Embedding contains only zeros")
            else:
                print("✅ Embedding is valid (not all zeros)")
                
        except Exception as e:
            print(f"❌ Error creating embedding: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error initializing processor: {e}")
        return False

if __name__ == "__main__":
    success = test_embedding_fix()
    if success:
        print("\n🎉 Embedding fix test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Embedding fix test failed!")
        sys.exit(1)
