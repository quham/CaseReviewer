#!/usr/bin/env python3
"""
Test script for DeepInfra API integration
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def test_deepinfra_embeddings():
    """Test DeepInfra API for embeddings"""
    print("🔍 Testing DeepInfra API integration...")
    
    # Check for token
    deepinfra_token = os.getenv("DEEPINFRA_TOKEN")
    if not deepinfra_token:
        print("❌ DEEPINFRA_TOKEN not found in environment variables")
        print("Please set DEEPINFRA_TOKEN in your .env file")
        return False
    
    try:
        # Initialize client
        client = OpenAI(
            api_key=deepinfra_token,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        
        print("✅ DeepInfra client initialized successfully")
        
        # Test text
        test_text = "This is a test case for child protection"
        print(f"📝 Test text: {test_text}")
        
        # Create embedding
        print("🔄 Creating embedding...")
        result = client.embeddings.create(
            model="Qwen/Qwen3-Embedding-8B",
            input=test_text,
            encoding_format="float"
        )
        
        # Check result
        embedding = result.data[0].embedding
        print(f"✅ Embedding created successfully!")
        print(f"   Dimensions: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   Usage tokens: {result.usage.prompt_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing DeepInfra API: {e}")
        return False

def test_batch_embeddings():
    """Test batch embedding creation"""
    print("\n🔍 Testing batch embeddings...")
    
    deepinfra_token = os.getenv("DEEPINFRA_TOKEN")
    if not deepinfra_token:
        print("❌ DEEPINFRA_TOKEN not found")
        return False
    
    try:
        client = OpenAI(
            api_key=deepinfra_token,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        
        # Test multiple texts
        texts = [
            "Child protection case review",
            "Social work intervention",
            "Risk assessment methodology"
        ]
        
        print(f"📝 Creating embeddings for {len(texts)} texts...")
        
        result = client.embeddings.create(
            model="Qwen/Qwen3-Embedding-8B",
            input=texts,
            encoding_format="float"
        )
        
        print(f"✅ Batch embeddings created successfully!")
        print(f"   Total embeddings: {len(result.data)}")
        print(f"   Usage tokens: {result.usage.prompt_tokens}")
        
        for i, data in enumerate(result.data):
            print(f"   Text {i+1}: {len(data.embedding)} dimensions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing batch embeddings: {e}")
        return False

if __name__ == "__main__":
    print("🚀 DeepInfra API Integration Test")
    print("=" * 40)
    
    # Test single embedding
    success1 = test_deepinfra_embeddings()
    
    # Test batch embeddings
    success2 = test_batch_embeddings()
    
    print("\n" + "=" * 40)
    if success1 and success2:
        print("🎉 All tests passed! DeepInfra integration is working correctly.")
    else:
        print("❌ Some tests failed. Please check your configuration.")
    
    print("\n💡 Next steps:")
    print("1. Ensure DEEPINFRA_TOKEN is set in your .env file")
    print("2. Run the server: python start_server.py")
    print("3. Test the API endpoints: python test_server.py")

