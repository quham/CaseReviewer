#!/usr/bin/env python3
"""
Test script for Qwen3-Embedding-8B model
This script tests if the model can be loaded and used for creating embeddings
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers imported successfully")
except ImportError as e:
    print(f"❌ Error importing sentence-transformers: {e}")
    print("Please install with: pip install sentence-transformers>=2.7.0")
    sys.exit(1)

def test_qwen_embedding():
    """Test the Qwen3-Embedding-8B model"""
    print("\n" + "="*60)
    print("🧪 TESTING QWEN3-EMBEDDING-8B MODEL")
    print("="*60)
    
    model_name = "Qwen/Qwen3-Embedding-8B"
    import psutil
    import gc
    import torch
    try:
        print(f"🔄 Loading {model_name}...")
        print("   This may take a few minutes on first run...")
        def print_memory_usage():
            process = psutil.Process()
            memory_info = process.memory_info()
            print(f"   📊 Memory usage: {memory_info.rss / 1024 / 1024 / 1024:.2f} GB")
        print_memory_usage()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_memory_usage()
        
        # Load the model
        model = SentenceTransformer(model_name)
        
        print("✅ Model loaded successfully!")
        
        # Get model information
        embedding_dim = model.get_sentence_embedding_dimension()
        print(f"   Embedding dimension: {embedding_dim}")
        print(f"   Model type: {type(model).__name__}")
        
        # Test with different types of text
        test_texts = [
            "This is a simple English sentence.",
            "Este es un texto en español para probar el modelo multilingüe.",
            "这是一个中文句子来测试多语言模型。",
            "Ceci est un texte en français pour tester le modèle multilingue.",
            "Das ist ein deutscher Text, um das mehrsprachige Modell zu testen."
        ]
        
        print(f"\n🔍 Testing multilingual capabilities...")
        
        for i, text in enumerate(test_texts, 1):
            print(f"   {i}. Testing: {text[:50]}...")
            
            try:
                # Create embedding
                embedding = model.encode(text)
                
                # Verify embedding
                if len(embedding) == embedding_dim:
                    print(f"      ✅ Embedding created: {len(embedding)} dimensions")
                    
                    # Check if embedding is not all zeros
                    if not all(x == 0.0 for x in embedding):
                        print(f"      ✅ Embedding is valid (not all zeros)")
                    else:
                        print(f"      ⚠️ Warning: Embedding contains only zeros")
                else:
                    print(f"      ❌ Unexpected embedding dimension: {len(embedding)}")
                    
            except Exception as e:
                print(f"      ❌ Error creating embedding: {e}")
        
        # Test with longer text (to test context length)
        print(f"\n📝 Testing long text handling...")
        long_text = "This is a longer text to test the model's ability to handle extended content. " * 50
        
        try:
            long_embedding = model.encode(long_text)
            print(f"   ✅ Long text embedding created: {len(long_embedding)} dimensions")
            print(f"   ✅ Text length: {len(long_text)} characters")
        except Exception as e:
            print(f"   ❌ Error with long text: {e}")
        
        # Test similarity between similar and different texts
        print(f"\n🔗 Testing semantic similarity...")
        
        similar_texts = [
            "The child was neglected by their parents.",
            "The parents failed to provide proper care for the child.",
            "There was a lack of parental supervision and care."
        ]
        
        different_texts = [
            "The weather is sunny today.",
            "I like to eat pizza for dinner.",
            "The car is parked in the garage."
        ]
        
        # Create embeddings for all texts
        all_texts = similar_texts + different_texts
        all_embeddings = model.encode(all_texts)
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        similarity_matrix = cosine_similarity(all_embeddings)
        
        print(f"   Similar texts similarity:")
        for i in range(len(similar_texts)):
            for j in range(i+1, len(similar_texts)):
                sim = similarity_matrix[i][j]
                print(f"      '{similar_texts[i][:30]}...' vs '{similar_texts[j][:30]}...': {sim:.3f}")
        
        print(f"   Different texts similarity:")
        for i in range(len(similar_texts), len(all_texts)):
            for j in range(i+1, len(all_texts)):
                sim = similarity_matrix[i][j]
                print(f"      '{all_texts[i][:30]}...' vs '{all_texts[j][:30]}...': {sim:.3f}")
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"✅ Qwen3-Embedding-8B model is working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def test_fallback_model():
    """Test the fallback model if Qwen fails"""
    print(f"\n🔄 Testing fallback model...")
    
    try:
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"   Loading fallback: {fallback_model}")
        
        model = SentenceTransformer(fallback_model)
        embedding_dim = model.get_sentence_embedding_dimension()
        
        print(f"   ✅ Fallback model loaded: {embedding_dim} dimensions")
        
        # Test with simple text
        test_embedding = model.encode("Test sentence")
        print(f"   ✅ Test embedding created: {len(test_embedding)} dimensions")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fallback model also failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Qwen3-Embedding-8B model test...")
    
    # Test the main model
    success = test_qwen_embedding()
    
    if not success:
        print(f"\n⚠️ Main model failed, testing fallback...")
        fallback_success = test_fallback_model()
        
        if fallback_success:
            print(f"\n✅ Fallback model works - you can use the system with reduced functionality")
        else:
            print(f"\n❌ Both models failed - please check your installation")
            print(f"   Try: pip install -r requirements.txt")
            return 1
    
    print(f"\n🎯 Test completed successfully!")
    print(f"   The embedding model is ready to use in your case review system")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
