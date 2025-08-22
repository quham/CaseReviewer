#!/usr/bin/env python3
"""
Test script for OpenRouter model fallback mechanism
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_fallback():
    """Test the model fallback mechanism"""
    try:
        from services.openrouter_insights import openrouter_insights_service
        
        if not openrouter_insights_service.llm_enabled:
            logger.error("❌ OpenRouter service not enabled")
            return False
        
        logger.info("🧪 Testing model fallback mechanism...")
        
        # Test 1: Get model status
        logger.info("📋 Getting model status...")
        status = openrouter_insights_service.get_model_status()
        logger.info(f"Primary model: {status['primary_model']}")
        logger.info(f"Fallback models: {len(status['fallback_models'])}")
        logger.info(f"First fallback: {status['fallback_models'][0]}")
        
        # Test 2: Test a simple prompt
        logger.info("🔄 Testing simple prompt with fallback...")
        test_prompt = "Hello, this is a test. Please respond with 'OK' if you can see this message."
        
        result = openrouter_insights_service._call_openrouter(test_prompt, max_tokens=50)
        
        if result:
            response, used_model = result
            logger.info(f"✅ Success! Model used: {used_model}")
            logger.info(f"Response: {response}")
        else:
            logger.error("❌ All models failed")
            return False
        
        # Test 3: Test personalized recommendations
        logger.info("🔄 Testing personalized recommendations...")
        test_cases = [{
            "title": "Test Case",
            "summary": "A test case for validation",
            "risk_types": ["test"],
            "agencies": ["test_agency"]
        }]
        
        recommendations = openrouter_insights_service.generate_personalized_recommendations(
            "test query", test_cases, "social_worker"
        )
        
        if recommendations and "metadata" in recommendations:
            logger.info(f"✅ Recommendations generated successfully")
            logger.info(f"Model used: {recommendations['metadata'].get('llm_model', 'unknown')}")
            logger.info(f"Source: {recommendations['metadata'].get('source', 'unknown')}")
        else:
            logger.warning("⚠️ Recommendations generation had issues")
        
        # Test 4: Test all models
        logger.info("🧪 Testing all models...")
        test_results = openrouter_insights_service.test_models()
        
        logger.info(f"Total models: {test_results['total_models']}")
        logger.info(f"Working models: {test_results['working_models']}")
        logger.info(f"Failed models: {test_results['failed_models']}")
        
        # Show working models
        working_models = [model for model, result in test_results['results'].items() 
                         if result['status'] == 'working']
        if working_models:
            logger.info(f"✅ Working models: {', '.join(working_models)}")
        
        # Show failed models
        failed_models = [model for model, result in test_results['results'].items() 
                        if result['status'] != 'working']
        if failed_models:
            logger.warning(f"❌ Failed models: {', '.join(failed_models)}")
        
        logger.info("🎉 Model fallback testing completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    load_dotenv()
    
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("❌ OPENROUTER_API_KEY not set")
        sys.exit(1)
    
    success = test_model_fallback()
    sys.exit(0 if success else 1)
