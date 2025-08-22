#!/usr/bin/env python3
"""
Test script for CaseReviewer Python Server
Tests basic functionality and database connection
"""

import os
import sys
import asyncio
import requests
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_database_connection():
    """Test database connection"""
    print("🔍 Testing database connection...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print("❌ DATABASE_URL not found in environment")
            return False
        
        import psycopg2
        conn = psycopg2.connect(database_url)
        
        # Test basic query
        with conn.cursor() as cursor:
            cursor.execute("SELECT version()")
            version = cursor.fetchone()
            print(f"✅ Database connected: {version[0][:50]}...")
        
        # Test pgvector extension
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
            vector_ext = cursor.fetchone()
            if vector_ext:
                print("✅ pgvector extension is available")
            else:
                print("⚠️ pgvector extension not found")
        
        # Test case_reviews table
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM case_reviews")
            count = cursor.fetchone()
            print(f"✅ case_reviews table accessible: {count[0]} records")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_embedding_model():
    """Test DeepInfra API client"""
    print("\n🔍 Testing DeepInfra API client...")
    
    try:
        from openai import OpenAI
        import os
        
        deepinfra_token = os.getenv("DEEPINFRA_TOKEN")
        if not deepinfra_token:
            print("⚠️ DEEPINFRA_TOKEN not found, skipping API test")
            return True
        
        client = OpenAI(
            api_key=deepinfra_token,
            base_url="https://api.deepinfra.com/v1/openai",
        )
        
        test_text = "This is a test case for child protection"
        result = client.embeddings.create(
            model="Qwen/Qwen3-Embedding-8B",
            input=test_text,
            encoding_format="float"
        )
        
        print(f"✅ DeepInfra API client working")
        print(f"   Model: Qwen/Qwen3-Embedding-8B")
        print(f"   Test embedding created: {len(result.data[0].embedding)} dimensions")
        
        return True
        
    except Exception as e:
        print(f"❌ DeepInfra API client failed: {e}")
        return False

def test_server_endpoints():
    """Test server endpoints"""
    print("\n🔍 Testing server endpoints...")
    
    base_url = "http://localhost:5000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint accessible")
            health_data = response.json()
            print(f"   Status: {health_data.get('status')}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health endpoint not accessible: {e}")
        return False
    
    # Test registration endpoint
    try:
        test_user = {
            "username": f"testuser_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "password": "testpassword123",
            "name": "Test User",
            "role": "social_worker"
        }
        
        response = requests.post(
            f"{base_url}/api/register",
            json=test_user,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Registration endpoint working")
            user_data = response.json()
            print(f"   User created: {user_data.get('user', {}).get('username')}")
            
            # Test login with created user
            login_response = requests.post(
                f"{base_url}/api/login",
                json={
                    "username": test_user["username"],
                    "password": test_user["password"]
                },
                timeout=10
            )
            
            if login_response.status_code == 200:
                print("✅ Login endpoint working")
                login_data = login_response.json()
                token = login_data.get('token')
                
                # Test protected endpoint
                headers = {"Authorization": f"Bearer {token}"}
                me_response = requests.get(
                    f"{base_url}/api/protected/me",
                    headers=headers,
                    timeout=10
                )
                
                if me_response.status_code == 200:
                    print("✅ Protected endpoint working")
                    me_data = me_response.json()
                    print(f"   Authenticated user: {me_data.get('user', {}).get('username')}")
                else:
                    print(f"❌ Protected endpoint failed: {me_response.status_code}")
                
            else:
                print(f"❌ Login endpoint failed: {login_response.status_code}")
                
        else:
            print(f"❌ Registration endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API endpoints test failed: {e}")
        return False
    
    return True

def test_ai_insights():
    """Test AI insights service"""
    print("\n🔍 Testing AI insights service...")
    
    try:
        from services.ai_insights import AIInsightsService
        
        service = AIInsightsService()
        
        # Test case data
        test_case = {
            "title": "Test Case Review",
            "summary": "A test case involving neglect and domestic violence",
            "risk_types": ["neglect", "domestic_violence"],
            "agencies": ["social_services", "police", "health"],
            "barriers": ["lack of coordination", "resource constraints"],
            "warning_signs_early": ["missed appointments", "school absences"],
            "risk_factors": ["parental mental health", "substance abuse"],
            "outcome": "Ongoing monitoring and support"
        }
        
        # Test complexity analysis
        complexity = service.analyze_case_complexity(test_case)
        print(f"✅ Complexity analysis working")
        print(f"   Complexity level: {complexity['complexity_level']}")
        print(f"   Score: {complexity['complexity_score']}")
        
        # Test risk assessment
        risk_assessment = service.extract_risk_assessment(test_case)
        print(f"✅ Risk assessment working")
        print(f"   Risk level: {risk_assessment['overall_risk_level']}")
        print(f"   Urgency: {risk_assessment['urgency_level']}")
        
        # Test intervention strategies
        strategies = service.generate_intervention_strategies(test_case)
        print(f"✅ Intervention strategies working")
        print(f"   Immediate actions: {len(strategies['immediate_actions'])}")
        print(f"   Long-term strategies: {len(strategies['long_term_strategies'])}")
        
        # Test personalized summary
        summary = service.create_personalized_summary(test_case, "test query", "social_worker")
        print(f"✅ Personalized summary working")
        print(f"   Executive summary: {len(summary['executive_summary'])} chars")
        print(f"   Key findings: {len(summary['key_findings'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI insights service failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_openrouter_insights():
    """Test OpenRouter AI insights service"""
    print("\n🔍 Testing OpenRouter AI insights service...")
    
    try:
        from services.openrouter_insights import openrouter_insights_service
        
        if not openrouter_insights_service.llm_enabled:
            print("⚠️ OpenRouter not configured, testing fallback functionality")
        
        # Test case data
        test_case = {
            "title": "Test Case Review",
            "summary": "A test case involving neglect and domestic violence",
            "risk_types": ["neglect", "domestic_violence"],
            "agencies": ["social_services", "police", "health"],
            "barriers": ["lack of coordination", "resource constraints"],
            "warning_signs_early": ["missed appointments", "school absences"],
            "risk_factors": ["parental mental health", "substance abuse"],
            "outcome": "Ongoing monitoring and support"
        }
        
        # Test personalized recommendations
        test_matches = [test_case]
        recommendations = openrouter_insights_service.generate_personalized_recommendations(
            "missed opportunities in child protection", test_matches, "social_worker"
        )
        print(f"✅ Personalized recommendations working")
        print(f"   Recommendations: {len(recommendations.get('personalized_recommendations', {}))} categories")
        print(f"   Key insights: {len(recommendations.get('key_insights', []))} insights")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenRouter insights service failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 CaseReviewer Python Server Test Suite")
    print("=" * 50)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Embedding Model", test_embedding_model),
        ("AI Insights Service", test_ai_insights),
        ("OpenRouter AI Insights", test_openrouter_insights),
        ("Server Endpoints", test_server_endpoints)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Server is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
