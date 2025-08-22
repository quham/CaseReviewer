#!/usr/bin/env python3
"""
Simple script to create a demo user using the existing backend API
"""

import requests
import json

# Backend URL
BACKEND_URL = "http://localhost:8000"

def create_demo_user():
    """Create demo user using the registration API"""
    try:
        # Demo user data
        user_data = {
            "username": "sarah.johnson",
            "password": "socialworker2024",
            "name": "Sarah Johnson",
            "role": "social_worker",
            "organization": "Riverside Children's Services"
        }
        
        print("🚀 Creating demo user via API...")
        
        # Try to register the user
        response = requests.post(f"{BACKEND_URL}/api/register", json=user_data)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Demo user created successfully!")
            print(f"   Username: {user_data['username']}")
            print(f"   Password: {user_data['password']}")
            print(f"   Token: {data['token'][:20]}...")
            return True
        elif response.status_code == 400:
            # User might already exist, try to login
            print("ℹ️  User might already exist, testing login...")
            login_data = {
                "username": user_data["username"],
                "password": user_data["password"]
            }
            
            login_response = requests.post(f"{BACKEND_URL}/api/login", json=login_data)
            
            if login_response.status_code == 200:
                print("✅ Demo user login successful!")
                print(f"   Username: {user_data['username']}")
                print(f"   Password: {user_data['password']}")
                return True
            else:
                print(f"❌ Login failed: {login_response.text}")
                return False
        else:
            print(f"❌ Registration failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Make sure the server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🎯 CaseReviewer Demo User Setup")
    print("=" * 40)
    
    success = create_demo_user()
    
    if success:
        print("\n✨ Demo user setup complete!")
        print("\nYou can now login with:")
        print("   Username: sarah.johnson")
        print("   Password: socialworker2024")
    else:
        print("\n❌ Demo user setup failed!")
        print("Please check that:")
        print("1. The backend server is running")
        print("2. The database is properly configured")
        print("3. All required dependencies are installed")
