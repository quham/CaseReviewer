#!/usr/bin/env python3
"""
Script to create a demo user for testing the CaseReviewer application
"""

import os
import sys
import bcrypt
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("❌ DATABASE_URL environment variable is required")
    sys.exit(1)

def create_demo_user():
    """Create demo user in the database"""
    try:
        # Connect to database
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if demo user already exists
        cursor.execute("SELECT id FROM users WHERE username = %s", ("lashonte",))
        existing_user = cursor.fetchone()
        
        if existing_user:
            print("✅ Demo user 'lashonte' already exists")
            return
        
        # Hash password
        password = "socialworker2025"
        hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        
        # Create demo user
        cursor.execute("""
            INSERT INTO users (username, password, name, role, organization)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, username, name, role
        """, ("lashonte", hashed_password, "Lashonte Royal", "Children's Social Worker", "Children's Services"))
        
        user_data = cursor.fetchone()
        conn.commit()
        
        print("✅ Demo user created successfully!")
        print(f"   Username: {user_data['username']}")
        print(f"   Password: {password}")
        print(f"   Name: {user_data['name']}")
        print(f"   Role: {user_data['role']}")
        print(f"   Organization: {user_data['organization']}")
        
    except Exception as e:
        print(f"❌ Error creating demo user: {e}")
        if conn:
            conn.rollback()
        sys.exit(1)
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("🚀 Creating demo user for CaseReviewer...")
    create_demo_user()
    print("✨ Demo user setup complete!")
