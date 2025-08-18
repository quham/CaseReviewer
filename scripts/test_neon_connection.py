# test_neon_connection.py
import os
import psycopg2
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

try:
    # Get connection string from environment - using DATABASE_URL as you have it
    connection_string = os.getenv('DATABASE_URL')
    
    if not connection_string:
        print("❌ DATABASE_URL not found in .env file")
        print("Make sure your .env file contains:")
        print("DATABASE_URL='postgresql://neondb_owner:npg_y2rQJVkqG3Mu@ep-small-bread-abzccfym-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require'")
        exit(1)
    
    print(" Parsing connection string...")
    print(f"Connection string: {connection_string[:50]}...")
    
    # Parse the connection string to validate it
    parsed = urlparse(connection_string)
    print(f"✅ Parsed successfully:")
    print(f"   Host: {parsed.hostname}")
    print(f"   Port: {parsed.port or 5432}")
    print(f"   Database: {parsed.path[1:] if parsed.path else 'default'}")
    print(f"   Username: {parsed.username}")
    print(f"   SSL Mode: {parsed.query}")
    
    print("\n Connecting to Neon PostgreSQL...")
    conn = psycopg2.connect(connection_string)
    
    print("✅ Successfully connected to Neon!")
    
    # Test pgvector extension
    with conn.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("✅ pgvector extension enabled")
        
        # Show database info
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f" PostgreSQL version: {version}")
        
        # Show current database
        cursor.execute("SELECT current_database();")
        db_name = cursor.fetchone()[0]
        print(f"🗄️ Database: {db_name}")
        
        # Show current user
        cursor.execute("SELECT current_user;")
        user = cursor.fetchone()[0]
        print(f"👤 User: {user}")
        
    conn.close()
    print("✅ Connection test successful!")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Check your .env file format:")
    print("   DATABASE_URL='postgresql://neondb_owner:npg_y2rQJVkqG3Mu@ep-small-bread-abzccfym-pooler.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require'")
    print("2. Make sure there are no extra spaces or quotes")
    print("3. Verify the connection string from Neon dashboard")
    print("4. Check if the database is active")