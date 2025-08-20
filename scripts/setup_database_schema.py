#!/usr/bin/env python3
"""
Database Schema Setup Script for NSPCC Case Review System

This script creates the necessary tables and columns for the PDF processing pipeline.
Run this before running the main PDF processor script.

Required environment variables:
- DATABASE_URL: PostgreSQL connection string
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

class DatabaseSchemaSetup:
    def __init__(self):
        """Initialize database connection"""
        self.database_url = os.getenv('DATABASE_URL')
        
        if not self.database_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        # Parse connection string for display
        parsed = urlparse(self.database_url)
        self.pg_host = parsed.hostname
        self.pg_port = parsed.port or 5432
        self.pg_database = parsed.path[1:] if parsed.path else 'default'
        self.pg_user = parsed.username
        
        print(f"✅ Database configuration loaded:")
        print(f"   Host: {self.pg_host}")
        print(f"   Port: {self.pg_port}")
        print(f"   Database: {self.pg_database}")
        print(f"   User: {self.pg_user}")
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.database_url)
            print("✅ Database connection established successfully")
        except Exception as e:
            print(f"❌ Error connecting to database: {e}")
            raise
    
    def enable_pgvector(self):
        """Enable pgvector extension for vector operations"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.conn.commit()
            print("✅ pgvector extension enabled")
        except Exception as e:
            print(f"❌ Error enabling pgvector: {e}")
            raise
    
    def create_case_reviews_table(self):
        """Create the case_reviews table with all required columns"""
        try:
            with self.conn.cursor() as cursor:
                # Check if table already exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'case_reviews'
                    )
                """)
                
                if cursor.fetchone()[0]:
                    print("⚠️ Table 'case_reviews' already exists")
                    return False
                
                # Create the table with all required columns
                cursor.execute("""
                    CREATE TABLE case_reviews (
                        id SERIAL PRIMARY KEY,
                        title TEXT,
                        summary TEXT,
                        child_age INTEGER,
                        risk_types JSONB DEFAULT '[]'::jsonb,
                        outcome TEXT,
                        review_date DATE,
                        agencies JSONB DEFAULT '[]'::jsonb,
                        warning_signs_early JSONB DEFAULT '[]'::jsonb,
                        risk_factors JSONB DEFAULT '[]'::jsonb,
                        barriers JSONB DEFAULT '[]'::jsonb,
                        relationship_model JSONB DEFAULT '{}'::jsonb,
                        embedding vector(1536), -- Adjust dimension based on your embedding model
                        source_file TEXT,
                        file_hash TEXT UNIQUE,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX idx_case_reviews_file_hash ON case_reviews(file_hash)")
                cursor.execute("CREATE INDEX idx_case_reviews_created_at ON case_reviews(created_at)")
                cursor.execute("CREATE INDEX idx_case_reviews_review_date ON case_reviews(review_date)")
                cursor.execute("CREATE INDEX idx_case_reviews_child_age ON case_reviews(child_age)")
                
                # Create GIN indexes for JSONB columns for efficient searching
                cursor.execute("CREATE INDEX idx_case_reviews_risk_types ON case_reviews USING GIN(risk_types)")
                cursor.execute("CREATE INDEX idx_case_reviews_agencies ON case_reviews USING GIN(agencies)")
                cursor.execute("CREATE INDEX idx_case_reviews_warning_signs_early ON case_reviews USING GIN(warning_signs_early)")
                cursor.execute("CREATE INDEX idx_case_reviews_risk_factors ON case_reviews USING GIN(risk_factors)")
                cursor.execute("CREATE INDEX idx_case_reviews_barriers ON case_reviews USING GIN(barriers)")
                cursor.execute("CREATE INDEX idx_case_reviews_relationship_model ON case_reviews USING GIN(relationship_model)")
                
                self.conn.commit()
                print("✅ Table 'case_reviews' created successfully with all required columns")
                print("✅ Performance indexes created")
                return True
                
        except Exception as e:
            print(f"❌ Error creating case_reviews table: {e}")
            self.conn.rollback()
            raise
    
    def create_timeline_events_table(self):
        """Create the timeline_events table"""
        try:
            with self.conn.cursor() as cursor:
                # Check if table already exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'timeline_events'
                    )
                """)
                
                if cursor.fetchone()[0]:
                    print("⚠️ Table 'timeline_events' already exists")
                    return False
                
                # Create the table
                cursor.execute("""
                    CREATE TABLE timeline_events (
                        id SERIAL PRIMARY KEY,
                        case_review_id INTEGER NOT NULL REFERENCES case_reviews(id) ON DELETE CASCADE,
                        event_date TEXT, -- Using TEXT for flexible date formats
                        event_type TEXT DEFAULT 'other',
                        description TEXT,
                        impact TEXT,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX idx_timeline_events_case_review_id ON timeline_events(case_review_id)")
                cursor.execute("CREATE INDEX idx_timeline_events_event_type ON timeline_events(event_type)")
                cursor.execute("CREATE INDEX idx_timeline_events_event_date ON timeline_events(event_date)")
                
                self.conn.commit()
                print("✅ Table 'timeline_events' created successfully")
                print("✅ Performance indexes created")
                return True
                
        except Exception as e:
            print(f"❌ Error creating timeline_events table: {e}")
            self.conn.rollback()
            raise
    
    def create_users_table(self):
        """Create a basic users table for authentication (optional)"""
        try:
            with self.conn.cursor() as cursor:
                # Check if table already exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'users'
                    )
                """)
                
                if cursor.fetchone()[0]:
                    print("⚠️ Table 'users' already exists")
                    return False
                
                # Create a basic users table
                cursor.execute("""
                    CREATE TABLE users (
                        id SERIAL PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE,
                        password_hash TEXT,
                        role TEXT DEFAULT 'user',
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX idx_users_username ON users(username)")
                cursor.execute("CREATE INDEX idx_users_email ON users(email)")
                
                self.conn.commit()
                print("✅ Table 'users' created successfully")
                print("✅ Performance indexes created")
                return True
                
        except Exception as e:
            print(f"❌ Error creating users table: {e}")
            self.conn.rollback()
            raise
    
    def verify_schema(self):
        """Verify that all required tables and columns exist"""
        try:
            with self.conn.cursor() as cursor:
                print("\n🔍 Verifying database schema...")
                
                # Check case_reviews table
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = 'case_reviews'
                    ORDER BY ordinal_position
                """)
                
                case_reviews_columns = cursor.fetchall()
                print(f"\n📋 case_reviews table columns ({len(case_reviews_columns)}):")
                for col in case_reviews_columns:
                    nullable = "NULL" if col[2] == "YES" else "NOT NULL"
                    print(f"   - {col[0]}: {col[1]} ({nullable})")
                
                # Check timeline_events table
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = 'timeline_events'
                    ORDER BY ordinal_position
                """)
                
                timeline_columns = cursor.fetchall()
                print(f"\n📋 timeline_events table columns ({len(timeline_columns)}):")
                for col in timeline_columns:
                    nullable = "NULL" if col[2] == "YES" else "NOT NULL"
                    print(f"   - {col[0]}: {col[1]} ({nullable})")
                
                # Check users table
                cursor.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = 'users'
                    ORDER BY ordinal_position
                """)
                
                users_columns = cursor.fetchall()
                print(f"\n📋 users table columns ({len(users_columns)}):")
                for col in users_columns:
                    nullable = "NULL" if col[2] == "YES" else "NOT NULL"
                    print(f"   - {col[0]}: {col[1]} ({nullable})")
                
                print("\n✅ Schema verification completed")
                
        except Exception as e:
            print(f"❌ Error verifying schema: {e}")
            raise
    
    def setup_complete_schema(self):
        """Set up the complete database schema"""
        try:
            print("🚀 Setting up complete database schema...")
            
            # Connect to database
            self.connect()
            
            # Enable pgvector extension
            self.enable_pgvector()
            
            # Create tables
            case_reviews_created = self.create_case_reviews_table()
            timeline_events_created = self.create_timeline_events_table()
            users_created = self.create_users_table()
            
            # Verify schema
            self.verify_schema()
            
            print("\n🎉 Database schema setup completed successfully!")
            
            if case_reviews_created or timeline_events_created or users_created:
                print("📝 New tables were created. You can now run the PDF processor script.")
            else:
                print("ℹ️ All tables already existed. Schema is ready for use.")
            
        except Exception as e:
            print(f"❌ Schema setup failed: {e}")
            raise
        finally:
            if hasattr(self, 'conn'):
                self.conn.close()
                print("✅ Database connection closed")
    


def main():
    """Main function to set up database schema"""
    try:
        setup = DatabaseSchemaSetup()
        
        print("Choose an option:")
        print("1. Setup complete schema (recommended)")
        print("2. Verify existing schema")
        print("3. Drop all tables (⚠️ DANGEROUS)")
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            setup.setup_complete_schema()
        elif choice == "2":
            setup.connect()
            setup.verify_schema()
            setup.conn.close()
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
