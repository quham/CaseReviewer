#!/usr/bin/env python3
"""
Script to check timeline events in the database
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_timeline_events():
    """Check timeline events in the database"""
    
    # Get database connection details
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not found in environment variables")
        return
    
    try:
        # Connect to database
        conn = psycopg2.connect(database_url)
        print("✅ Connected to database")
        
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Check if timeline_events table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'timeline_events'
                )
            """)
            
            table_exists = cursor.fetchone()[0]
            print(f"📊 Timeline events table exists: {table_exists}")
            
            if not table_exists:
                print("❌ Timeline events table does not exist!")
                return
            
            # Check table structure
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'timeline_events'
                ORDER BY ordinal_position
            """)
            
            columns = cursor.fetchall()
            print("\n📋 Table structure:")
            for col in columns:
                print(f"  - {col['column_name']}: {col['data_type']} (nullable: {col['is_nullable']})")
            
            # Count total timeline events
            cursor.execute("SELECT COUNT(*) as total FROM timeline_events")
            total_events = cursor.fetchone()['total']
            print(f"\n📊 Total timeline events: {total_events}")
            
            if total_events > 0:
                # Get sample events
                cursor.execute("""
                    SELECT * FROM timeline_events 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """)
                
                sample_events = cursor.fetchall()
                print(f"\n🔍 Sample events (last 5):")
                for i, event in enumerate(sample_events, 1):
                    print(f"\n  Event {i}:")
                    for key, value in event.items():
                        print(f"    {key}: {value}")
                
                # Check case reviews with timeline events
                cursor.execute("""
                    SELECT 
                        cr.id,
                        cr.title,
                        COUNT(te.id) as event_count
                    FROM case_reviews cr
                    LEFT JOIN timeline_events te ON cr.id = te.case_review_id
                    GROUP BY cr.id, cr.title
                    HAVING COUNT(te.id) > 0
                    ORDER BY event_count DESC
                    LIMIT 10
                """)
                
                cases_with_events = cursor.fetchall()
                print(f"\n📊 Cases with timeline events:")
                for case in cases_with_events:
                    print(f"  - Case {case['id']}: {case['title']} ({case['event_count']} events)")
                
                # Check for any malformed events
                cursor.execute("""
                    SELECT * FROM timeline_events 
                    WHERE event_date IS NULL 
                       OR description IS NULL 
                       OR event_type IS NULL
                    LIMIT 5
                """)
                
                malformed_events = cursor.fetchall()
                if malformed_events:
                    print(f"\n⚠️  Found {len(malformed_events)} malformed events:")
                    for event in malformed_events:
                        print(f"  - Event {event['id']}: date={event['event_date']}, type={event['event_type']}, desc={event['description']}")
                else:
                    print("\n✅ No malformed events found")
                    
            else:
                print("\n❌ No timeline events found in database")
                
                # Check if case_reviews table has data
                cursor.execute("SELECT COUNT(*) as total FROM case_reviews")
                total_cases = cursor.fetchone()['total']
                print(f"📊 Total case reviews: {total_cases}")
                
                if total_cases > 0:
                    print("💡 Case reviews exist but no timeline events. This suggests:")
                    print("   - Timeline events were not extracted from PDFs")
                    print("   - Timeline events were not saved to database")
                    print("   - There's an issue with the PDF processing pipeline")
        
        conn.close()
        print("\n✅ Database connection closed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_timeline_events()
