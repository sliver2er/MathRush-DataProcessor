"""
Check Supabase database contents after processing.
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from processors import DatabaseSaver
from config.settings import settings
import json

def check_database():
    """Check the current state of the database."""
    print("=== Checking Supabase Database ===\n")
    
    try:
        # Initialize database connection
        db_saver = DatabaseSaver()
        
        # Test connection
        print("1. Testing connection...")
        if db_saver.test_connection():
            print("✅ Database connection successful\n")
        else:
            print("❌ Database connection failed\n")
            return
        
        # Get total problem count
        print("2. Getting total problem count...")
        total_count = db_saver.get_problem_count()
        print(f"📊 Total problems in database: {total_count}\n")
        
        # Get recent problems
        print("3. Fetching recent problems...")
        try:
            # Query recent problems
            result = db_saver.client.table(settings.SUPABASE_TABLE).select("*").order("created_at", desc=True).limit(10).execute()
            
            if result.data:
                print(f"📋 Found {len(result.data)} recent problems:\n")
                
                for i, problem in enumerate(result.data, 1):
                    print(f"Problem {i}:")
                    print(f"  ID: {problem.get('id', 'N/A')}")
                    print(f"  Content: {problem.get('content', 'N/A')[:100]}...")
                    print(f"  Problem Type: {problem.get('problem_type', 'N/A')}")
                    print(f"  Level: {problem.get('level', 'N/A')}")
                    print(f"  Subject: {problem.get('subject', 'N/A')}")
                    print(f"  Images: {problem.get('images', [])}")
                    print(f"  Created: {problem.get('created_at', 'N/A')}")
                    print(f"  Source: {problem.get('source_info', {}).get('problems_file', 'N/A')}")
                    print()
            else:
                print("📭 No problems found in database\n")
                
        except Exception as e:
            print(f"❌ Error querying problems: {e}\n")
        
        # Check for problems from our test file
        print("4. Checking for 2020-12-03 suneung problems...")
        try:
            result = db_saver.client.table(settings.SUPABASE_TABLE).select("*").contains("source_info", {"problems_file": "2020-12-03_suneung_problems.pdf"}).execute()
            
            if result.data:
                print(f"🎯 Found {len(result.data)} problems from 2020-12-03 suneung exam:")
                
                for i, problem in enumerate(result.data, 1):
                    print(f"  Problem {i}:")
                    print(f"    ID: {problem.get('id')}")
                    print(f"    Problem Number: {problem.get('problem_number', 'N/A')}")
                    print(f"    Content: {problem.get('content', 'N/A')[:150]}...")
                    print(f"    Correct Answer: {problem.get('correct_answer', 'N/A')}")
                    print(f"    Images: {len(problem.get('images', []))} files")
                    if problem.get('images'):
                        print(f"      Image files: {problem.get('images')}")
                    print(f"    Match Method: {problem.get('match_method', 'N/A')}")
                    print()
            else:
                print("📭 No problems found from 2020-12-03 suneung exam")
                
        except Exception as e:
            print(f"❌ Error querying suneung problems: {e}")
        
        # Check database schema
        print("5. Checking database schema...")
        try:
            # Try to get one record to see the schema
            result = db_saver.client.table(settings.SUPABASE_TABLE).select("*").limit(1).execute()
            
            if result.data:
                sample_record = result.data[0]
                print("📋 Database schema (sample record fields):")
                for key, value in sample_record.items():
                    value_type = type(value).__name__
                    value_preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    print(f"  {key}: {value_type} = {value_preview}")
                print()
            else:
                print("📭 No records available to check schema")
                
        except Exception as e:
            print(f"❌ Error checking schema: {e}")
            
    except Exception as e:
        print(f"❌ Database check failed: {e}")

if __name__ == "__main__":
    check_database()