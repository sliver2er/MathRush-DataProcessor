#!/usr/bin/env python3
"""
Database checker utility for MathRush DataProcessor.
Quick way to check database content and statistics.
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from processors import DatabaseSaver

def check_database():
    """Check database content and show statistics."""
    try:
        db = DatabaseSaver()
        
        # Test connection
        print("🔍 CHECKING DATABASE CONNECTION...")
        if not db.test_connection():
            print("❌ Database connection failed")
            return False
        
        print("✅ Database connection successful")
        
        # Get all records
        result = db.client.table('problems').select('*').execute()
        records = result.data
        
        print(f"\n📊 DATABASE STATISTICS")
        print(f"{'='*50}")
        print(f"Total records: {len(records)}")
        
        if not records:
            print("No records found in database")
            return True
        
        # Group by exam
        exams = {}
        answered_count = 0
        
        for record in records:
            source_info = record.get('source_info', {})
            exam_name = source_info.get('exam_name', 'Unknown')
            
            if exam_name not in exams:
                exams[exam_name] = {
                    'total': 0,
                    'answered': 0,
                    'multiple_choice': 0,
                    'subjective': 0
                }
            
            exams[exam_name]['total'] += 1
            
            if record.get('correct_answer'):
                answered_count += 1
                exams[exam_name]['answered'] += 1
            
            problem_type = record.get('problem_type', 'unknown')
            if problem_type == 'multiple_choice':
                exams[exam_name]['multiple_choice'] += 1
            elif problem_type == 'subjective':
                exams[exam_name]['subjective'] += 1
        
        print(f"Total answered: {answered_count}/{len(records)} ({answered_count/len(records)*100:.1f}%)")
        print(f"Total exams: {len(exams)}")
        
        # Show exam details
        print(f"\n📚 EXAM BREAKDOWN")
        print(f"{'='*50}")
        for exam_name, stats in exams.items():
            answered_pct = stats['answered']/stats['total']*100 if stats['total'] > 0 else 0
            print(f"\n📋 {exam_name}")
            print(f"  Total problems: {stats['total']}")
            print(f"  Answered: {stats['answered']}/{stats['total']} ({answered_pct:.1f}%)")
            print(f"  Multiple choice: {stats['multiple_choice']}")
            print(f"  Subjective: {stats['subjective']}")
        
        # Show sample records
        print(f"\n📄 SAMPLE RECORDS")
        print(f"{'='*50}")
        
        # Show first few records
        for i, record in enumerate(records[:5]):
            source_info = record.get('source_info', {})
            problem_num = source_info.get('problem_number', 'Unknown')
            exam_name = source_info.get('exam_name', 'Unknown')
            content = record.get('content', '')[:80] + '...'
            answer = record.get('correct_answer', '(empty)')
            problem_type = record.get('problem_type', 'unknown')
            
            print(f"\n{i+1}. Problem {problem_num} ({exam_name})")
            print(f"   Type: {problem_type}")
            print(f"   Content: {content}")
            print(f"   Answer: {answer}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return False

def search_problems(query=None, exam_name=None):
    """Search for specific problems."""
    try:
        db = DatabaseSaver()
        
        # Build query
        query_builder = db.client.table('problems').select('*')
        
        if exam_name:
            query_builder = query_builder.eq('source_info->>exam_name', exam_name)
        
        result = query_builder.execute()
        records = result.data
        
        print(f"\n🔍 SEARCH RESULTS")
        print(f"{'='*50}")
        print(f"Found {len(records)} records")
        
        if query:
            # Filter by content
            filtered_records = []
            for record in records:
                content = record.get('content', '').lower()
                if query.lower() in content:
                    filtered_records.append(record)
            records = filtered_records
            print(f"Filtered to {len(records)} records containing '{query}'")
        
        for i, record in enumerate(records[:10]):  # Show first 10
            source_info = record.get('source_info', {})
            problem_num = source_info.get('problem_number', 'Unknown')
            exam_name = source_info.get('exam_name', 'Unknown')
            content = record.get('content', '')[:100] + '...'
            answer = record.get('correct_answer', '(empty)')
            
            print(f"\n{i+1}. Problem {problem_num} ({exam_name})")
            print(f"   Content: {content}")
            print(f"   Answer: {answer}")
        
        if len(records) > 10:
            print(f"\n... and {len(records) - 10} more records")
        
    except Exception as e:
        print(f"❌ Error searching database: {e}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check MathRush database content")
    parser.add_argument("--search", help="Search for problems containing this text")
    parser.add_argument("--exam", help="Filter by exam name")
    parser.add_argument("--stats-only", action="store_true", help="Show only statistics")
    
    args = parser.parse_args()
    
    print("🤖 MathRush Database Checker")
    print("=" * 50)
    
    if args.search or args.exam:
        search_problems(query=args.search, exam_name=args.exam)
    else:
        check_database()

if __name__ == "__main__":
    main()