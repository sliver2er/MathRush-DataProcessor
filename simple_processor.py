#!/usr/bin/env python3
"""
Simple Image Processor for MathRush DataProcessor.
Processes manually segmented problem images and updates database with GPT-extracted content.
"""

import os
import sys
import argparse
import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time

# Import utilities and processors
try:
    from config.settings import settings
    from processors.db_saver import DatabaseSaver
    from lightweight_gpt_extractor import LightweightGPTExtractor
    from utils.filename_parser import FilenameParser
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from config.settings import settings
    from processors.db_saver import DatabaseSaver
    from lightweight_gpt_extractor import LightweightGPTExtractor
    from utils.filename_parser import FilenameParser

logger = logging.getLogger(__name__)


class SimpleProcessor:
    """Simple processor for manually segmented problem images."""
    
    def __init__(self):
        """Initialize the simple processor."""
        self.db_saver = DatabaseSaver()
        self.gpt_extractor = LightweightGPTExtractor()
        self.filename_parser = FilenameParser()
        
        # Test database connection
        if not self.db_saver.test_connection():
            raise ConnectionError("Failed to connect to database")
            
        logger.info("Simple Processor initialized")
    
    def scan_exam_directory(self, exam_dir: str) -> Dict[str, List[str]]:
        """
        Scan exam directory for problem and math content images.
        
        Args:
            exam_dir: Path to exam directory
            
        Returns:
            Dictionary mapping problem numbers to image files
        """
        exam_path = Path(exam_dir)
        if not exam_path.exists():
            raise FileNotFoundError(f"Exam directory not found: {exam_dir}")
        
        # Patterns for different file types
        problem_pattern = re.compile(r'(.+)_problem_(\d+)\.png$')
        content_pattern = re.compile(r'(.+)_problem_(\d+)_(?:diagram|content|graph|figure).*\.png$')
        
        problems = {}
        
        # First pass: find all problem images
        for file_path in exam_path.glob("*.png"):
            match = problem_pattern.match(file_path.name)
            if match:
                exam_name = match.group(1)
                problem_number = int(match.group(2))
                
                if problem_number not in problems:
                    problems[problem_number] = {
                        'exam_name': exam_name,
                        'problem_image': str(file_path),
                        'math_content_images': []
                    }
                else:
                    problems[problem_number]['problem_image'] = str(file_path)
        
        # Second pass: find math content images
        for file_path in exam_path.glob("*.png"):
            match = content_pattern.match(file_path.name)
            if match:
                problem_number = int(match.group(2))
                if problem_number in problems:
                    problems[problem_number]['math_content_images'].append(str(file_path))
        
        # Convert to sorted list format
        sorted_problems = []
        for problem_num in sorted(problems.keys()):
            problem_data = problems[problem_num]
            problem_data['problem_number'] = problem_num
            sorted_problems.append(problem_data)
        
        logger.info(f"Found {len(sorted_problems)} problems in {exam_dir}")
        for problem in sorted_problems:
            content_count = len(problem['math_content_images'])
            if content_count > 0:
                logger.info(f"  Problem {problem['problem_number']}: {content_count} math content images")
        
        return sorted_problems
    
    def find_manual_answers_in_db(self, exam_name: str) -> Dict[int, Dict[str, any]]:
        """
        Find manually input answers for this exam in the database.
        
        Args:
            exam_name: Name of the exam
            
        Returns:
            Dictionary mapping problem numbers to database records
        """
        try:
            # Query database for records with this exam name
            result = self.db_saver.client.table(self.db_saver.table_name)\
                .select("*")\
                .eq("source_info->>exam_name", exam_name)\
                .execute()
            
            if not result.data:
                logger.warning(f"No manual answers found for exam: {exam_name}")
                return {}
            
            # Organize by problem number
            manual_answers = {}
            for record in result.data:
                source_info = record.get('source_info', {})
                problem_number = source_info.get('problem_number')
                
                if problem_number:
                    manual_answers[problem_number] = record
            
            logger.info(f"Found manual answers for {len(manual_answers)} problems in exam: {exam_name}")
            return manual_answers
            
        except Exception as e:
            logger.error(f"Error querying manual answers for {exam_name}: {e}")
            return {}
    
    def process_problem_images(self, problems: List[Dict], manual_answers: Dict[int, Dict]) -> List[Dict]:
        """
        Process problem images through GPT extraction.
        
        Args:
            problems: List of problem data from directory scan
            manual_answers: Manual answers from database
            
        Returns:
            List of processed problem records
        """
        processed_problems = []
        
        for problem in problems:
            problem_num = problem['problem_number']
            problem_image = problem['problem_image']
            math_content_images = problem['math_content_images']
            
            try:
                logger.info(f"Processing Problem {problem_num}: {os.path.basename(problem_image)}")
                
                # Extract content using lightweight GPT extractor
                extracted_data = self.gpt_extractor.extract_from_image(
                    problem_image, 
                    math_content_images if math_content_images else None
                )
                
                # Get manual answer data if available
                manual_data = manual_answers.get(problem_num, {})
                
                # Combine extracted content with manual answer
                combined_data = {
                    # GPT extracted fields
                    'content': extracted_data.get('content', f'[Extraction Failed] Problem {problem_num}'),
                    'problem_type': extracted_data.get('problem_type', 'subjective'),
                    'choices': extracted_data.get('choices'),
                    
                    # Manual fields (from existing DB record)
                    'correct_answer': manual_data.get('correct_answer', ''),
                    'explanation': '',  # Not needed per requirements
                    
                    # Metadata fields
                    'curriculum': manual_data.get('curriculum', '2015개정'),
                    'level': manual_data.get('level', '고3'),
                    'subject': manual_data.get('subject', '수학영역'),
                    'chapter': manual_data.get('chapter', ''),
                    'difficulty': manual_data.get('difficulty', 'medium'),
                    'tags': manual_data.get('tags', []),
                    
                    # Source and image info
                    'source_info': {
                        **manual_data.get('source_info', {}),
                        'problem_image': problem_image,
                        'math_content_images': math_content_images,
                        'gpt_processed': True,
                        'gpt_processed_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
                    },
                    'images': [problem_image] + math_content_images
                }
                
                # Add database ID if updating existing record
                if manual_data.get('id'):
                    combined_data['id'] = manual_data['id']
                    combined_data['update_mode'] = True
                else:
                    combined_data['update_mode'] = False
                
                processed_problems.append(combined_data)
                
                logger.info(f"  ✅ Problem {problem_num}: {extracted_data.get('problem_type', 'unknown')} - Content: {len(extracted_data.get('content', ''))[:50]}...")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to process Problem {problem_num}: {e}")
                
                # Create fallback record
                fallback_data = {
                    'content': f'[Processing Failed] Problem {problem_num} from {problem["exam_name"]}',
                    'problem_type': 'subjective',
                    'choices': None,
                    'correct_answer': manual_answers.get(problem_num, {}).get('correct_answer', ''),
                    'explanation': '',
                    'curriculum': '2015개정',
                    'level': '고3',
                    'subject': '수학영역',
                    'chapter': '',
                    'difficulty': 'medium',
                    'tags': ['processing_failed'],
                    'source_info': {
                        'exam_name': problem['exam_name'],
                        'problem_number': problem_num,
                        'problem_image': problem_image,
                        'processing_error': str(e)
                    },
                    'images': [problem_image],
                    'update_mode': manual_answers.get(problem_num, {}).get('id') is not None
                }
                
                if manual_answers.get(problem_num, {}).get('id'):
                    fallback_data['id'] = manual_answers[problem_num]['id']
                
                processed_problems.append(fallback_data)
        
        return processed_problems
    
    def update_database(self, processed_problems: List[Dict]) -> Dict[str, int]:
        """
        Update database with processed problems.
        
        Args:
            processed_problems: List of processed problem records
            
        Returns:
            Dictionary with update statistics
        """
        stats = {
            'updated': 0,
            'inserted': 0,
            'failed': 0,
            'total': len(processed_problems)
        }
        
        for problem in processed_problems:
            try:
                if problem.get('update_mode') and problem.get('id'):
                    # Update existing record
                    record_id = problem.pop('id')
                    problem.pop('update_mode', None)
                    
                    # Update database record
                    result = self.db_saver.client.table(self.db_saver.table_name)\
                        .update(problem)\
                        .eq('id', record_id)\
                        .execute()
                    
                    if result.data:
                        stats['updated'] += 1
                        logger.debug(f"Updated record ID {record_id}")
                    else:
                        stats['failed'] += 1
                        logger.error(f"Failed to update record ID {record_id}")
                        
                else:
                    # Insert new record
                    problem.pop('update_mode', None)
                    result = self.db_saver.insert_problem(problem)
                    
                    if result:
                        stats['inserted'] += 1
                        logger.debug(f"Inserted new record ID {result.get('id')}")
                    else:
                        stats['failed'] += 1
                        logger.error("Failed to insert new record")
                        
            except Exception as e:
                stats['failed'] += 1
                logger.error(f"Database operation failed: {e}")
        
        return stats
    
    def process_exam_directory(self, exam_dir: str) -> bool:
        """
        Process entire exam directory.
        
        Args:
            exam_dir: Path to exam directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            exam_dir_path = Path(exam_dir)
            exam_name = exam_dir_path.name
            
            print(f"\n{'='*60}")
            print(f"📂 PROCESSING EXAM: {exam_name}")
            print(f"📁 Directory: {exam_dir}")
            print(f"{'='*60}")
            
            # Step 1: Scan directory for images
            print("\n📷 STEP 1: Scanning for problem images...")
            problems = self.scan_exam_directory(exam_dir)
            
            if not problems:
                print("❌ No problems found in directory")
                return False
            
            print(f"✅ Found {len(problems)} problems")
            
            # Step 2: Find manual answers in database
            print("\n🔍 STEP 2: Looking for manual answers in database...")
            manual_answers = self.find_manual_answers_in_db(exam_name)
            
            if not manual_answers:
                print(f"⚠️  No manual answers found. Run: python manual_answer_input.py {exam_dir}")
                proceed = input("Continue without manual answers? (y/N): ").strip().lower()
                if proceed != 'y':
                    return False
            else:
                print(f"✅ Found manual answers for {len(manual_answers)} problems")
            
            # Step 3: Process images through GPT
            print("\n🤖 STEP 3: Processing images through GPT...")
            processed_problems = self.process_problem_images(problems, manual_answers)
            
            # Step 4: Update database
            print("\n💾 STEP 4: Updating database...")
            stats = self.update_database(processed_problems)
            
            # Display final results
            print(f"\n{'='*60}")
            print(f"📊 PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"✅ Updated records: {stats['updated']}")
            print(f"➕ New records: {stats['inserted']}")
            print(f"❌ Failed: {stats['failed']}")
            print(f"📈 Total processed: {stats['total']}")
            
            success_rate = (stats['updated'] + stats['inserted']) / stats['total'] * 100
            print(f"📊 Success rate: {success_rate:.1f}%")
            
            return stats['failed'] == 0
            
        except Exception as e:
            logger.error(f"Error processing exam directory {exam_dir}: {e}")
            print(f"❌ Error: {e}")
            return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Simple Image Processor for MathRush DataProcessor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific exam directory
  python simple_processor.py input/2020-12-03_suneung/
  
  # Process with verbose logging
  python simple_processor.py input/2020-12-03_suneung/ --verbose
  
  # Process multiple exams in input directory
  python simple_processor.py input/ --recursive
"""
    )
    
    parser.add_argument(
        "exam_dir",
        help="Exam directory path"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Process all exam directories in the given path"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        processor = SimpleProcessor()
        
        if args.recursive:
            # Process all exam directories
            input_path = Path(args.exam_dir)
            if not input_path.exists():
                print(f"❌ Directory not found: {args.exam_dir}")
                return 1
            
            exam_dirs = []
            for item in input_path.iterdir():
                if item.is_dir():
                    # Check if directory contains problem images
                    problem_files = list(item.glob("*_problem_*.png"))
                    if problem_files:
                        exam_dirs.append(str(item))
            
            if not exam_dirs:
                print(f"❌ No exam directories found in {args.exam_dir}")
                return 1
            
            print(f"📚 Found {len(exam_dirs)} exam directories to process")
            
            success_count = 0
            for exam_dir in exam_dirs:
                if processor.process_exam_directory(exam_dir):
                    success_count += 1
                print()  # Empty line between exams
            
            print(f"📊 Final Results: {success_count}/{len(exam_dirs)} exams processed successfully")
            return 0 if success_count == len(exam_dirs) else 1
            
        else:
            # Process single exam directory
            success = processor.process_exam_directory(args.exam_dir)
            return 0 if success else 1
            
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())