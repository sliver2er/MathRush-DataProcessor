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
from typing import Dict, List, Tuple
from pathlib import Path
import time

# Import utilities and processors
try:
    from processors import DatabaseSaver, GPTExtractor
    from utils.filename_parser import FilenameParser
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from processors import DatabaseSaver, GPTExtractor
    from utils.filename_parser import FilenameParser

logger = logging.getLogger(__name__)


class SimpleProcessor:
    """Simple processor for manually segmented problem images."""
    
    def __init__(self):
        """Initialize the simple processor."""
        self.db_saver = DatabaseSaver()
        self.gpt_extractor = GPTExtractor()
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
        
        # Patterns for different file types (support png, jpg, jpeg)
        problem_pattern = re.compile(r'(.+)_problem_(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)
        content_pattern = re.compile(r'(.+)_problem_(\d+)_(?:diagram|content|graph|figure).*\.(png|jpg|jpeg)$', re.IGNORECASE)
        
        problems = {}
        
        # Supported image extensions
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        
        # First pass: find all problem images
        for ext in image_extensions:
            for file_path in exam_path.glob(ext):
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
        for ext in image_extensions:
            for file_path in exam_path.glob(ext):
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
    
    def process_problem_images(self, problems: List[Dict], exam_metadata: Dict) -> Tuple[List[Dict], List[Dict]]:
        """
        Process problem images through GPT extraction and create database records.
        
        Args:
            problems: List of problem data from directory scan
            exam_metadata: Exam metadata for all problems
            
        Returns:
            Tuple of (processed_problems, failed_problems)
        """
        processed_problems = []
        failed_problems = []
        
        for problem in problems:
            problem_num = problem['problem_number']
            problem_image = problem['problem_image']
            math_content_images = problem['math_content_images']
            
            try:
                logger.info(f"Processing Problem {problem_num}: {os.path.basename(problem_image)}")
                
                # Extract content using GPT extractor
                extracted_data = self.gpt_extractor.extract_from_image(
                    problem_image, 
                    math_content_images if math_content_images else None
                )
                
                # Create new record with GPT content (no manual answers yet)
                problem_record = {
                    # GPT extracted fields
                    'content': extracted_data.get('content', f'[Extraction Failed] Problem {problem_num}'),
                    'problem_type': extracted_data.get('problem_type', 'subjective'),
                    'choices': extracted_data.get('choices'),
                    
                    # Empty answer fields (to be filled later by manual_answer_input.py)
                    'correct_answer': '',
                    'explanation': '',
                    
                    # Metadata fields from exam metadata
                    'level': exam_metadata.get('level', '고3'),
                    'subject': exam_metadata.get('subject', '수학영역'),
                    'chapter': exam_metadata.get('chapter', ''),
                    'difficulty': exam_metadata.get('difficulty', 'medium'),
                    'tags': ['gpt_extracted'],
                    
                    # Source and image info
                    'source_info': {
                        'exam_name': problem['exam_name'],
                        'problem_number': problem_num,
                        'exam_type': exam_metadata.get('exam_type', '수능'),
                        'exam_date': exam_metadata.get('exam_date', ''),
                        'filename': os.path.basename(problem_image),
                        'file_path': problem_image,
                        'problem_image': problem_image,
                        'math_content_images': math_content_images,
                        'gpt_processed': True,
                        'gpt_processed_timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
                    },
                    'images': [problem_image] + math_content_images
                }
                
                processed_problems.append(problem_record)
                
                logger.info(f"  ✅ Problem {problem_num}: {extracted_data.get('problem_type', 'unknown')} - Content: {str(extracted_data.get('content', ''))[:50]}...")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to process Problem {problem_num}: {e}")
                # Track failed problems separately
                failed_problems.append({
                    'problem_number': problem_num,
                    'problem_image': problem_image,
                    'error': str(e),
                    'exam_name': problem['exam_name']
                })
        
        return processed_problems, failed_problems
    
    def insert_problems_to_database(self, processed_problems: List[Dict]) -> Dict[str, int]:
        """
        Insert new problem records to database.
        
        Args:
            processed_problems: List of processed problem records
            
        Returns:
            Dictionary with insertion statistics
        """
        stats = {
            'inserted': 0,
            'failed': 0,
            'total': len(processed_problems)
        }
        
        # Use bulk insert for efficiency
        results = self.db_saver.bulk_insert_problems(processed_problems)
        
        stats['inserted'] = results['successful']
        stats['failed'] = results['failed']
        
        return stats
    
    def process_exam_directory(self, exam_dir: str) -> bool:
        """
        Process entire exam directory and create database records with GPT content.
        
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
            
            # Step 2: Extract exam metadata
            print("\n📋 STEP 2: Extracting exam metadata...")
            exam_metadata = self.filename_parser.parse_exam_filename(exam_name)
            print(f"✅ Exam metadata: {exam_metadata['exam_type']} ({exam_metadata['exam_date']})")
            
            # Step 3: Process images through GPT
            print("\n🤖 STEP 3: Processing images through GPT...")
            processed_problems, failed_problems = self.process_problem_images(problems, exam_metadata)
            
            # Step 4: Insert to database
            print("\n💾 STEP 4: Inserting records to database...")
            stats = self.insert_problems_to_database(processed_problems)
            
            # Add failed problems to stats
            stats['processing_failed'] = len(failed_problems)
            
            # Display final results
            print(f"\n{'='*60}")
            print(f"📊 PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"➕ New records: {stats['inserted']}")
            print(f"❌ Database failures: {stats['failed']}")
            print(f"🔄 Processing failures: {stats['processing_failed']}")
            print(f"📈 Total processed: {stats['total']}")
            
            # Show failed problems if any
            if failed_problems:
                print(f"\n⚠️  Failed to process {len(failed_problems)} problems:")
                for failed in failed_problems:
                    print(f"   Problem {failed['problem_number']}: {failed['error']}")
            
            total_failures = stats['failed'] + stats['processing_failed']
            success_rate = stats['inserted'] / len(problems) * 100
            print(f"📊 Success rate: {success_rate:.1f}%")
            
            print(f"\n💡 Next step: Run manual_answer_input.py to add answers to these records")
            
            return total_failures == 0
            
        except Exception as e:
            logger.error(f"Error processing exam directory {exam_dir}: {e}")
            print(f"❌ Error: {e}")
            return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Simple Image Processor for MathRush DataProcessor - Extracts content from images and creates database records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific exam directory (creates records with GPT content)
  python simple_processor.py input/2020-12-03_suneung_가형/
  
  # Process with verbose logging
  python simple_processor.py input/2020-12-03_suneung_가형/ --verbose
  
  # Process multiple exams in input directory
  python simple_processor.py input/ --recursive

Workflow:
  1. Run simple_processor.py to extract content and create records
  2. Run manual_answer_input.py to add answers to existing records
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