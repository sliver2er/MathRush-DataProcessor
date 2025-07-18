#!/usr/bin/env python3
"""
GPT Image Processor for MathRush DataProcessor.
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


class GptProcessor:
    """Simple processor for manually segmented problem images."""
    
    def __init__(self):
        """Initialize the GPT processor."""
        self.db_saver = DatabaseSaver()
        self.gpt_extractor = GPTExtractor()
        self.filename_parser = FilenameParser()
        
        # Test database connection
        if not self.db_saver.test_connection():
            raise ConnectionError("Failed to connect to database")
            
        logger.info("GPT Processor initialized")
    
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
        problem_pattern = re.compile(r'(.+)_problem_(\d+)\\.(png|jpg|jpeg)$', re.IGNORECASE)
        content_pattern = re.compile(r'(.+)_problem_(\d+)_(?:diagram|content|graph|figure).*\\.(png|jpg|jpeg)$', re.IGNORECASE)
        
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
    
    def find_failed_extractions(self, exam_name: str) -> List[Dict]:
        """
        Find existing records with failed extractions.
        
        Args:
            exam_name: Name of the exam
            
        Returns:
            List of failed extraction records
        """
        try:
            result = self.db_saver.client.table(self.db_saver.table_name)\
                .select("*")\
                .eq("source_info->>exam_name", exam_name)\
                .execute()
            
            failed_records = []
            for record in result.data:
                content = record.get('content', '')
                if 'Extraction Failed' in content:
                    source_info = record.get('source_info', {})
                    failed_records.append({
                        'record_id': record['id'],
                        'problem_number': source_info.get('problem_number'),
                        'content': content,
                        'correct_answer': record.get('correct_answer', '')
                    })
            
            return failed_records
            
        except Exception as e:
            logger.error(f"Error finding failed extractions: {e}")
            return []
    
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
                    math_content_images if math_content_images else None,
                    problem_number=problem_num
                )
                
                # Check if extraction failed
                if extracted_data.get('extraction_error', False):
                    raise Exception(f"GPT extraction failed: {extracted_data.get('content', 'Unknown error')}")
                
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
                    'correct_rate': None,  # For future difficulty determination based on answer rates
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
                    'images': math_content_images  # Only store math content images (diagrams)
                }
                
                processed_problems.append(problem_record)
                
                # Show success with more details
                content_preview = str(extracted_data.get('content', ''))[:50].replace('\n', ' ')
                logger.info(f"  ✅ SUCCESS Problem {problem_num}: {extracted_data.get('problem_type', 'unknown')} - {content_preview}...")
                
            except Exception as e:
                logger.error(f"  ❌ FAILED Problem {problem_num}: {e}")
                # Track failed problems separately
                failed_problems.append({
                    'problem_number': problem_num,
                    'problem_image': problem_image,
                    'error': str(e),
                    'exam_name': problem['exam_name']
                })
        
        return processed_problems, failed_problems
    
    def upsert_problems_to_database(self, processed_problems: List[Dict]) -> Dict[str, int]:
        """
        Upsert problem records to database (insert or update).
        
        Args:
            processed_problems: List of processed problem records
            
        Returns:
            Dictionary with upsert statistics
        """
        stats = {
            'inserted': 0,
            'failed': 0,
            'total': len(processed_problems)
        }
        
        # Use bulk upsert for efficiency
        results = self.db_saver.bulk_upsert_problems(processed_problems)
        
        stats['inserted'] = results['successful']
        stats['failed'] = results['failed']
        
        return stats
    
    def update_failed_records(self, processed_problems: List[Dict]) -> Dict[str, int]:
        """
        Update existing database records with new GPT content.
        
        Args:
            processed_problems: List of processed problem records
            
        Returns:
            Dictionary with update statistics
        """
        stats = {
            'inserted': 0,  # Keep same naming for consistency
            'failed': 0,
            'total': len(processed_problems)
        }
        
        for problem in processed_problems:
            try:
                # Find the record ID from source_info
                source_info = problem.get('source_info', {})
                exam_name = source_info.get('exam_name')
                problem_number = source_info.get('problem_number')
                
                # Get the record ID by querying the database
                result = self.db_saver.client.table(self.db_saver.table_name)\
                    .select("id")\
                    .eq("source_info->>exam_name", exam_name)\
                    .eq("source_info->>problem_number", problem_number)\
                    .execute()
                
                if result.data:
                    record_id = result.data[0]['id']
                    
                    # Update only the content-related fields
                    update_data = {
                        'content': problem.get('content'),
                        'problem_type': problem.get('problem_type'),
                        'choices': problem.get('choices'),
                        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S')
                    }
                    
                    # Update the record
                    update_result = self.db_saver.client.table(self.db_saver.table_name)\
                        .update(update_data)\
                        .eq('id', record_id)\
                        .execute()
                    
                    if update_result.data:
                        stats['inserted'] += 1
                        logger.info(f"Updated record for Problem {problem_number}")
                    else:
                        stats['failed'] += 1
                        logger.error(f"Failed to update record for Problem {problem_number}")
                else:
                    stats['failed'] += 1
                    logger.error(f"Could not find record for Problem {problem_number}")
                    
            except Exception as e:
                stats['failed'] += 1
                logger.error(f"Error updating record: {e}")
        
        return stats
    
    def process_exam_directory(self, exam_dir: str, retry_failed: bool = False) -> bool:
        """
        Process entire exam directory and upsert database records with GPT content.
        
        Args:
            exam_dir: Path to exam directory
            retry_failed: If True, only retry failed extractions (legacy parameter)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            exam_dir_path = Path(exam_dir)
            exam_name = exam_dir_path.name
            
            print(f"\\n{'='*60}")
            print(f" PROCESSING EXAM: {exam_name}")
            print(f" Directory: {exam_dir}")
            print(f"{'='*60}")
            
            # Step 1: Scan directory for images
            print("\\n STEP 1: Scanning for problem images...")
            problems = self.scan_exam_directory(exam_dir)
            
            if not problems:
                print("❌ No problems found in directory")
                return False
            
            print(f"✅ Found {len(problems)} problems")
            
            # Step 2: Extract exam metadata
            print("\\n STEP 2: Extracting exam metadata...")
            exam_metadata = self.filename_parser.parse_exam_filename(exam_name)
            print(f"✅ Exam metadata: {exam_metadata['exam_type']} ({exam_metadata['exam_date']})")
            
            # Filter for retry_failed if requested (legacy support)
            if retry_failed:
                print("\\n STEP 2.5: Checking for failed extractions...")
                failed_records = self.find_failed_extractions(exam_name)
                if not failed_records:
                    print("✅ No failed extractions found!")
                    return True
                print(f" Found {len(failed_records)} failed extractions to retry")
                
                # Filter problems to only those with failed extractions
                failed_problem_numbers = [record['problem_number'] for record in failed_records]
                problems = [p for p in problems if p['problem_number'] in failed_problem_numbers]
                print(f" Filtered to {len(problems)} problems for retry")
            
            # Step 3: Process images through GPT
            print("\\n STEP 3: Processing images through GPT...")
            processed_problems, failed_problems = self.process_problem_images(problems, exam_metadata)
            
            # Step 4: Upsert to database (insert or update automatically)
            print("\\n STEP 4: Upserting records to database...")
            stats = self.upsert_problems_to_database(processed_problems)
            
            # Add failed problems to stats
            stats['processing_failed'] = len(failed_problems)
            
            # Display final results
            print(f"\\n{'='*60}")
            print(f" PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"✅ Successful extractions: {stats['inserted']}")
            print(f"❌ Database failures: {stats['failed']}")
            print(f" Extraction failures: {stats['processing_failed']}")
            print(f" Total attempted: {len(problems)}")
            
            # Show failed problems if any
            if failed_problems:
                print(f"\\n⚠️  Failed to extract {len(failed_problems)} problems:")
                for failed in failed_problems:
                    error_msg = failed['error']
                    if len(error_msg) > 80:
                        error_msg = error_msg[:77] + "..."
                    print(f"   Problem {failed['problem_number']}: {error_msg}")
            
            total_failures = stats['failed'] + stats['processing_failed']
            success_rate = (len(problems) - total_failures) / len(problems) * 100 if len(problems) > 0 else 0
            print(f" Success rate: {success_rate:.1f}%")
            
            print(f"\\n Next step: Run manual_answer_input.py to add answers to these records")
            
            return total_failures == 0
            
        except Exception as e:
            logger.error(f"Error processing exam directory {exam_dir}: {e}")
            print(f"❌ Error: {e}")
            return False


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="GPT Image Processor for MathRush DataProcessor - Extracts content from images and creates database records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific exam directory (upserts records with GPT content)
  python gpt_processor.py input/2020-12-03_suneung_가형/
  
  # Retry only failed extractions (legacy option, upsert handles this automatically)
  python gpt_processor.py input/2020-12-03_suneung_가형/ --retry-failed
  
  # Process with verbose logging
  python gpt_processor.py input/2020-12-03_suneung_가형/ --verbose
  
  # Process multiple exams in input directory
  python gpt_processor.py input/ --recursive

Workflow:
  1. Run gpt_processor.py to extract content and upsert records (creates or updates automatically)
  2. Run manual_answer_input.py to add answers to existing records
  3. Re-run gpt_processor.py anytime to update existing records with new GPT extractions
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
    
    parser.add_argument(
        "--retry-failed", "-rf",
        action="store_true",
        help="Only retry failed extractions (updates existing records)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        processor = GptProcessor()
        
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
            
            print(f" Found {len(exam_dirs)} exam directories to process")
            
            success_count = 0
            for exam_dir in exam_dirs:
                if processor.process_exam_directory(exam_dir, args.retry_failed):
                    success_count += 1
                print()  # Empty line between exams
            
            print(f" Final Results: {success_count}/{len(exam_dirs)} exams processed successfully")
            return 0 if success_count == len(exam_dirs) else 1
            
        else:
            # Process single exam directory
            success = processor.process_exam_directory(args.exam_dir, args.retry_failed)
            return 0 if success else 1
            
    except KeyboardInterrupt:
        print("\\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())