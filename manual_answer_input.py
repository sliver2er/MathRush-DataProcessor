#!/usr/bin/env python3
"""
Manual Answer Input Utility for MathRush DataProcessor.
Interactive utility for manually inputting correct answers for entire exams.
"""

import os
import sys
import argparse
import re
import logging
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Import settings and database
try:
    from config.settings import settings
    from processors import DatabaseSaver
    from utils.filename_parser import FilenameParser
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from config.settings import settings
    from processors import DatabaseSaver
    from utils.filename_parser import FilenameParser

logger = logging.getLogger(__name__)


class ManualAnswerInput:
    """Interactive utility for manually inputting exam answers."""
    
    def __init__(self):
        """Initialize the manual answer input utility."""
        self.db_saver = DatabaseSaver()
        self.filename_parser = FilenameParser()
        
        # Test database connection
        if not self.db_saver.test_connection():
            raise ConnectionError("Failed to connect to database")
            
        logger.info("Manual Answer Input utility initialized")
    
    def scan_exam_directory(self, exam_dir: str) -> List[Dict[str, str]]:
        """
        Scan exam directory for problem images.
        
        Args:
            exam_dir: Path to exam directory
            
        Returns:
            List of problem info dictionaries
        """
        exam_path = Path(exam_dir)
        if not exam_path.exists():
            raise FileNotFoundError(f"Exam directory not found: {exam_dir}")
        
        # Find all problem images (not diagram/content images) - support multiple formats
        problem_pattern = re.compile(r'(.+)_problem_(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)
        problems = []
        
        # Supported image extensions
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        
        for ext in image_extensions:
            for file_path in exam_path.glob(ext):
                match = problem_pattern.match(file_path.name)
                if match:
                    exam_name = match.group(1)
                    problem_number = int(match.group(2))
                    
                    problems.append({
                        'exam_name': exam_name,
                        'problem_number': problem_number,
                        'file_path': str(file_path),
                        'filename': file_path.name
                    })
        
        # Sort by problem number
        problems.sort(key=lambda x: x['problem_number'])
        
        logger.info(f"Found {len(problems)} problems in {exam_dir}")
        return problems
    
    def extract_exam_metadata(self, exam_name: str) -> Dict[str, str]:
        """
        Extract exam metadata from exam name.
        
        Args:
            exam_name: Exam identifier (e.g., "2020-12-03_suneung")
            
        Returns:
            Dictionary with exam metadata
        """
        # Parse exam metadata from filename
        metadata = self.filename_parser.parse_exam_filename(exam_name)
        
        # Add default values for required fields
        exam_metadata = {
            'exam_type': metadata.get('exam_type', '수능'),
            'exam_date': metadata.get('exam_date', ''),
            'level': metadata.get('level', '고3'),
            'subject': metadata.get('subject', '수학영역'),
            'chapter': metadata.get('chapter', ''),
            'difficulty': 'medium'
        }
        
        return exam_metadata
    
    def get_problem_type_input(self, problem_number: int) -> str:
        """
        Get problem type from user input.
        
        Args:
            problem_number: Problem number
            
        Returns:
            Problem type ('multiple_choice' or 'subjective')
        """
        while True:
            print(f"  Problem {problem_number} type:")
            print("    1. Multiple Choice (객관식)")
            print("    2. Subjective (주관식)")
            
            choice = input("  Enter choice (1/2): ").strip()
            
            if choice == '1':
                return 'multiple_choice'
            elif choice == '2':
                return 'subjective'
            else:
                print("  ❌ Invalid choice. Please enter 1 or 2.")
    
    def get_answer_input(self, problem_number: int, problem_type: str) -> str:
        """
        Get answer from user input.
        
        Args:
            problem_number: Problem number
            problem_type: Type of problem
            
        Returns:
            Correct answer as string
        """
        while True:
            if problem_type == 'multiple_choice':
                answer = input(f"  Problem {problem_number} answer (1-5): ").strip()
                if answer in ['1', '2', '3', '4', '5']:
                    return answer
                else:
                    print("  ❌ Multiple choice answer must be 1, 2, 3, 4, or 5")
            else:
                answer = input(f"  Problem {problem_number} answer: ").strip()
                if answer:
                    return answer
                else:
                    print("  ❌ Answer cannot be empty")
    
    def process_batch_answers(self, exam_dir: str, batch_answers: Dict[int, Dict[str, str]], sequential_mapping: bool = False) -> bool:
        """
        Update existing database records with manual answers.
        
        Args:
            exam_dir: Path to exam directory
            batch_answers: Dictionary mapping problem numbers to answer data
            sequential_mapping: If True, map answers sequentially to problems found
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Scan directory for problems
            problems = self.scan_exam_directory(exam_dir)
            
            if not problems:
                print(f"❌ No problems found in {exam_dir}")
                return False
            
            # Get exam name
            exam_name = problems[0]['exam_name']
            
            print(f"\n📚 Exam: {exam_name}")
            print(f"📊 Found {len(problems)} problems in directory")
            
            # Find existing records in database
            print(f"\n🔍 Looking for existing records in database...")
            existing_records = self.find_existing_records(exam_name)
            
            if not existing_records:
                print(f"❌ No existing records found for exam: {exam_name}")
                print(f"💡 Run: python simple_processor.py {exam_dir} first")
                return False
            
            print(f"✅ Found {len(existing_records)} existing records")
            
            # Process batch answers
            updates = []
            
            print(f"\n{'='*50}")
            print("📝 UPDATING RECORDS WITH ANSWERS")
            print(f"{'='*50}")
            
            # For sequential mapping, map answers to problems in order
            if sequential_mapping:
                answer_keys = sorted(batch_answers.keys())
                for i, problem in enumerate(problems):
                    problem_num = problem['problem_number']
                    
                    # Check if we have an answer for this position
                    if i < len(answer_keys):
                        answer_key = answer_keys[i]
                        answer_data = batch_answers[answer_key]
                        correct_answer = answer_data['answer']
                        
                        # Check if record exists
                        if problem_num in existing_records:
                            record_id = existing_records[problem_num]['id']
                            updates.append({
                                'id': record_id,
                                'problem_number': problem_num,
                                'correct_answer': correct_answer,
                                'answer_type': answer_data['type']
                            })
                            print(f"📄 Problem {problem_num}: {answer_data['type']} → {correct_answer}")
                        else:
                            print(f"⚠️  Problem {problem_num}: No existing record found, skipping")
                    else:
                        print(f"⚠️  Problem {problem_num}: No answer provided, skipping")
            else:
                # Direct mapping by problem number
                for problem_num, answer_data in batch_answers.items():
                    correct_answer = answer_data['answer']
                    
                    # Check if record exists
                    if problem_num in existing_records:
                        record_id = existing_records[problem_num]['id']
                        updates.append({
                            'id': record_id,
                            'problem_number': problem_num,
                            'correct_answer': correct_answer,
                            'answer_type': answer_data['type']
                        })
                        print(f"📄 Problem {problem_num}: {answer_data['type']} → {correct_answer}")
                    else:
                        print(f"⚠️  Problem {problem_num}: No existing record found, skipping")
            
            if not updates:
                print("❌ No records to update")
                return False
            
            # Update database records
            print(f"\n{'='*50}")
            print("💾 UPDATING DATABASE")
            print(f"{'='*50}")
            
            results = self.update_existing_records(updates)
            
            # Display results
            print(f"\n📊 Results:")
            print(f"  ✅ Successfully updated: {results['successful']}/{results['total']} problems")
            
            if results['failed'] > 0:
                print(f"  ❌ Failed: {results['failed']} problems")
                if results['errors']:
                    print("  🔍 Errors:")
                    for error in results['errors'][:5]:  # Show first 5 errors
                        print(f"    - {error}")
            
            return results['successful'] > 0
            
        except Exception as e:
            logger.error(f"Error in batch answer processing: {e}")
            print(f"❌ Error: {e}")
            return False
    
    def input_answers_for_exam(self, exam_dir: str) -> bool:
        """
        Interactive input of answers for existing exam records.
        
        Args:
            exam_dir: Path to exam directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Scan directory for problems
            problems = self.scan_exam_directory(exam_dir)
            
            if not problems:
                print(f"❌ No problems found in {exam_dir}")
                return False
            
            # Get exam name
            exam_name = problems[0]['exam_name']
            
            print(f"\n📚 Exam: {exam_name}")
            print(f"📊 Found {len(problems)} problems in directory")
            
            # Find existing records in database
            print(f"\n🔍 Looking for existing records in database...")
            existing_records = self.find_existing_records(exam_name)
            
            if not existing_records:
                print(f"❌ No existing records found for exam: {exam_name}")
                print(f"💡 Run: python simple_processor.py {exam_dir} first")
                return False
            
            print(f"✅ Found {len(existing_records)} existing records")
            
            # Confirm before proceeding
            confirm = input(f"\n📝 Ready to input answers for {len(existing_records)} existing records? (y/N): ").strip().lower()
            if confirm != 'y':
                print("❌ Operation cancelled")
                return False
            
            # Input answers for each existing record
            updates = []
            print(f"\n{'='*50}")
            print("📝 INTERACTIVE ANSWER INPUT")
            print(f"{'='*50}")
            
            for problem_num in sorted(existing_records.keys()):
                record = existing_records[problem_num]
                record_id = record['id']
                
                # Show problem content preview
                content_preview = record.get('content', '')[:100] + "..."
                print(f"\n📄 Problem {problem_num}:")
                print(f"   Content: {content_preview}")
                print(f"   Current answer: {record.get('correct_answer', '(empty)')}")
                
                # Get problem type
                problem_type = self.get_problem_type_input(problem_num)
                
                # Get correct answer
                correct_answer = self.get_answer_input(problem_num, problem_type)
                
                updates.append({
                    'id': record_id,
                    'problem_number': problem_num,
                    'correct_answer': correct_answer,
                    'answer_type': problem_type
                })
                
                print(f"  ✅ Problem {problem_num}: {problem_type} → {correct_answer}")
            
            if not updates:
                print("❌ No updates to process")
                return False
            
            # Update database records
            print(f"\n{'='*50}")
            print("💾 UPDATING DATABASE")
            print(f"{'='*50}")
            
            results = self.update_existing_records(updates)
            
            # Display results
            print(f"\n📊 Results:")
            print(f"  ✅ Successfully updated: {results['successful']}/{results['total']} problems")
            
            if results['failed'] > 0:
                print(f"  ❌ Failed: {results['failed']} problems")
                if results['errors']:
                    print("  🔍 Errors:")
                    for error in results['errors'][:5]:  # Show first 5 errors
                        print(f"    - {error}")
            
            return results['successful'] > 0
            
        except Exception as e:
            logger.error(f"Error in answer input process: {e}")
            print(f"❌ Error: {e}")
            return False
    
    def list_exams(self, input_dir: str) -> List[str]:
        """
        List available exam directories.
        
        Args:
            input_dir: Base input directory
            
        Returns:
            List of exam directory names
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            return []
        
        exam_dirs = []
        for item in input_path.iterdir():
            if item.is_dir():
                # Check if directory contains problem images
                problem_files = list(item.glob("*_problem_*.png"))
                if problem_files:
                    exam_dirs.append(item.name)
        
        return sorted(exam_dirs)
    
    def find_existing_records(self, exam_name: str) -> Dict[int, Dict[str, any]]:
        """
        Find existing records for this exam in the database.
        
        Args:
            exam_name: Name of the exam
            
        Returns:
            Dictionary mapping problem numbers to database records
        """
        try:
            # Query database for records with this exam name using direct column
            result = self.db_saver.client.table(self.db_saver.table_name)\
                .select("*")\
                .eq("exam_name", exam_name)\
                .execute()
            
            if not result.data:
                logger.warning(f"No existing records found for exam: {exam_name}")
                return {}
            
            # Organize by problem number
            existing_records = {}
            for record in result.data:
                problem_number = record.get('problem_number')
                
                if problem_number:
                    existing_records[problem_number] = record
            
            logger.info(f"Found existing records for {len(existing_records)} problems in exam: {exam_name}")
            return existing_records
            
        except Exception as e:
            logger.error(f"Error querying existing records for {exam_name}: {e}")
            # Fallback to old JSON query method for backward compatibility
            try:
                logger.info("Trying fallback JSON query method...")
                result = self.db_saver.client.table(self.db_saver.table_name)\
                    .select("*")\
                    .eq("source_info->>exam_name", exam_name)\
                    .execute()
                
                if not result.data:
                    return {}
                
                # Organize by problem number from source_info
                existing_records = {}
                for record in result.data:
                    source_info = record.get('source_info', {})
                    problem_number = source_info.get('problem_number')
                    
                    if problem_number:
                        existing_records[problem_number] = record
                
                logger.info(f"Found existing records for {len(existing_records)} problems in exam: {exam_name} (fallback)")
                return existing_records
                
            except Exception as fallback_e:
                logger.error(f"Fallback query also failed for {exam_name}: {fallback_e}")
                return {}
    
    def update_existing_records(self, updates: List[Dict[str, any]]) -> Dict[str, int]:
        """
        Update existing records with manual answers.
        
        Args:
            updates: List of update data with id, problem_number, correct_answer, etc.
            
        Returns:
            Dictionary with update results
        """
        results = {
            'total': len(updates),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        for update in updates:
            try:
                record_id = update['id']
                update_data = {
                    'correct_answer': update['correct_answer'],
                    'updated_at': datetime.now().isoformat()
                }
                
                # Add manual answer tags
                result = self.db_saver.client.table(self.db_saver.table_name)\
                    .update(update_data)\
                    .eq('id', record_id)\
                    .execute()
                
                if result.data:
                    results['successful'] += 1
                    logger.debug(f"Updated record ID {record_id} with answer: {update['correct_answer']}")
                else:
                    results['failed'] += 1
                    error_msg = f"Problem {update['problem_number']}: Update returned no data"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
                    
            except Exception as e:
                results['failed'] += 1
                error_msg = f"Problem {update['problem_number']}: {str(e)}"
                results['errors'].append(error_msg)
                logger.error(f"Error updating record {record_id}: {e}")
        
        logger.info(f"Update completed: {results['successful']}/{results['total']} successful")
        return results
    
    def parse_answers_input(self, answers_input: str, format_type: str = 'comma') -> Dict[int, Dict[str, str]]:
        """
        Parse answers from different input formats.
        
        Args:
            answers_input: Input string containing answers
            format_type: Format type ('comma', 'json')
            
        Returns:
            Dictionary mapping problem numbers to answer data
        """
        if format_type == 'comma':
            return self._parse_comma_answers(answers_input)
        elif format_type == 'json':
            return self._parse_json_answers(answers_input)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _parse_comma_answers(self, answers_str: str) -> Dict[int, Dict[str, str]]:
        """
        Parse comma-separated answers.
        
        Args:
            answers_str: Comma-separated answers (e.g., '1,2,3,4,5')
            
        Returns:
            Dictionary mapping problem numbers to answer data
        """
        answers = {}
        answer_list = [ans.strip() for ans in answers_str.split(',')]
        
        for i, answer in enumerate(answer_list, 1):
            if not answer:
                continue
                
            # Determine if it's multiple choice (1-5) or subjective
            if answer in ['1', '2', '3', '4', '5']:
                problem_type = 'multiple_choice'
            else:
                problem_type = 'subjective'
            
            answers[i] = {
                'answer': answer,
                'type': problem_type
            }
        
        return answers
    
    def _parse_json_answers(self, json_str: str) -> Dict[int, Dict[str, str]]:
        """
        Parse JSON format answers.
        
        Args:
            json_str: JSON string with answer data
            
        Returns:
            Dictionary mapping problem numbers to answer data
        """
        try:
            data = json.loads(json_str)
            answers = {}
            
            for prob_num_str, answer_data in data.items():
                prob_num = int(prob_num_str)
                
                if isinstance(answer_data, dict):
                    # Full format: {"1": {"answer": "3", "type": "multiple_choice"}}
                    answers[prob_num] = {
                        'answer': answer_data.get('answer', ''),
                        'type': answer_data.get('type', 'subjective')
                    }
                else:
                    # Simple format: {"1": "3"}
                    answer = str(answer_data)
                    if answer in ['1', '2', '3', '4', '5']:
                        problem_type = 'multiple_choice'
                    else:
                        problem_type = 'subjective'
                    
                    answers[prob_num] = {
                        'answer': answer,
                        'type': problem_type
                    }
            
            return answers
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    def parse_answers_file(self, file_path: str) -> Dict[int, Dict[str, str]]:
        """
        Parse answers from a file.
        
        Args:
            file_path: Path to answers file
            
        Returns:
            Dictionary mapping problem numbers to answer data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Answers file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Determine format based on file extension and content
        if file_path.suffix.lower() == '.json' or content.startswith('{'):
            return self.parse_answers_input(content, 'json')
        else:
            # Assume comma-separated format
            return self.parse_answers_input(content, 'comma')


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Manual Answer Input for MathRush DataProcessor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (updates existing records)
  python manual_answer_input.py input/2020-12-03_suneung_가형/
  
  # Single problem answer update
  python manual_answer_input.py input/2020-12-03_suneung_가형/ --problem 27 --answer 5 --type subjective
  
  # Comma-separated answers (updates existing records in order)
  python manual_answer_input.py input/2020-12-03_suneung_가형/ --answers "1,2,3,4,5"
  
  # From JSON file
  python manual_answer_input.py input/2020-12-03_suneung_가형/ --answers-file answers.json
  
  # List available exams
  python manual_answer_input.py input/ --list

Workflow:
  1. Run simple_processor.py to extract content and create records
  2. Run manual_answer_input.py to add answers to existing records
  
Note: This tool now UPDATES existing records instead of creating new ones.
"""
    )
    
    parser.add_argument(
        "input_path",
        nargs='?',
        help="Exam directory path or base input directory"
    )
    
    parser.add_argument(
        "--exam",
        help="Exam name (searches in input/ directory)"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available exam directories"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Answer input options
    parser.add_argument(
        "--problem",
        type=int,
        help="Specific problem number (use with --answer and --type)"
    )
    
    parser.add_argument(
        "--answer",
        help="Answer for the problem (use with --problem)"
    )
    
    parser.add_argument(
        "--type",
        choices=['multiple_choice', 'subjective'],
        help="Problem type (use with --problem and --answer)"
    )
    
    parser.add_argument(
        "--answers",
        help="Comma-separated answers for all problems (e.g., '1,2,3,4,5')"
    )
    
    parser.add_argument(
        "--answers-file",
        help="Path to file containing answers (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        answer_input = ManualAnswerInput()
        
        # List mode
        if args.list:
            input_dir = args.input_path or "input"
            exams = answer_input.list_exams(input_dir)
            
            if exams:
                print(f"📚 Available exams in {input_dir}:")
                for exam in exams:
                    print(f"  - {exam}")
            else:
                print(f"❌ No exam directories found in {input_dir}")
            return 0
        
        # Determine exam directory
        if args.exam:
            exam_dir = f"input/{args.exam}"
        elif args.input_path:
            exam_dir = args.input_path
        else:
            parser.print_help()
            return 1
        
        # Validate argument combinations
        if args.problem is not None:
            if not args.answer or not args.type:
                print("❌ When using --problem, both --answer and --type are required")
                return 1
            if args.answers or args.answers_file:
                print("❌ Cannot use --problem with --answers or --answers-file")
                return 1
        
        if args.answers and args.answers_file:
            print("❌ Cannot use both --answers and --answers-file")
            return 1
        
        # Process based on input mode
        if args.problem is not None:
            # Single problem mode
            batch_answers = {
                args.problem: {
                    'answer': args.answer,
                    'type': args.type
                }
            }
            success = answer_input.process_batch_answers(exam_dir, batch_answers)
            
        elif args.answers:
            # Comma-separated answers mode
            try:
                batch_answers = answer_input.parse_answers_input(args.answers, 'comma')
                success = answer_input.process_batch_answers(exam_dir, batch_answers, sequential_mapping=True)
            except Exception as e:
                print(f"❌ Error parsing answers: {e}")
                return 1
                
        elif args.answers_file:
            # Answer file mode
            try:
                batch_answers = answer_input.parse_answers_file(args.answers_file)
                success = answer_input.process_batch_answers(exam_dir, batch_answers)
            except Exception as e:
                print(f"❌ Error parsing answers file: {e}")
                return 1
                
        else:
            # Interactive mode (default)
            success = answer_input.input_answers_for_exam(exam_dir)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())