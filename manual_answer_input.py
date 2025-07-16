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
        Process answers from batch input (command line or file).
        
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
            
            # Get exam metadata
            exam_name = problems[0]['exam_name']
            exam_metadata = self.extract_exam_metadata(exam_name)
            
            print(f"\n📚 Exam: {exam_name}")
            print(f"📊 Found {len(problems)} problems")
            print(f"📅 Exam Date: {exam_metadata['exam_date']}")
            print(f"📖 Subject: {exam_metadata['subject']}")
            print(f"🎓 Level: {exam_metadata['level']}")
            
            # Process batch answers
            problem_records = []
            
            print(f"\n{'='*50}")
            print("📝 BATCH ANSWER PROCESSING")
            print(f"{'='*50}")
            
            # For sequential mapping, map answers to problems in order
            if sequential_mapping:
                answer_keys = sorted(batch_answers.keys())
                for i, problem in enumerate(problems):
                    problem_num = problem['problem_number']
                    filename = problem['filename']
                    
                    # Check if we have an answer for this position
                    if i < len(answer_keys):
                        answer_key = answer_keys[i]
                        answer_data = batch_answers[answer_key]
                        problem_type = answer_data['type']
                        correct_answer = answer_data['answer']
                        
                        print(f"📄 Problem {problem_num}: {problem_type} → {correct_answer}")
                    else:
                        print(f"⚠️  Problem {problem_num}: No answer provided, skipping")
                        continue
                        
                    # Create problem record
                    problem_record = {
                        'content': f"[Manual Input] Problem {problem_num} from {exam_name}",
                        'correct_answer': correct_answer,
                        'problem_type': problem_type,
                        'choices': None,
                        'explanation': '',
                        **exam_metadata,
                        'source_info': {
                            'exam_name': exam_name,
                            'problem_number': problem_num,
                            'filename': filename,
                            'file_path': problem['file_path'],
                            'manual_input': True,
                            'batch_input': True,
                            'sequential_mapping': True,
                            'input_timestamp': datetime.now().isoformat()
                        },
                        'tags': ['manual_input', 'batch_input'],
                        'images': [problem['file_path']]
                    }
                    problem_records.append(problem_record)
            else:
                # Direct mapping by problem number
                for problem in problems:
                    problem_num = problem['problem_number']
                    filename = problem['filename']
                    
                    # Check if answer provided for this problem
                    if problem_num not in batch_answers:
                        print(f"⚠️  Problem {problem_num}: No answer provided, skipping")
                        continue
                    
                    answer_data = batch_answers[problem_num]
                    problem_type = answer_data['type']
                    correct_answer = answer_data['answer']
                
                    print(f"📄 Problem {problem_num}: {problem_type} → {correct_answer}")
                    
                    # Create minimal problem record for database
                    problem_record = {
                        'content': f"[Manual Input] Problem {problem_num} from {exam_name}",  # Placeholder
                        'correct_answer': correct_answer,
                        'problem_type': problem_type,
                        'choices': None,  # Will be filled by GPT processing
                        'explanation': '',  # Not needed per requirements
                        **exam_metadata,
                        'source_info': {
                            'exam_name': exam_name,
                            'problem_number': problem_num,
                            'filename': filename,
                            'file_path': problem['file_path'],
                            'manual_input': True,
                            'batch_input': True,
                            'input_timestamp': datetime.now().isoformat()
                        },
                        'tags': ['manual_input', 'batch_input'],
                        'images': [problem['file_path']]
                    }
                    
                    problem_records.append(problem_record)
            
            if not problem_records:
                print("❌ No problems to process")
                return False
            
            # Save to database
            print(f"\n{'='*50}")
            print("💾 SAVING TO DATABASE")
            print(f"{'='*50}")
            
            results = self.db_saver.bulk_insert_problems(problem_records)
            
            # Display results
            print(f"\n📊 Results:")
            print(f"  ✅ Successfully saved: {results['successful']}/{results['total']} problems")
            
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
        Interactive input of answers for entire exam.
        
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
            
            # Get exam metadata
            exam_name = problems[0]['exam_name']
            exam_metadata = self.extract_exam_metadata(exam_name)
            
            print(f"\n📚 Exam: {exam_name}")
            print(f"📊 Found {len(problems)} problems")
            print(f"📅 Exam Date: {exam_metadata['exam_date']}")
            print(f"📖 Subject: {exam_metadata['subject']}")
            print(f"🎓 Level: {exam_metadata['level']}")
            
            # Confirm before proceeding
            confirm = input(f"\n📝 Ready to input answers for {len(problems)} problems? (y/N): ").strip().lower()
            if confirm != 'y':
                print("❌ Operation cancelled")
                return False
            
            # Input answers for each problem
            problem_records = []
            print(f"\n{'='*50}")
            print("📝 ANSWER INPUT PHASE")
            print(f"{'='*50}")
            
            for problem in problems:
                problem_num = problem['problem_number']
                filename = problem['filename']
                
                print(f"\n📄 Processing: {filename}")
                
                # Get problem type
                problem_type = self.get_problem_type_input(problem_num)
                
                # Get correct answer
                correct_answer = self.get_answer_input(problem_num, problem_type)
                
                # Create minimal problem record for database
                problem_record = {
                    'content': f"[Manual Input] Problem {problem_num} from {exam_name}",  # Placeholder
                    'correct_answer': correct_answer,
                    'problem_type': problem_type,
                    'choices': None,  # Will be filled by GPT processing
                    'explanation': '',  # Not needed per requirements
                    **exam_metadata,
                    'source_info': {
                        'exam_name': exam_name,
                        'problem_number': problem_num,
                        'filename': filename,
                        'file_path': problem['file_path'],
                        'manual_input': True,
                        'input_timestamp': datetime.now().isoformat()
                    },
                    'tags': ['manual_input'],
                    'images': [problem['file_path']]
                }
                
                problem_records.append(problem_record)
                
                print(f"  ✅ Problem {problem_num}: {problem_type} → {correct_answer}")
            
            # Save to database
            print(f"\n{'='*50}")
            print("💾 SAVING TO DATABASE")
            print(f"{'='*50}")
            
            results = self.db_saver.bulk_insert_problems(problem_records)
            
            # Display results
            print(f"\n📊 Results:")
            print(f"  ✅ Successfully saved: {results['successful']}/{results['total']} problems")
            
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
  # Interactive mode (default)
  python manual_answer_input.py input/2020-12-03_suneung_가형/
  
  # Single problem input
  python manual_answer_input.py input/2020-12-03_suneung_가형/ --problem 27 --answer 5 --type subjective
  
  # Comma-separated answers (assumes problem order 1,2,3...)
  python manual_answer_input.py input/2020-12-03_suneung_가형/ --answers "1,2,3,4,5"
  
  # From JSON file
  python manual_answer_input.py input/2020-12-03_suneung_가형/ --answers-file answers.json
  
  # List available exams
  python manual_answer_input.py input/ --list
  
  # Input answers for exam by name (searches in input/)
  python manual_answer_input.py --exam 2020-12-03_suneung_가형 --answers "1,2,3,4,5"
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