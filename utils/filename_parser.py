"""
Filename parser for MathRush DataProcessor.
Extracts exam date and metadata from PDF filenames.
"""

import re
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FilenameParser:
    """Parse exam information from PDF filenames."""
    
    def __init__(self):
        """Initialize filename parser with supported patterns."""
        # Supported filename patterns
        self.patterns = [
            # YYYY-MM-DD_ExamType_problems.pdf
            r'(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})_(?P<exam_type>[^_]+)_(?P<file_type>problems|solutions)\.pdf',
            
            # YYYYMMDD_ExamType_problems.pdf  
            r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_(?P<exam_type>[^_]+)_(?P<file_type>problems|solutions)\.pdf',
            
            # YYYY-MM-DD_problems.pdf (exam_type defaults to "exam")
            r'(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})_(?P<file_type>problems|solutions)\.pdf',
            
            # YYYYMMDD_problems.pdf
            r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_(?P<file_type>problems|solutions)\.pdf'
        ]
        
        # Common exam type mappings
        self.exam_type_aliases = {
            'suneung': 'suneung',
            '수능': 'suneung', 
            'mock': 'mock_exam',
            'practice': 'mock_exam',
            '모의고사': 'mock_exam',
            'school': 'school_exam',
            '학교시험': 'school_exam',
            'monthly': 'monthly_exam',
            '월례고사': 'monthly_exam',
            'final': 'final_exam',
            '기말고사': 'final_exam',
            'midterm': 'midterm_exam',
            '중간고사': 'midterm_exam'
        }
    
    def parse_filename(self, filename: str) -> Dict[str, Any]:
        """
        Parse exam information from filename.
        
        Args:
            filename: PDF filename to parse
            
        Returns:
            Dictionary with exam information
        """
        # Remove path if present
        basename = os.path.basename(filename)
        
        # Try each pattern
        for pattern in self.patterns:
            match = re.match(pattern, basename, re.IGNORECASE)
            if match:
                data = match.groupdict()
                
                # Convert to integers
                year = int(data['year'])
                month = int(data['month'])
                day = int(data['day'])
                
                # Validate date
                try:
                    exam_date = datetime(year, month, day)
                    exam_date_str = exam_date.strftime('%Y-%m-%d')
                except ValueError as e:
                    logger.warning(f"Invalid date in filename {filename}: {e}")
                    exam_date_str = f"{year:04d}-{month:02d}-{day:02d}"
                
                # Normalize exam type
                exam_type = data.get('exam_type', 'exam').lower()
                exam_type = self.exam_type_aliases.get(exam_type, exam_type)
                
                # File type
                file_type = data['file_type'].lower()
                
                result = {
                    'original_filename': filename,
                    'exam_date': exam_date_str,
                    'exam_year': year,
                    'exam_month': month,
                    'exam_day': day,
                    'exam_type': exam_type,
                    'file_type': file_type,
                    'base_name': self._get_base_name(basename),
                    'is_valid': True
                }
                
                logger.debug(f"Parsed {filename}: {result}")
                return result
        
        # No pattern matched
        logger.warning(f"Could not parse filename: {filename}")
        return {
            'original_filename': filename,
            'exam_date': None,
            'exam_year': None,
            'exam_month': None,
            'exam_day': None,
            'exam_type': 'unknown',
            'file_type': 'unknown',
            'base_name': basename,
            'is_valid': False,
            'error': 'No matching pattern found'
        }
    
    def _get_base_name(self, filename: str) -> str:
        """
        Get base name for matching problem/solution pairs.
        
        Args:
            filename: PDF filename
            
        Returns:
            Base name without file_type suffix
        """
        # Remove .pdf extension
        name = filename.replace('.pdf', '')
        
        # Remove _problems or _solutions suffix
        if name.endswith('_problems'):
            return name[:-9]  # Remove '_problems'
        elif name.endswith('_solutions'):
            return name[:-10]  # Remove '_solutions'
        
        return name
    
    def find_pdf_pairs(self, directory: str) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Find matching problem/solution PDF pairs in directory.
        
        Args:
            directory: Directory to search for PDF files
            
        Returns:
            List of tuples: (problems_pdf, solutions_pdf, metadata)
        """
        if not os.path.isdir(directory):
            logger.error(f"Directory not found: {directory}")
            return []
        
        # Find all PDF files
        pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        # Parse all filenames
        parsed_files = {}
        invalid_files = []
        
        for pdf_file in pdf_files:
            parsed = self.parse_filename(pdf_file)
            if parsed['is_valid']:
                base_name = parsed['base_name']
                file_type = parsed['file_type']
                
                if base_name not in parsed_files:
                    parsed_files[base_name] = {}
                
                parsed_files[base_name][file_type] = {
                    'filename': pdf_file,
                    'full_path': os.path.join(directory, pdf_file),
                    'metadata': parsed
                }
            else:
                invalid_files.append(pdf_file)
        
        if invalid_files:
            logger.warning(f"Invalid filenames found: {invalid_files}")
        
        # Find complete pairs
        pairs = []
        incomplete_pairs = []
        
        for base_name, files in parsed_files.items():
            if 'problems' in files and 'solutions' in files:
                problems_info = files['problems']
                solutions_info = files['solutions']
                
                # Validate that both files have same exam date
                prob_meta = problems_info['metadata']
                sol_meta = solutions_info['metadata']
                
                if (prob_meta['exam_date'] == sol_meta['exam_date'] and 
                    prob_meta['exam_type'] == sol_meta['exam_type']):
                    
                    # Create combined metadata
                    combined_metadata = {
                        'exam_date': prob_meta['exam_date'],
                        'exam_year': prob_meta['exam_year'],
                        'exam_month': prob_meta['exam_month'],
                        'exam_day': prob_meta['exam_day'],
                        'exam_type': prob_meta['exam_type'],
                        'base_name': base_name,
                        'problems_file': problems_info['filename'],
                        'solutions_file': solutions_info['filename']
                    }
                    
                    pairs.append((
                        problems_info['full_path'],
                        solutions_info['full_path'], 
                        combined_metadata
                    ))
                    
                    logger.info(f"Found valid pair: {base_name}")
                else:
                    logger.warning(f"Date/type mismatch for pair: {base_name}")
                    incomplete_pairs.append(base_name)
            else:
                missing = []
                if 'problems' not in files:
                    missing.append('problems')
                if 'solutions' not in files:
                    missing.append('solutions')
                
                logger.warning(f"Incomplete pair {base_name}: missing {missing}")
                incomplete_pairs.append(base_name)
        
        logger.info(f"Found {len(pairs)} complete PDF pairs")
        if incomplete_pairs:
            logger.warning(f"Incomplete pairs: {incomplete_pairs}")
        
        # Sort pairs by exam date
        pairs.sort(key=lambda x: x[2]['exam_date'])
        
        return pairs
    
    def validate_filename_format(self, filename: str) -> Dict[str, Any]:
        """
        Validate if filename follows expected format.
        
        Args:
            filename: Filename to validate
            
        Returns:
            Validation result with suggestions
        """
        parsed = self.parse_filename(filename)
        
        if parsed['is_valid']:
            return {
                'is_valid': True,
                'message': 'Filename format is valid',
                'parsed_info': parsed
            }
        else:
            # Provide suggestions
            suggestions = [
                "Use format: YYYY-MM-DD_ExamType_problems.pdf or YYYY-MM-DD_ExamType_solutions.pdf",
                "Examples: 2024-06-06_suneung_problems.pdf, 2024-03-15_mock_solutions.pdf",
                "Exam types: suneung, mock, school, monthly, final, midterm",
                "Alternative format: YYYYMMDD_ExamType_problems.pdf"
            ]
            
            return {
                'is_valid': False,
                'message': f'Invalid filename format: {filename}',
                'suggestions': suggestions,
                'error': parsed.get('error', 'Unknown error')
            }


def test_filename_parser():
    """Test filename parser with various formats."""
    parser = FilenameParser()
    
    test_files = [
        "2024-06-06_suneung_problems.pdf",
        "2024-06-06_suneung_solutions.pdf", 
        "2024-03-15_mock_problems.pdf",
        "2024-03-15_mock_solutions.pdf",
        "20240606_school_problems.pdf",
        "20240315_monthly_solutions.pdf",
        "2023-11-16_problems.pdf",
        "invalid_filename.pdf",
        "2024-13-45_impossible_problems.pdf"  # Invalid date
    ]
    
    print("=== Filename Parser Test ===")
    
    for filename in test_files:
        print(f"\nTesting: {filename}")
        result = parser.parse_filename(filename)
        
        if result['is_valid']:
            print(f"  ✅ Valid: {result['exam_date']} - {result['exam_type']} - {result['file_type']}")
        else:
            print(f"  ❌ Invalid: {result.get('error', 'Unknown error')}")
        
        # Test validation
        validation = parser.validate_filename_format(filename)
        if not validation['is_valid']:
            print(f"  💡 Suggestion: {validation['suggestions'][0]}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_filename_parser()