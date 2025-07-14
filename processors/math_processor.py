"""
Integrated Math Processor for MathRush DataProcessor.
Combines PDF conversion, GPT extraction, and database operations.
Processes problem-solution pairs with filename-based matching.

REQUIRED FILENAME FORMATS:
  Standard format:
    YYYY-MM-DD_ExamType_problems.pdf
    YYYY-MM-DD_ExamType_solutions.pdf
    
  Alternative format:
    YYYYMMDD_ExamType_problems.pdf
    YYYYMMDD_ExamType_solutions.pdf
    
  Simplified format:
    YYYY-MM-DD_problems.pdf
    YYYY-MM-DD_solutions.pdf

EXAMPLES:
  2024-06-06_suneung_problems.pdf + 2024-06-06_suneung_solutions.pdf
  2024-03-15_mock_problems.pdf + 2024-03-15_mock_solutions.pdf
  20240606_school_problems.pdf + 20240606_school_solutions.pdf
  2023-11-16_problems.pdf + 2023-11-16_solutions.pdf

SUPPORTED EXAM TYPES:
  suneung, mock, school, monthly, final, midterm
  (Korean: 수능, 모의고사, 학교시험, 월례고사, 기말고사, 중간고사)
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import signal
from pathlib import Path

# Import all necessary modules
try:
    from config.settings import settings
    from utils.filename_parser import FilenameParser
    from processors.pdf_converter import PDFConverter
    from processors.gpt_extractor import GPTExtractor
    from processors.db_saver import DatabaseSaver
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config.settings import settings
    from utils.filename_parser import FilenameParser
    from processors.pdf_converter import PDFConverter
    from processors.gpt_extractor import GPTExtractor
    from processors.db_saver import DatabaseSaver

logger = logging.getLogger(__name__)


class MathProcessor:
    """Integrated processor for math problems from PDF to database."""
    
    def __init__(self, save_images: bool = False, save_json: bool = False, max_concurrent: int = 3):
        """
        Initialize math processor with all components.
        
        Args:
            save_images: Whether to save converted images
            save_json: Whether to save extraction JSON files
            max_concurrent: Maximum concurrent GPT API calls
        """
        self.save_images = save_images
        self.save_json = save_json
        self.max_concurrent = max_concurrent
        self._interrupted = False
        
        # Initialize components
        self.filename_parser = FilenameParser()
        self.pdf_converter = PDFConverter()
        self.gpt_extractor = GPTExtractor()
        self.db_saver = DatabaseSaver()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Test database connection
        if not self.db_saver.test_connection():
            raise ConnectionError("Failed to connect to Supabase database")
        
        logger.info("Math Processor initialized successfully")
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully."""
        logger.info(f"Received signal {signum}. Saving progress and shutting down gracefully...")
        self._interrupted = True
    
    def _get_checkpoint_path(self, base_name: str) -> str:
        """Get checkpoint file path for a given base name."""
        checkpoint_dir = os.path.join(settings.OUTPUT_DIR, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        return os.path.join(checkpoint_dir, f"{base_name}_checkpoint.json")
    
    def _save_checkpoint(self, base_name: str, checkpoint_data: Dict[str, Any]):
        """Save checkpoint data to file."""
        try:
            checkpoint_path = self._get_checkpoint_path(base_name)
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, base_name: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data from file."""
        try:
            checkpoint_path = self._get_checkpoint_path(base_name)
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                logger.info(f"Checkpoint loaded: {checkpoint_path}")
                return checkpoint_data
            return None
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _remove_checkpoint(self, base_name: str):
        """Remove checkpoint file after successful completion."""
        try:
            checkpoint_path = self._get_checkpoint_path(base_name)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                logger.debug(f"Checkpoint removed: {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint: {e}")
    
    def process_pdf_pair(self, problems_pdf: str, solutions_pdf: str, resume: bool = False) -> Dict[str, Any]:
        """
        Process a single PDF pair from start to database.
        
        Args:
            problems_pdf: Path to problems PDF
            solutions_pdf: Path to solutions PDF
            resume: Whether to resume from checkpoint
            
        Returns:
            Processing results summary
        """
        start_time = time.time()
        
        # Parse filenames and validate
        prob_info = self.filename_parser.parse_filename(problems_pdf)
        sol_info = self.filename_parser.parse_filename(solutions_pdf)
        
        if not prob_info['is_valid'] or not sol_info['is_valid']:
            raise ValueError(f"Invalid filename format. Problems: {prob_info.get('error')}, Solutions: {sol_info.get('error')}")
        
        # Validate that both files are from same exam
        if (prob_info['exam_date'] != sol_info['exam_date'] or 
            prob_info['exam_type'] != sol_info['exam_type']):
            raise ValueError(f"PDF pair mismatch: {prob_info['exam_date']} vs {sol_info['exam_date']}")
        
        # Create exam metadata
        exam_metadata = {
            'exam_date': prob_info['exam_date'],
            'exam_year': prob_info['exam_year'],
            'exam_month': prob_info['exam_month'],
            'exam_day': prob_info['exam_day'],
            'exam_type': prob_info['exam_type'],
            'problems_file': os.path.basename(problems_pdf),
            'solutions_file': os.path.basename(solutions_pdf),
            'base_name': prob_info['base_name']
        }
        
        logger.info(f"Processing exam pair: {exam_metadata['exam_date']} ({exam_metadata['exam_type']})")
        
        try:
            # Step 1: Convert PDFs to images
            logger.info("Step 1: Converting PDFs to images")
            
            # Create temporary directories for images
            temp_base = os.path.join(settings.TEMP_DIR, f"processing_{exam_metadata['base_name']}_{int(time.time())}")
            problems_image_dir = os.path.join(temp_base, "problems")
            solutions_image_dir = os.path.join(temp_base, "solutions")
            
            os.makedirs(problems_image_dir, exist_ok=True)
            os.makedirs(solutions_image_dir, exist_ok=True)
            
            # Convert problems PDF
            problems_images = self.pdf_converter.convert_pdf_to_images(
                problems_pdf, 
                output_dir=problems_image_dir
            )
            
            # Convert solutions PDF
            solutions_images = self.pdf_converter.convert_pdf_to_images(
                solutions_pdf,
                output_dir=solutions_image_dir
            )
            
            logger.info(f"Converted {len(problems_images)} problem pages, {len(solutions_images)} solution pages")
            
            # Step 2: Extract problems and solutions using GPT
            logger.info("Step 2: Extracting problems and solutions with GPT")
            
            # Extract problems
            problems_results = self.gpt_extractor.extract_from_image_list(
                problems_images, 
                "problems",
                delay=settings.RETRY_DELAY,
                max_concurrent=getattr(self, 'max_concurrent', 3)
            )
            
            # Extract solutions
            solutions_results = self.gpt_extractor.extract_from_image_list(
                solutions_images,
                "solutions", 
                delay=settings.RETRY_DELAY,
                max_concurrent=getattr(self, 'max_concurrent', 3)
            )
            
            # Step 3: Match problems with solutions (scoped to this exam)
            logger.info("Step 3: Matching problems with solutions")
            
            matched_problems, matching_report = self.gpt_extractor.match_problems_and_solutions_scoped(
                problems_results,
                solutions_results,
                exam_metadata
            )
            
            # Step 4: Save to database
            logger.info("Step 4: Saving to database")
            
            if matched_problems:
                db_results = self.db_saver.bulk_insert_problems(matched_problems)
                logger.info(f"Database insert: {db_results['successful']}/{db_results['total']} successful")
            else:
                db_results = {'total': 0, 'successful': 0, 'failed': 0, 'errors': []}
                logger.warning("No problems to insert into database")
            
            # Step 5: Optional side outputs
            if self.save_json:
                self._save_json_output(matched_problems, matching_report, exam_metadata, temp_base)
            
            if self.save_images:
                self._save_image_output(problems_images, solutions_images, exam_metadata)
            
            # Step 6: Cleanup temporary files
            if not self.save_images:
                self._cleanup_temp_files(temp_base)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create summary
            summary = {
                'exam_metadata': exam_metadata,
                'processing_time': round(processing_time, 2),
                'images_converted': {
                    'problems': len(problems_images),
                    'solutions': len(solutions_images)
                },
                'extraction_results': {
                    'total_problems': matching_report['total_problems'],
                    'total_solutions': matching_report['total_solutions'],
                    'matched_problems': matching_report['total_matched'],
                    'success_rate': matching_report['success_rate']
                },
                'database_results': db_results,
                'status': 'success'
            }
            
            logger.info(f"Successfully processed {exam_metadata['exam_date']}: {matching_report['total_matched']} problems in {processing_time:.1f}s")
            return summary
            
        except Exception as e:
            logger.error(f"Error processing PDF pair: {e}")
            
            # Cleanup on error
            if 'temp_base' in locals():
                self._cleanup_temp_files(temp_base)
            
            return {
                'exam_metadata': exam_metadata,
                'processing_time': time.time() - start_time,
                'status': 'error',
                'error': str(e)
            }
    
    def process_directory(self, directory: str, max_pairs: Optional[int] = None) -> Dict[str, Any]:
        """
        Process all PDF pairs in a directory.
        
        Args:
            directory: Directory containing PDF files
            max_pairs: Maximum number of pairs to process (None for all)
            
        Returns:
            Overall processing results
        """
        start_time = time.time()
        
        # Find PDF pairs
        pdf_pairs = self.filename_parser.find_pdf_pairs(directory)
        
        if not pdf_pairs:
            logger.warning(f"No valid PDF pairs found in {directory}")
            return {
                'directory': directory,
                'total_pairs': 0,
                'processed_pairs': 0,
                'successful_pairs': 0,
                'failed_pairs': 0,
                'results': [],
                'status': 'no_pairs_found'
            }
        
        # Limit pairs if specified
        if max_pairs:
            pdf_pairs = pdf_pairs[:max_pairs]
        
        logger.info(f"Processing {len(pdf_pairs)} PDF pairs from {directory}")
        
        results = []
        successful_pairs = 0
        failed_pairs = 0
        
        for i, (problems_pdf, solutions_pdf, metadata) in enumerate(pdf_pairs, 1):
            logger.info(f"Processing pair {i}/{len(pdf_pairs)}: {metadata['exam_date']}")
            
            try:
                result = self.process_pdf_pair(problems_pdf, solutions_pdf)
                results.append(result)
                
                if result['status'] == 'success':
                    successful_pairs += 1
                else:
                    failed_pairs += 1
                    
            except Exception as e:
                logger.error(f"Failed to process pair {i}: {e}")
                failed_pairs += 1
                results.append({
                    'exam_metadata': metadata,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Calculate overall statistics
        total_problems = sum(r.get('extraction_results', {}).get('matched_problems', 0) for r in results if r['status'] == 'success')
        total_db_inserts = sum(r.get('database_results', {}).get('successful', 0) for r in results if r['status'] == 'success')
        processing_time = time.time() - start_time
        
        summary = {
            'directory': directory,
            'processing_time': round(processing_time, 2),
            'total_pairs': len(pdf_pairs),
            'processed_pairs': len(results),
            'successful_pairs': successful_pairs,
            'failed_pairs': failed_pairs,
            'total_problems_extracted': total_problems,
            'total_problems_inserted': total_db_inserts,
            'success_rate': successful_pairs / len(pdf_pairs) if pdf_pairs else 0,
            'results': results,
            'status': 'completed'
        }
        
        logger.info(f"Directory Processing Complete")
        logger.info(f"Processed: {successful_pairs}/{len(pdf_pairs)} pairs successfully")
        logger.info(f"Total problems: {total_problems} extracted, {total_db_inserts} inserted")
        logger.info(f"Processing time: {processing_time:.1f} seconds")
        
        return summary
    
    def _save_json_output(self, matched_problems: List[Dict[str, Any]], matching_report: Dict[str, Any], exam_metadata: Dict[str, Any], temp_base: str):
        """Save JSON output for debugging/backup."""
        try:
            json_output = {
                'exam_metadata': exam_metadata,
                'matched_problems': matched_problems,
                'matching_report': matching_report,
                'exported_at': datetime.now().isoformat()
            }
            
            json_file = os.path.join(settings.OUTPUT_DIR, f"{exam_metadata['base_name']}_extracted.json")
            os.makedirs(os.path.dirname(json_file), exist_ok=True)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, ensure_ascii=False, indent=2)
            
            logger.info(f"JSON backup saved: {json_file}")
            
        except Exception as e:
            logger.error(f"Failed to save JSON output: {e}")
    
    def _save_image_output(self, problems_images: List[str], solutions_images: List[str], exam_metadata: Dict[str, Any]):
        """Save converted images to permanent location."""
        try:
            # Create permanent image directories
            image_base = os.path.join(settings.OUTPUT_DIR, "images", exam_metadata['base_name'])
            prob_dir = os.path.join(image_base, "problems")
            sol_dir = os.path.join(image_base, "solutions")
            
            os.makedirs(prob_dir, exist_ok=True)
            os.makedirs(sol_dir, exist_ok=True)
            
            # Copy images to permanent location
            import shutil
            
            for img_path in problems_images:
                filename = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(prob_dir, filename))
            
            for img_path in solutions_images:
                filename = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(sol_dir, filename))
            
            logger.info(f"Images saved to: {image_base}")
            
        except Exception as e:
            logger.error(f"Failed to save images: {e}")
    
    def _cleanup_temp_files(self, temp_base: str):
        """Clean up temporary files."""
        try:
            import shutil
            if os.path.exists(temp_base):
                shutil.rmtree(temp_base)
                logger.debug(f"Cleaned up temporary files: {temp_base}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary files {temp_base}: {e}")


def main():
    """CLI interface for integrated math processing."""
    parser = argparse.ArgumentParser(
        description="Integrated Math Processor: PDF → Images → GPT → Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
REQUIRED FILENAME FORMATS:
  Standard: YYYY-MM-DD_ExamType_problems.pdf, YYYY-MM-DD_ExamType_solutions.pdf
  Alternative: YYYYMMDD_ExamType_problems.pdf, YYYYMMDD_ExamType_solutions.pdf
  Simplified: YYYY-MM-DD_problems.pdf, YYYY-MM-DD_solutions.pdf

SUPPORTED EXAM TYPES:
  suneung, mock, school, monthly, final, midterm
  (Korean: 수능, 모의고사, 학교시험, 월례고사, 기말고사, 중간고사)

EXAMPLES:
  # Process single PDF pair
  python math_processor.py --problems 2024-06-06_suneung_problems.pdf --solutions 2024-06-06_suneung_solutions.pdf
  
  # Process with 5 concurrent GPT calls (faster)
  python math_processor.py --problems file1.pdf --solutions file2.pdf --concurrent 5
  
  # Resume interrupted processing
  python math_processor.py --problems file1.pdf --solutions file2.pdf --resume
  
  # Auto-process all pairs in directory
  python math_processor.py --directory /path/to/pdfs/ --concurrent 3
  
  # Process with debug outputs and custom timeout
  python math_processor.py --directory pdfs/ --save-images --save-json --timeout 3600
  
  # Process limited number of pairs
  python math_processor.py --directory pdfs/ --max-pairs 5

FILENAME EXAMPLES:
  ✓ 2024-06-06_suneung_problems.pdf + 2024-06-06_suneung_solutions.pdf
  ✓ 2024-03-15_mock_problems.pdf + 2024-03-15_mock_solutions.pdf
  ✓ 20240606_school_problems.pdf + 20240606_school_solutions.pdf
  ✓ 2023-11-16_problems.pdf + 2023-11-16_solutions.pdf
  ✗ random_filename.pdf (invalid format)
"""
    )
    
    # Input options
    parser.add_argument(
        "--problems",
        help="Problems PDF file"
    )
    
    parser.add_argument(
        "--solutions",
        help="Solutions PDF file"
    )
    
    parser.add_argument(
        "--directory",
        help="Directory containing PDF pairs"
    )
    
    # Processing options
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Maximum number of PDF pairs to process"
    )
    
    # Output options
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save converted images to output directory"
    )
    
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save extraction results as JSON files"
    )
    
    # Other options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--test-db",
        action="store_true",
        help="Test database connection and exit"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,  # 30 minutes default
        help="Processing timeout in seconds (default: 1800)"
    )
    
    parser.add_argument(
        "--concurrent",
        type=int,
        default=3,
        help="Number of concurrent GPT API calls (default: 3)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing from last checkpoint"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Test database connection if requested
        if args.test_db:
            print("Testing database connection...")
            db_saver = DatabaseSaver()
            if db_saver.test_connection():
                print("✅ Database connection successful")
                return 0
            else:
                print("❌ Database connection failed")
                return 1
        
        # Validate input
        if args.directory:
            if not os.path.isdir(args.directory):
                print(f"Error: Directory not found: {args.directory}")
                return 1
        elif args.problems and args.solutions:
            if not os.path.isfile(args.problems) or not os.path.isfile(args.solutions):
                print("Error: One or both PDF files not found")
                return 1
        else:
            parser.error("Must specify either --directory or both --problems and --solutions")
        
        # Initialize processor
        processor = MathProcessor(
            save_images=args.save_images,
            save_json=args.save_json,
            max_concurrent=args.concurrent
        )
        
        # Process input
        if args.directory:
            # Process directory
            results = processor.process_directory(args.directory, args.max_pairs)
            
            # Print summary
            print(f"\n=== Processing Complete ===")
            print(f"Directory: {results['directory']}")
            print(f"Processed: {results['successful_pairs']}/{results['total_pairs']} pairs successfully")
            print(f"Total problems: {results['total_problems_extracted']} extracted, {results['total_problems_inserted']} inserted")
            print(f"Processing time: {results['processing_time']:.1f} seconds")
            
            if results['failed_pairs'] > 0:
                print(f"\n⚠️  {results['failed_pairs']} pairs failed to process")
                return 1
            
        else:
            # Process single pair
            results = processor.process_pdf_pair(args.problems, args.solutions, resume=args.resume)
            
            # Print summary
            print(f"\n=== Processing Complete ===")
            print(f"Exam: {results['exam_metadata']['exam_date']} ({results['exam_metadata']['exam_type']})")
            print(f"Problems: {results['extraction_results']['matched_problems']}/{results['extraction_results']['total_problems']}")
            print(f"Database: {results['database_results']['successful']}/{results['database_results']['total']} inserted")
            print(f"Processing time: {results['processing_time']:.1f} seconds")
            
            if results['status'] != 'success':
                print(f"\n❌ Processing failed: {results.get('error', 'Unknown error')}")
                return 1
        
        print("\n🎉 All processing completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Unexpected error in main")
        return 1


if __name__ == "__main__":
    exit(main())