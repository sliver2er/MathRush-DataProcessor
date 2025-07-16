#!/usr/bin/env python3
"""
Automated processor to extract problem data from images and save to database.
"""

import json
import logging
from processors import DatabaseSaver, GPTExtractor
from utils.filename_parser import FilenameParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_image_to_db(image_path: str, manual_answer: str = None) -> dict:
    """
    Automated function to extract problem data from image and save to database.
    
    Args:
        image_path: Path to the problem image file
        manual_answer: Manual answer if available (optional)
        
    Returns:
        Dictionary with processing results
    """
    results = {
        "success": False,
        "image_path": image_path,
        "extracted_data": None,
        "inserted_id": None,
        "error": None
    }
    
    try:
        # 1. Extract data using GPT
        logging.info(f"Extracting data from: {image_path}")
        extractor = GPTExtractor()
        extracted_data = extractor.extract_from_image(image_path)
        
        if not extracted_data:
            results["error"] = "GPT extraction failed"
            return results
            
        results["extracted_data"] = extracted_data
        logging.info(f"GPT extraction successful: {extracted_data['problem_type']}")
        
        # 2. Parse filename for metadata
        filename = image_path.split('/')[-1]
        # Parse image filename: exam_name_problem_number.extension
        import re
        match = re.match(r'(.+)_problem_(\d+)\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
        if match:
            exam_name = match.group(1)
            problem_number = int(match.group(2))
            
            # Parse exam metadata
            parser = FilenameParser()
            parsed_info = parser.parse_exam_filename(exam_name)
            parsed_info['problem_number'] = problem_number
        else:
            # Fallback for unknown format
            parsed_info = {
                'exam_name': filename.split('.')[0],
                'exam_date': '2020-12-03',
                'problem_number': 1,
                'exam_type': '수능',
                'curriculum': '2015개정',
                'level': '고3',
                'subject': '수학'
            }
        
        # 3. Build complete problem data (only required and basic fields)
        problem_data = {
            "content": extracted_data["content"],
            "problem_type": extracted_data["problem_type"],
            "choices": extracted_data["choices"],
            "correct_answer": manual_answer or "미입력",
            "explanation": "미입력",
            "source_info": {
                "exam_type": parsed_info.get("exam_type", "수능"),
                "exam_name": parsed_info.get("exam_name", exam_name if 'exam_name' in locals() else "unknown"),
                "exam_date": parsed_info.get("exam_date", "2020-12-03"),
                "problem_number": parsed_info.get("problem_number", 1),
                "total_points": 4,
                "problem_pdf": f"{parsed_info.get('exam_name', exam_name if 'exam_name' in locals() else 'unknown')}_problems.pdf",
                "solution_pdf": f"{parsed_info.get('exam_name', exam_name if 'exam_name' in locals() else 'unknown')}_solutions.pdf"
            }
        }
        
        # 4. Save to database
        logging.info("Saving to database...")
        db_saver = DatabaseSaver()
        
        # Check for duplicates
        duplicate = db_saver.check_duplicate(problem_data["content"])
        if duplicate:
            logging.warning(f"Duplicate found with ID: {duplicate['id']}")
            results["error"] = f"Duplicate problem found (ID: {duplicate['id']})"
            return results
        
        # Insert problem
        inserted_problem = db_saver.insert_problem(problem_data)
        
        if inserted_problem:
            results["success"] = True
            results["inserted_id"] = inserted_problem.get('id')
            logging.info(f"Successfully saved problem with ID: {results['inserted_id']}")
        else:
            results["error"] = "Database insertion failed"
            
    except Exception as e:
        results["error"] = str(e)
        logging.error(f"Error processing {image_path}: {e}")
    
    return results

def process_directory(directory_path: str) -> dict:
    """
    Process all images in a directory.
    
    Args:
        directory_path: Path to directory containing problem images
        
    Returns:
        Dictionary with batch processing results
    """
    import os
    import glob
    
    results = {
        "total": 0,
        "successful": 0,
        "failed": 0,
        "results": [],
        "errors": []
    }
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))
    
    results["total"] = len(image_files)
    
    if not image_files:
        results["errors"].append("No image files found in directory")
        return results
    
    logging.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in image_files:
        logging.info(f"Processing: {image_path}")
        
        result = process_image_to_db(image_path)
        results["results"].append(result)
        
        if result["success"]:
            results["successful"] += 1
        else:
            results["failed"] += 1
            results["errors"].append(f"{image_path}: {result['error']}")
    
    logging.info(f"Batch processing completed: {results['successful']}/{results['total']} successful")
    return results

def verify_database():
    """Verify the database content after processing."""
    print("\n=== Database Status ===")
    
    db_saver = DatabaseSaver()
    count = db_saver.get_problem_count()
    print(f"📊 Total problems: {count}")
    
    # Get recent problems
    try:
        result = db_saver.client.table(db_saver.table_name).select("*").order("created_at", desc=True).limit(5).execute()
        if result.data:
            print("📋 Recent problems:")
            for i, problem in enumerate(result.data):
                print(f"   {i+1}. ID: {problem['id']}")
                print(f"      Type: {problem['problem_type']}")
                print(f"      Content: {problem['content'][:50]}...")
                print(f"      Created: {problem['created_at']}")
                print()
    except Exception as e:
        print(f"Error fetching recent problems: {e}")

if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) != 2:
        print("Usage: python automated_processor.py <image_path_or_directory>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isfile(path):
        # Process single image
        print(f"Processing single image: {path}")
        result = process_image_to_db(path)
        
        print(f"\n📊 Result:")
        print(f"   Success: {result['success']}")
        if result['success']:
            print(f"   Database ID: {result['inserted_id']}")
        else:
            print(f"   Error: {result['error']}")
            
    elif os.path.isdir(path):
        # Process directory
        print(f"Processing directory: {path}")
        results = process_directory(path)
        
        print(f"\n📊 Batch Results:")
        print(f"   Total: {results['total']}")
        print(f"   Successful: {results['successful']}")
        print(f"   Failed: {results['failed']}")
        
        if results['errors']:
            print("\n❌ Errors:")
            for error in results['errors']:
                print(f"   {error}")
    else:
        print(f"Error: {path} is not a valid file or directory")
        sys.exit(1)
    
    # Show database status
    verify_database()