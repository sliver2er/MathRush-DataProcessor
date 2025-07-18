"""
Database operations module for MathRush DataProcessor.
Handles Supabase database operations for math problems.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from supabase import create_client, Client
import time

# Import settings
try:
    from config.settings import settings
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config.settings import settings

logger = logging.getLogger(__name__)


class DatabaseSaver:
    """Handle Supabase database operations for math problems."""
    
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Initialize database connection.
        
        Args:
            url: Supabase URL (uses settings if None)
            key: Supabase key (uses settings if None)
        """
        self.url = url or settings.SUPABASE_URL
        self.key = key or settings.SUPABASE_KEY
        self.table_name = settings.SUPABASE_TABLE
        
        if not self.url or not self.key:
            raise ValueError("Supabase URL and key must be provided in .env file")
        
        try:
            self.client: Client = create_client(self.url, self.key)
            logger.info(f"Connected to Supabase database: {self.table_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to query the table (limit 1 to minimize data transfer)
            result = self.client.table(self.table_name).select("id").limit(1).execute()
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def validate_problem_data(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean problem data before insertion.
        
        Args:
            problem: Problem data dictionary
            
        Returns:
            Validated and cleaned problem data
        """
        validated = {}
        
        # Extract exam_name and problem_number from source_info if not provided directly
        if 'exam_name' not in problem and 'source_info' in problem:
            source_info = problem.get('source_info', {})
            if 'exam_name' in source_info:
                problem['exam_name'] = source_info['exam_name']
        
        if 'problem_number' not in problem and 'source_info' in problem:
            source_info = problem.get('source_info', {})
            if 'problem_number' in source_info:
                problem['problem_number'] = source_info['problem_number']
        
        # Required fields
        required_fields = settings.REQUIRED_FIELDS
        for field in required_fields:
            if field not in problem:
                raise ValueError(f"Missing required field: {field}")
            validated[field] = problem[field]
        
        # Optional fields with defaults (only fields that exist in database)
        validated.update({
            'problem_type': problem.get('problem_type', 'subjective'),
            'choices': problem.get('choices', None),
            'explanation': problem.get('explanation', None),
            'level': problem.get('level', None),
            'subject': problem.get('subject', None),
            'chapter': problem.get('chapter', None),
            'difficulty': problem.get('difficulty', None),
            'correct_rate': problem.get('correct_rate', None),  # For future difficulty determination
            'source_info': problem.get('source_info', None),
            'tags': problem.get('tags', None),
            'images': problem.get('images', None),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        })
        
        # Validate enums
        valid_problem_types = ['multiple_choice', 'subjective']
        if validated['problem_type'] not in valid_problem_types:
            logger.warning(f"Invalid problem_type: {validated['problem_type']}, defaulting to 'subjective'")
            validated['problem_type'] = 'subjective'
        
        # Validate exam_name and problem_number
        if not isinstance(validated['exam_name'], str) or not validated['exam_name'].strip():
            raise ValueError("exam_name must be a non-empty string")
        
        if not isinstance(validated['problem_number'], int) and not str(validated['problem_number']).isdigit():
            raise ValueError("problem_number must be a positive integer")
        
        # Ensure problem_number is integer
        validated['problem_number'] = int(validated['problem_number'])
        
# Curriculum field removed from database schema
        
        valid_difficulties = ['easy', 'medium', 'hard']
        if validated['difficulty'] is not None and validated['difficulty'] not in valid_difficulties:
            logger.warning(f"Invalid difficulty: {validated['difficulty']}, defaulting to 'medium'")
            validated['difficulty'] = 'medium'
        
        # Validate correct_rate (for future difficulty determination)
        if validated['correct_rate'] is not None:
            try:
                correct_rate = float(validated['correct_rate'])
                if correct_rate < settings.MIN_CORRECT_RATE or correct_rate > settings.MAX_CORRECT_RATE:
                    logger.warning(f"Invalid correct_rate: {correct_rate}, must be between {settings.MIN_CORRECT_RATE} and {settings.MAX_CORRECT_RATE}")
                    validated['correct_rate'] = None
                else:
                    validated['correct_rate'] = correct_rate
            except (ValueError, TypeError):
                logger.warning(f"Invalid correct_rate format: {validated['correct_rate']}, setting to None")
                validated['correct_rate'] = None
        
        # Validate content length
        content_length = len(validated['content'])
        if content_length < settings.MIN_PROBLEM_LENGTH:
            raise ValueError(f"Content too short: {content_length} chars")
        if content_length > settings.MAX_PROBLEM_LENGTH:
            raise ValueError(f"Content too long: {content_length} chars")
        
        # Ensure JSON fields are properly formatted
        if validated['choices'] and not isinstance(validated['choices'], dict):
            logger.warning("Choices field is not a dictionary, converting to None")
            validated['choices'] = None
        
        if validated['source_info'] is not None and not isinstance(validated['source_info'], dict):
            validated['source_info'] = None
        
        if validated['tags'] is not None and not isinstance(validated['tags'], list):
            validated['tags'] = None
        
        # Validate images field
        if validated['images'] is not None and not isinstance(validated['images'], list):
            validated['images'] = None
        
        # Ensure all image paths are strings if images is not None
        if validated['images'] is not None:
            validated['images'] = [str(img) for img in validated['images'] if img]
        
        return validated
    
    def insert_problem(self, problem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Insert a single problem into the database.
        
        Args:
            problem: Problem data dictionary
            
        Returns:
            Inserted problem data with ID, or None if failed
        """
        try:
            # Validate data
            validated_problem = self.validate_problem_data(problem)
            
            # Insert into database
            result = self.client.table(self.table_name).insert(validated_problem).execute()
            
            if result.data:
                inserted_problem = result.data[0]
                logger.info(f"Successfully inserted problem with ID: {inserted_problem.get('id')}")
                return inserted_problem
            else:
                logger.error("Insert operation returned no data")
                return None
                
        except Exception as e:
            logger.error(f"Error inserting problem: {e}")
            return None
    
    def bulk_insert_problems(self, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Insert multiple problems in batch.
        
        Args:
            problems: List of problem data dictionaries
            
        Returns:
            Dictionary with insertion results
        """
        results = {
            'total': len(problems),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        if not problems:
            logger.warning("No problems to insert")
            return results
        
        logger.info(f"Starting bulk insert of {len(problems)} problems")
        
        # Validate all problems first
        validated_problems = []
        for i, problem in enumerate(problems):
            try:
                validated = self.validate_problem_data(problem)
                validated_problems.append(validated)
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'index': i,
                    'error': str(e),
                    'problem_content': problem.get('content', '')[:100] + '...'
                })
                logger.error(f"Validation failed for problem {i}: {e}")
        
        # Insert validated problems in batches
        batch_size = 100  # Supabase recommended batch size
        for i in range(0, len(validated_problems), batch_size):
            batch = validated_problems[i:i + batch_size]
            
            try:
                result = self.client.table(self.table_name).insert(batch).execute()
                
                if result.data:
                    batch_success = len(result.data)
                    results['successful'] += batch_success
                    logger.info(f"Successfully inserted batch {i//batch_size + 1}: {batch_success} problems")
                else:
                    results['failed'] += len(batch)
                    results['errors'].append({
                        'batch': i//batch_size + 1,
                        'error': "Insert operation returned no data"
                    })
                    logger.error(f"Batch {i//batch_size + 1} insert failed: no data returned")
                
            except Exception as e:
                results['failed'] += len(batch)
                results['errors'].append({
                    'batch': i//batch_size + 1,
                    'error': str(e)
                })
                logger.error(f"Batch {i//batch_size + 1} insert failed: {e}")
            
            # Rate limiting to avoid overwhelming the database
            time.sleep(0.1)
        
        logger.info(f"Bulk insert completed: {results['successful']}/{results['total']} successful")
        return results
    
    def bulk_upsert_problems(self, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Upsert multiple problems in batch.
        
        Args:
            problems: List of problem data dictionaries
            
        Returns:
            Dictionary with upsert results
        """
        results = {
            'total': len(problems),
            'successful': 0,
            'failed': 0,
            'errors': [],
            'inserted': 0,
            'updated': 0
        }
        
        if not problems:
            logger.warning("No problems to upsert")
            return results
        
        logger.info(f"Starting bulk upsert of {len(problems)} problems")
        
        for i, problem in enumerate(problems):
            try:
                # Validate first
                validated = self.validate_problem_data(problem)
                exam_name = validated['exam_name']
                problem_number = validated['problem_number']
                
                # Check if exists
                existing = self.find_existing_problem(exam_name, problem_number)
                
                if existing:
                    # Update existing
                    existing_id = existing['id']
                    update_data = {k: v for k, v in validated.items() if k != 'created_at'}
                    update_data['updated_at'] = datetime.now().isoformat()
                    
                    result = self.client.table(self.table_name).update(update_data).eq("id", existing_id).execute()
                    
                    if result.data:
                        results['successful'] += 1
                        results['updated'] += 1
                        logger.debug(f"Updated: {exam_name} #{problem_number}")
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Update failed for {exam_name} #{problem_number}")
                else:
                    # Insert new
                    result = self.client.table(self.table_name).insert(validated).execute()
                    
                    if result.data:
                        results['successful'] += 1
                        results['inserted'] += 1
                        logger.debug(f"Inserted: {exam_name} #{problem_number}")
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"Insert failed for {exam_name} #{problem_number}")
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Problem {i}: {str(e)}")
                logger.error(f"Error upserting problem {i}: {e}")
        
        logger.info(f"Bulk upsert completed: {results['successful']}/{results['total']} successful ({results['inserted']} inserted, {results['updated']} updated)")
        return results
    
    def check_duplicate(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Check if a problem with similar content already exists.
        
        Args:
            content: Problem content to check
            
        Returns:
            Existing problem data if duplicate found, None otherwise
        """
        try:
            # Search for problems with exact content match
            result = self.client.table(self.table_name).select("*").eq("content", content).execute()
            
            if result.data:
                logger.info(f"Found duplicate problem with content: {content[:50]}...")
                return result.data[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking duplicate: {e}")
            return None
    
    def find_existing_problem(self, exam_name: str, problem_number: int) -> Optional[Dict[str, Any]]:
        """
        Find existing problem by exam_name and problem_number.
        
        Args:
            exam_name: Name of the exam
            problem_number: Problem number
            
        Returns:
            Existing problem data if found, None otherwise
        """
        try:
            result = self.client.table(self.table_name)\
                .select("*")\
                .eq("exam_name", exam_name)\
                .eq("problem_number", problem_number)\
                .execute()
            
            if result.data:
                logger.info(f"Found existing problem: {exam_name} #{problem_number}")
                return result.data[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding existing problem: {e}")
            return None
    
    def upsert_problem(self, problem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Insert or update a problem based on exam_name and problem_number.
        
        Args:
            problem: Problem data dictionary
            
        Returns:
            Upserted problem data with ID, or None if failed
        """
        try:
            # Validate data first
            validated_problem = self.validate_problem_data(problem)
            
            exam_name = validated_problem['exam_name']
            problem_number = validated_problem['problem_number']
            
            # Check if record already exists
            existing_problem = self.find_existing_problem(exam_name, problem_number)
            
            if existing_problem:
                # Update existing record
                existing_id = existing_problem['id']
                update_data = {k: v for k, v in validated_problem.items() if k != 'created_at'}
                update_data['updated_at'] = datetime.now().isoformat()
                
                result = self.client.table(self.table_name).update(update_data).eq("id", existing_id).execute()
                
                if result.data:
                    updated_problem = result.data[0]
                    logger.info(f"Updated existing problem: {exam_name} #{problem_number} (ID: {existing_id})")
                    return updated_problem
                else:
                    logger.error(f"Update operation returned no data for: {exam_name} #{problem_number}")
                    return None
            else:
                # Insert new record
                result = self.client.table(self.table_name).insert(validated_problem).execute()
                
                if result.data:
                    inserted_problem = result.data[0]
                    logger.info(f"Inserted new problem: {exam_name} #{problem_number} (ID: {inserted_problem.get('id')})")
                    return inserted_problem
                else:
                    logger.error(f"Insert operation returned no data for: {exam_name} #{problem_number}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error upserting problem: {e}")
            return None
    
    def update_problem(self, problem_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an existing problem.
        
        Args:
            problem_id: ID of the problem to update
            updates: Dictionary of fields to update
            
        Returns:
            Updated problem data, or None if failed
        """
        try:
            # Add updated timestamp
            updates['updated_at'] = datetime.now().isoformat()
            
            result = self.client.table(self.table_name).update(updates).eq("id", problem_id).execute()
            
            if result.data:
                updated_problem = result.data[0]
                logger.info(f"Successfully updated problem with ID: {problem_id}")
                return updated_problem
            else:
                logger.error(f"Update operation returned no data for ID: {problem_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error updating problem {problem_id}: {e}")
            return None
    
    def delete_problem(self, problem_id: str) -> bool:
        """
        Delete a problem from the database.
        
        Args:
            problem_id: ID of the problem to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.client.table(self.table_name).delete().eq("id", problem_id).execute()
            
            if result.data:
                logger.info(f"Successfully deleted problem with ID: {problem_id}")
                return True
            else:
                logger.error(f"Delete operation returned no data for ID: {problem_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting problem {problem_id}: {e}")
            return False
    
    def get_problem_count(self) -> int:
        """
        Get total number of problems in the database.
        
        Returns:
            Total problem count
        """
        try:
            result = self.client.table(self.table_name).select("id", count="exact").execute()
            return result.count or 0
        except Exception as e:
            logger.error(f"Error getting problem count: {e}")
            return 0
    
    def get_problems_by_source(self, source_pdf: str) -> List[Dict[str, Any]]:
        """
        Get problems from a specific source PDF.
        
        Args:
            source_pdf: Name of the source PDF file
            
        Returns:
            List of problems from the source
        """
        try:
            result = self.client.table(self.table_name).select("*").contains("source_info", {"problem_pdf": source_pdf}).execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Error getting problems by source {source_pdf}: {e}")
            return []


def test_database_operations():
    """Test database connection and basic operations."""
    try:
        # Initialize database saver
        db_saver = DatabaseSaver()
        
        # Test connection
        print("Testing database connection...")
        if not db_saver.test_connection():
            print("❌ Database connection failed")
            return False
        
        print("✅ Database connection successful")
        
        # Test problem count
        count = db_saver.get_problem_count()
        print(f"📊 Current problem count: {count}")
        
        # Test sample problem insertion
        sample_problem = {
            "content": "테스트 문제입니다. 2 + 2 = ?",
            "problem_type": "multiple_choice",
            "choices": {
                "1": "3",
                "2": "4", 
                "3": "5",
                "4": "6"
            },
            "correct_answer": "2",
            "explanation": "2 + 2 = 4입니다.",
            "curriculum": "2015개정",
            "level": "중1",
            "subject": "수학",
            "chapter": "자연수의 덧셈",
            "difficulty": "easy",
            "tags": ["덧셈", "기초"],
            "source_info": {
                "problem_pdf": "test.pdf",
                "processed_date": datetime.now().isoformat()
            }
        }
        
        print("\nTesting problem insertion...")
        inserted = db_saver.insert_problem(sample_problem)
        
        if inserted:
            print(f"✅ Problem inserted successfully with ID: {inserted.get('id')}")
            
            # Test deletion
            print("Testing problem deletion...")
            if db_saver.delete_problem(inserted['id']):
                print("✅ Problem deleted successfully")
            else:
                print("❌ Problem deletion failed")
        else:
            print("❌ Problem insertion failed")
            return False
        
        print("\n🎉 All database tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_database_operations()