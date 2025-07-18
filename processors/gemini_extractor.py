"""
Gemini extractor for MathRush DataProcessor.
Extracts problem content and metadata from individual problem images using Gemini 1.5 Pro.
"""

import os
import json
import base64
import logging
import time
import random
from typing import Dict, Any, Optional, List
from PIL import Image
import google.generativeai as genai

# Import settings
try:
    from config.settings import settings
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config.settings import settings

logger = logging.getLogger(__name__)


class GeminiExtractor:
    """Gemini extractor for individual problem images using Gemini 1.5 Pro.
    
    Note: This extractor focuses only on content, problem type, and choices.
    It does NOT extract metadata fields like:
    - correct_rate: Used for future difficulty determination (manually input)
    - difficulty: Determined separately based on correct_rate data
    - explanation: Not extracted to keep costs low for MVP
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini extractor.
        
        Args:
            api_key: Google API key (uses settings if None)
        """
        self.api_key = api_key or settings.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY in .env file.")
        
        # Initialize Google client
        genai.configure(api_key=self.api_key)
        
        # Configuration from settings
        self.model = settings.GEMINI_MODEL
        self.max_tokens = settings.GEMINI_MAX_TOKENS
        self.temperature = settings.GEMINI_TEMPERATURE
        
        # Rate limiting configuration
        self.requests_per_minute = settings.GEMINI_REQUESTS_PER_MINUTE
        self.retry_attempts = settings.GEMINI_RETRY_ATTEMPTS
        self.retry_base_delay = settings.GEMINI_RETRY_BASE_DELAY
        self.retry_max_delay = settings.GEMINI_RETRY_MAX_DELAY
        
        # Rate limiting state
        self.last_request_time = 0.0
        self.min_request_interval = 60.0 / self.requests_per_minute  # seconds between requests
        
        logger.info(f"Gemini Extractor initialized with model: {self.model}")
        logger.info(f"Rate limiting: {self.requests_per_minute} requests/minute, {self.retry_attempts} retry attempts")
    
    def get_image_mime_type(self, image_path: str) -> str:
        """
        Get the MIME type for an image file based on its extension.
        
        Args:
            image_path: Path to image file
            
        Returns:
            MIME type string (e.g., 'image/png', 'image/jpeg')
        """
        extension = os.path.splitext(image_path)[1].lower()
        
        mime_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg'
        }
        
        return mime_types.get(extension, 'image/png')  # Default to PNG if unknown
    
    def _wait_for_rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.debug(f"Rate limiting: waiting {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _is_rate_limit_error(self, exception: Exception) -> bool:
        """
        Check if an exception is a rate limit error.
        
        Args:
            exception: Exception to check
            
        Returns:
            True if this is a rate limit error
        """
        error_str = str(exception)
        
        # Check for rate limit indicators
        rate_limit_indicators = [
            'rate limit',
            'rate_limit_exceeded',
            'Too Many Requests',
            '429',
            'Resource has been exhausted'
        ]
        
        return any(indicator.lower() in error_str.lower() for indicator in rate_limit_indicators)
    
    def _make_api_call_with_retry(self, contents: List[Any]) -> str:
        """
        Make Gemini API call with exponential backoff retry logic.
        
        Args:
            contents: Contents to send to the API
            
        Returns:
            Response content from the API
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                # Enforce rate limiting
                self._wait_for_rate_limit()
                
                logger.debug(f"Making API call (attempt {attempt + 1}/{self.retry_attempts})")
                
                # Make the API call
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(
                    contents,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=self.max_tokens,
                        temperature=self.temperature
                    )
                )
                
                if attempt > 0:
                    logger.info(f"API call succeeded after {attempt + 1} attempts")
                
                return response.text
                
            except Exception as e:
                last_exception = e
                error_str = str(e)
                
                # Check if this is a rate limit error
                if self._is_rate_limit_error(e):
                    if attempt < self.retry_attempts - 1:
                        # Calculate exponential backoff delay
                        base_delay = self.retry_base_delay * (2 ** attempt)
                        jitter = random.uniform(0.1, 0.5)  # Add jitter to avoid thundering herd
                        delay = min(base_delay + jitter, self.retry_max_delay)
                        
                        logger.warning(f"Rate limit hit (attempt {attempt + 1}/{self.retry_attempts}). Using exponential backoff: {delay:.2f}s...")
                        
                        time.sleep(delay)
                    else:
                        logger.error(f"Rate limit exceeded after {self.retry_attempts} attempts: {error_str}")
                        
                else:
                    # Non-rate-limit error
                    if attempt < self.retry_attempts - 1:
                        delay = self.retry_base_delay * (2 ** attempt)
                        logger.warning(f"API call failed (attempt {attempt + 1}/{self.retry_attempts}): {error_str[:100]}... Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"API call failed after {self.retry_attempts} attempts: {error_str}")
        
        # If we get here, all attempts failed
        raise last_exception or Exception("All retry attempts failed")
    
    def get_extraction_prompt(self, problem_number: Optional[int] = None) -> str:
        """
        Get the simplified prompt for content extraction only.
        
        Args:
            problem_number: Optional problem number to include in instructions
        
        Returns:
            Lightweight prompt focusing on content, type, and choices only
        """
        # Create dynamic prompt based on problem number
        problem_number_instruction = ""
        if problem_number is not None:
            problem_number_instruction = f"""
**중요: 문제 번호 규칙:**
- 문제 내용은 반드시 "{problem_number}. "으로 시작해야 합니다
- 예시: "{problem_number}. 다음 함수의 최댓값을 구하시오..."
"""
        
        return f"""당신은 한국 고등학교 수학 문제를 분석하는 전문가입니다.

주어진 이미지의 수학 문제에서 다음 정보만 추출해주세요:
{problem_number_instruction}
**추출할 정보:**
1. **문제 내용**: 문제의 전체 텍스트 (수식 포함)
2. **문제 유형**: 객관식인지 주관식인지
3. **선택지**: 객관식인 경우만 선택지들

**추출하지 말 것:**
- ❌ 정답 (correct_answer)
- ❌ 해설 (explanation)
- ❌ 복잡한 메타데이터

**문제 유형 분류:**
- "multiple_choice": 선택지가 있는 객관식 문제
- "subjective": 선택지가 없는 주관식 문제

**박스 내용 처리 규칙:**
- 이미지에서 박스로 둘러싸인 부분을 찾아서 적절히 표시
- 박스에 헤더가 있는 경우 (예: <보기>): 헤더를 **굵게** 표시하고 줄바꿈 후 내용을 ```로 감싸기
- 박스에 헤더가 없는 경우: 내용만 ```로 감싸서 박스임을 표시
- 박스 내부의 조건 표시 (ㄱ, ㄴ, ㄷ 또는 (가), (나), (다))는 원본 그대로 유지

**박스 형식화 예시:**
헤더가 있는 경우: 
**<보기>**
```
ㄱ. 조건 내용
ㄴ. 조건 내용
ㄷ. 조건 내용
```

헤더가 없는 경우:
```
(가) 조건 내용
(나) 조건 내용
```

**응답 형식** (JSON):
```json
{{
  "content": "문제 본문 전체 (문제 번호로 시작, 수식과 모든 텍스트 포함, 박스 형식화 적용)",
  "problem_type": "multiple_choice" 또는 "subjective",
  "choices": {{
    "1": "선택지1",
    "2": "선택지2",
    "3": "선택지3",
    "4": "선택지4",
    "5": "선택지5"
  }}
}}
```

**중요 규칙:**
- 객관식인 경우만 `choices` 필드 포함
- 주관식인 경우 `choices`는 null로 설정
- 문제 본문은 완전하고 정확하게 추출
- 수식이나 특수 기호도 모두 포함
- 박스 형식화 규칙을 반드시 적용

이미지를 분석하여 위 형식으로 응답해주세요."""
    
    def extract_from_image(self, image_path: str, math_content_images: Optional[List[str]] = None, problem_number: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract problem content from a single image.
        
        Args:
            image_path: Path to problem image
            math_content_images: Optional list of additional math content images
            problem_number: Optional problem number to include at start of content
            
        Returns:
            Dictionary with extracted problem data
        """
        try:
            logger.info(f"Extracting content from: {image_path}")
            
            # Prepare content for API call
            prompt = self.get_extraction_prompt(problem_number)
            
            main_image_mime = self.get_image_mime_type(image_path)
            with open(image_path, "rb") as image_file:
                main_image_data = image_file.read()

            content_list = [prompt, Image.open(image_path)]

            # Add math content images if provided
            if math_content_images:
                content_list.append(f"추가로 다음 {len(math_content_images)}개의 수학적 내용 이미지들이 이 문제에 포함됩니다:")
                
                for i, content_image_path in enumerate(math_content_images, 1):
                    try:
                        content_list.append(f"수학적 내용 {i}:")
                        content_list.append(Image.open(content_image_path))
                    except Exception as e:
                        logger.warning(f"Failed to encode math content image {content_image_path}: {e}")
            
            # Make API call with retry logic
            content = self._make_api_call_with_retry(content_list)
            problem_data = self._parse_response(content)
            
            # Validate extracted data
            if self._validate_extracted_data(problem_data):
                logger.info(f"Successfully extracted {problem_data['problem_type']} problem")
                return problem_data
            else:
                logger.error("Extracted data validation failed")
                return self._create_fallback_data(image_path)
                
        except Exception as e:
            logger.error(f"Error extracting from {image_path}: {e}")
            return self._create_fallback_data(image_path)
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """
        Parse Gemini response to extract JSON data.
        
        Args:
            content: Raw response content
            
        Returns:
            Parsed problem data dictionary
        """
        try:
            # Handle markdown code blocks
            json_content = content
            if content.strip().startswith('```json'):
                start = content.find('```json') + 7
                end = content.rfind('```')
                if end > start:
                    json_content = content[start:end].strip()
            elif content.strip().startswith('```'):
                start = content.find('```') + 3
                end = content.rfind('```')
                if end > start:
                    json_content = content[start:end].strip()
            
            # Parse JSON
            data = json.loads(json_content)
            
            # Ensure required fields exist
            required_fields = ['content', 'problem_type']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw content: {content}")
            raise
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            raise
    
    def _validate_extracted_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate extracted problem data.
        
        Args:
            data: Extracted problem data
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check required fields
            if not data.get('content') or not data.get('problem_type'):
                logger.error("Missing required fields: content or problem_type")
                return False
            
            # Validate problem type
            valid_types = ['multiple_choice', 'subjective']
            if data['problem_type'] not in valid_types:
                logger.error(f"Invalid problem_type: {data['problem_type']}")
                return False
            
            # Validate choices for multiple choice problems
            if data['problem_type'] == 'multiple_choice':
                choices = data.get('choices')
                if not choices or not isinstance(choices, dict):
                    logger.error("Multiple choice problem missing valid choices")
                    return False
                
                # Check if choices have reasonable keys
                expected_keys = ['1', '2', '3', '4', '5']
                if not any(key in choices for key in expected_keys):
                    logger.error("Multiple choice problem missing standard choice keys")
                    return False
            
            # Content length check
            content_length = len(data['content'])
            if content_length < 10:
                logger.error(f"Content too short: {content_length} characters")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False
    
    def _create_fallback_data(self, image_path: str) -> Dict[str, Any]:
        """
        Create fallback data when extraction fails.
        
        Args:
            image_path: Path to the image that failed
            
        Returns:
            Fallback problem data
        """
        filename = os.path.basename(image_path)
        return {
            'content': f'[Extraction Failed] Problem from {filename}',
            'problem_type': 'subjective',
            'choices': None,
            'extraction_error': True
        }
    
    def batch_extract(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Extract content from multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of extracted problem data
        """
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                result = self.extract_from_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append(self._create_fallback_data(image_path))
        
        logger.info(f"Batch extraction completed: {len(results)} images processed")
        return results


def main():
    """Test function for lightweight extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Lightweight Gemini Extractor")
    parser.add_argument("image_path", help="Path to problem image")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')
    
    try:
        extractor = GeminiExtractor()
        result = extractor.extract_from_image(args.image_path)
        
        print("Extraction Result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
