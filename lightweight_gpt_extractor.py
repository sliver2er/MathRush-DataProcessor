"""
Lightweight GPT extractor for MathRush DataProcessor.
Simplified version focusing only on content extraction without explanations or answers.
"""

import os
import json
import base64
import logging
from typing import Dict, Any, Optional, List
from PIL import Image
import openai

# Import settings
try:
    from config.settings import settings
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config.settings import settings

logger = logging.getLogger(__name__)


class LightweightGPTExtractor:
    """Lightweight GPT extractor for individual problem images."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize lightweight GPT extractor.
        
        Args:
            api_key: OpenAI API key (uses settings if None)
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in .env file.")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Simple configuration
        self.model = "gpt-4o-mini"  # Lightweight model for cost efficiency
        self.max_tokens = 2000     # Reduced tokens since no explanation needed
        self.temperature = 0.1     # Low temperature for consistent results
        
        logger.info(f"Lightweight GPT Extractor initialized with model: {self.model}")
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 for API.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, 'rb') as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise
    
    def get_extraction_prompt(self) -> str:
        """
        Get the simplified prompt for content extraction only.
        
        Returns:
            Lightweight prompt focusing on content, type, and choices only
        """
        return """당신은 한국 고등학교 수학 문제를 분석하는 전문가입니다.

주어진 이미지의 수학 문제에서 다음 정보만 추출해주세요:

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

**응답 형식** (JSON):
```json
{
  "content": "문제 본문 전체 (수식과 모든 텍스트 포함)",
  "problem_type": "multiple_choice" 또는 "subjective",
  "choices": {
    "1": "선택지1",
    "2": "선택지2",
    "3": "선택지3",
    "4": "선택지4",
    "5": "선택지5"
  }
}
```

**중요 규칙:**
- 객관식인 경우만 `choices` 필드 포함
- 주관식인 경우 `choices`는 null로 설정
- 문제 본문은 완전하고 정확하게 추출
- 수식이나 특수 기호도 모두 포함

이미지를 분석하여 위 형식으로 응답해주세요."""
    
    def extract_from_image(self, image_path: str, math_content_images: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract problem content from a single image.
        
        Args:
            image_path: Path to problem image
            math_content_images: Optional list of additional math content images
            
        Returns:
            Dictionary with extracted problem data
        """
        try:
            logger.info(f"Extracting content from: {image_path}")
            
            # Encode main problem image
            base64_image = self.encode_image(image_path)
            
            # Prepare content for API call
            content_list = [
                {
                    "type": "text",
                    "text": self.get_extraction_prompt()
                },
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
            
            # Add math content images if provided
            if math_content_images:
                content_list.append({
                    "type": "text",
                    "text": f"추가로 다음 {len(math_content_images)}개의 수학적 내용 이미지들이 이 문제에 포함됩니다:"
                })
                
                for i, content_image_path in enumerate(math_content_images, 1):
                    try:
                        content_base64 = self.encode_image(content_image_path)
                        content_list.extend([
                            {
                                "type": "text",
                                "text": f"수학적 내용 {i}:"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{content_base64}"
                                }
                            }
                        ])
                    except Exception as e:
                        logger.warning(f"Failed to encode math content image {content_image_path}: {e}")
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": content_list
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract and parse response
            content = response.choices[0].message.content
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
        Parse GPT response to extract JSON data.
        
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
    
    parser = argparse.ArgumentParser(description="Test Lightweight GPT Extractor")
    parser.add_argument("image_path", help="Path to problem image")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s - %(message)s')
    
    try:
        extractor = LightweightGPTExtractor()
        result = extractor.extract_from_image(args.image_path)
        
        print("Extraction Result:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())