"""
GPT-based math problem extraction module for MathRush DataProcessor.
Uses GPT-4o-mini to extract structured data from PDF page images.
Supports scoped problem number matching within PDF pairs.
"""

import os
import json
import base64
import argparse
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import openai
import logging
from tqdm import tqdm
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import signal

# Import settings and utilities
try:
    from config.settings import settings
    from utils.filename_parser import FilenameParser
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config.settings import settings
    from utils.filename_parser import FilenameParser

logger = logging.getLogger(__name__)


class GPTExtractor:
    """Extract math problems from images using GPT-4o-mini with scoped matching."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GPT extractor.
        
        Args:
            api_key: OpenAI API key (uses settings if None)
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in .env file.")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Configuration
        self.model = settings.OPENAI_MODEL
        self.max_tokens = settings.OPENAI_MAX_TOKENS
        self.temperature = settings.OPENAI_TEMPERATURE
        
        # Initialize filename parser
        self.filename_parser = FilenameParser()
        
        logger.info(f"GPT Extractor initialized with model: {self.model}")
    
    def _parse_gpt_json(self, content: str) -> dict:
        """Parse JSON response from GPT, handling markdown and LaTeX escape sequences."""
        # Remove markdown code blocks if present
        json_content = content
        if content.strip().startswith('```json'):
            # Extract JSON from markdown code block
            start = content.find('```json') + 7
            end = content.rfind('```')
            if end > start:
                json_content = content[start:end].strip()
        elif content.strip().startswith('```'):
            # Handle generic code blocks
            start = content.find('```') + 3
            end = content.rfind('```')
            if end > start:
                json_content = content[start:end].strip()
        
        # Try to parse JSON with proper error handling for escape sequences
        try:
            return json.loads(json_content)
        except json.JSONDecodeError as json_error:
            # Try to fix common escape sequence issues with LaTeX
            logger.warning(f"JSON parsing failed, attempting to fix escape sequences: {json_error}")
            
            # Replace problematic LaTeX escape sequences with proper JSON escapes
            fixed_content = json_content.replace('\\(', '\\\\(').replace('\\)', '\\\\)')
            fixed_content = fixed_content.replace('\\{', '\\\\{').replace('\\}', '\\\\}')
            fixed_content = fixed_content.replace('\\[', '\\\\[').replace('\\]', '\\\\]')
            
            # Handle comprehensive list of LaTeX commands
            latex_commands = [
                'frac', 'lim', 'sum', 'begin', 'end', 'cdot', 'alpha', 'to', 'geq', 'leq',
                'times', 'pm', 'mp', 'infty', 'sqrt', 'int', 'pi', 'theta', 'beta', 'gamma',
                'delta', 'epsilon', 'lambda', 'mu', 'nu', 'rho', 'sigma', 'tau', 'phi', 'chi',
                'psi', 'omega', 'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sup', 'inf',
                'max', 'min', 'arg', 'det', 'gcd', 'lcm', 'mod', 'bmod', 'pmod', 'cases',
                'left', 'right', 'big', 'Big', 'bigg', 'Bigg', 'text', 'textbf', 'textit',
                'mathbf', 'mathit', 'mathrm', 'mathcal', 'mathfrak', 'mathbb', 'overline',
                'underline', 'hat', 'tilde', 'vec', 'bar', 'dot', 'ddot', 'prime', 'dagger',
                'ddagger', 'star', 'ast', 'oplus', 'ominus', 'otimes', 'oslash', 'odot',
                'bigcup', 'bigcap', 'bigvee', 'bigwedge', 'biguplus', 'bigotimes', 'bigoplus'
            ]
            
            for cmd in latex_commands:
                fixed_content = fixed_content.replace(f'\\{cmd}', f'\\\\{cmd}')
            
            # Try parsing again
            return json.loads(fixed_content)
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 for GPT-4 Vision.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            raise
    
    def get_problems_extraction_prompt(self) -> str:
        """
        Get the system prompt for extracting problems (without solutions).
        Updated for high school focus and no curriculum classification.
        
        Returns:
            Prompt for extracting problem content, choices, and metadata
        """
        return """당신은 한국의 고등학교 수학 문제집 페이지를 분석하는 전문가입니다.
주어진 이미지에서 수학 문제들을 추출하되, 문제 내용과 메타데이터만 추출하고 정답이나 해설은 제외해주세요.

다음 규칙을 따라주세요:

1. **실제 수학 문제만 추출하라**: 페이지 번호, 헤더, 푸터, 출판사명, 시험 제목 등은 무시하고 오직 수학 문제 내용만 추출
2. **문제 번호 형식을 인식하라**: 1, 2, 3... 또는 (1), (2), (3)... 또는 ①, ②, ③... 등 다양한 형식의 문제 번호를 정확히 인식
3. **문제 유형 분류**:
   - "multiple_choice": 객관식 (선택지가 있는 경우)
   - "subjective": 주관식 (선택지가 없으면 주관식으로 분류)

4. **학년 분류** (고등학교만):
   - "고1", "고2", "고3"

5. **과목 분류** (고등학교 수학):
   - "수학상", "수학하", "수학Ⅰ", "수학Ⅱ", "미적분", "확률과통계", "기하"

6. **난이도 분류**:
   - "easy": 기초 수준
   - "medium": 표준 수준  
   - "hard": 심화 수준

6. **JSON 형식** (정답과 해설 제외):
```json
{
  "problems": [
    {
      "problem_number": 1,
      "content": "문제 본문 전체 (수식 포함)",
      "problem_type": "multiple_choice" 또는 "subjective",
      "choices": {
        "1": "선택지1",
        "2": "선택지2", 
        "3": "선택지3",
        "4": "선택지4",
        "5": "선택지5"
      }, // 객관식인 경우만, 주관식이면 null
      "level": "고1" 또는 "고2" 또는 "고3",
      "subject": "과목명",
      "chapter": "단원명",
      "difficulty": "easy/medium/hard",
      "tags": ["태그1", "태그2"], // 주요 개념 키워드
      "source_info": {
        "exam_type": "시험 유형 추정 (수능, 모의고사, 학교시험 등)",
        "problem_number": 문제번호,
        "total_points": 배점 // 명시되어 있는 경우만
      }
    }
  ]
}
```

중요사항:
- 수식은 LaTeX 형식으로 표현
- 그래프나 도형이 있으면 상세히 텍스트로 설명하고 "첨부된 이미지 참조"라고 명시
- 문제가 여러 개면 각각 분리하여 추출
- 정답과 해설은 절대 포함하지 마세요 (별도 파일에서 처리됩니다)
- 문제 번호를 정확히 인식하여 problem_number에 기록 (매우 중요!)
- 고등학교 수학만 처리 (중학교 내용 제외)
- 교육과정(2015개정/2022개정)은 파일명에서 처리되므로 제외
- 반드시 유효한 JSON 형식으로 응답

이제 이미지를 분석하여 고등학교 수학 문제들을 추출해주세요."""
    
    def get_single_problem_extraction_prompt(self) -> str:
        """
        Get the system prompt for extracting a single problem from an individual problem image.
        
        Returns:
            Prompt for extracting single problem content, choices, and metadata
        """
        return """당신은 한국의 고등학교 수학 문제집의 개별 문제를 분석하는 전문가입니다.
주어진 이미지는 하나의 수학 문제를 담고 있으며, 이 문제의 내용과 메타데이터를 추출해주세요.

다음 규칙을 따라주세요:

1. **실제 수학 문제만 추출하라**: 이미지에서 수학 문제의 핵심 내용만 추출하고, 불필요한 텍스트는 무시
2. **문제 번호 형식을 인식하라**: 1, 2, 3... 또는 (1), (2), (3)... 또는 ①, ②, ③... 등 다양한 형식의 문제 번호를 정확히 인식
3. **문제 유형 분류**:
   - "multiple_choice": 객관식 (선택지가 있는 경우)
   - "subjective": 주관식 (선택지가 없으면 주관식으로 분류)

4. **학년 분류** (고등학교만):
   - "고1", "고2", "고3"

5. **과목 분류** (고등학교 수학):
   - "수학상", "수학하", "수학Ⅰ", "수학Ⅱ", "미적분", "확률과통계", "기하"

6. **난이도 분류**:
   - "easy": 기초 수준
   - "medium": 표준 수준  
   - "hard": 심화 수준

7. **JSON 형식** (단일 문제):
```json
{
  "problems": [
    {
      "problem_number": 1,
      "content": "문제 본문 전체 (수식 포함)",
      "problem_type": "multiple_choice" 또는 "subjective",
      "choices": {
        "1": "선택지1",
        "2": "선택지2", 
        "3": "선택지3",
        "4": "선택지4",
        "5": "선택지5"
      }, // 객관식인 경우만, 주관식이면 null
      "level": "고1" 또는 "고2" 또는 "고3",
      "subject": "과목명",
      "chapter": "단원명",
      "difficulty": "easy/medium/hard",
      "tags": ["태그1", "태그2"], // 주요 개념 키워드
      "source_info": {
        "exam_type": "시험 유형 추정 (수능, 모의고사, 학교시험 등)",
        "problem_number": 문제번호,
        "total_points": 배점 // 명시되어 있는 경우만
      }
    }
  ]
}
```

중요사항:
- 수식은 LaTeX 형식으로 표현
- 추가 수학적 내용 이미지가 제공되면 "첨부된 이미지 참조"라고 명시하고 상세히 설명
- 정답과 해설은 절대 포함하지 마세요
- 문제 번호를 정확히 인식하여 problem_number에 기록 (매우 중요!)
- 고등학교 수학만 처리 (중학교 내용 제외)
- 반드시 유효한 JSON 형식으로 응답
- 이미지는 하나의 문제만 포함하므로 problems 배열에는 하나의 문제만 있어야 함

이제 이미지를 분석하여 이 개별 고등학교 수학 문제를 추출해주세요."""

    def get_solutions_extraction_prompt(self) -> str:
        """
        Get the system prompt for extracting solutions only.
        Updated for high school focus.
        
        Returns:
            Prompt for extracting correct answers and explanations
        """
        return """당신은 한국의 고등학교 수학 문제집 해답지 페이지를 분석하는 전문가입니다.
주어진 이미지에서 수학 문제의 정답과 해설만을 추출해주세요.

다음 규칙을 따라주세요:

1. **실제 수학 문제 해답만 추출하라**: 페이지 번호, 헤더, 푸터, 출판사명 등은 무시하고 오직 수학 문제의 해답과 해설만 추출
2. **문제 번호 형식을 인식하라**: 1, 2, 3... 또는 (1), (2), (3)... 또는 ①, ②, ③... 등 다양한 형식의 문제 번호를 정확히 인식하여 각 해답이 어느 문제에 대응되는지 번호로 정확히 파악 (매우 중요!)
3. **정답 추출**: 객관식은 번호(1,2,3,4,5), 주관식은 답을 정확히 추출
4. **해설 추출**: 상세한 풀이 과정과 설명을 모두 포함

4. **JSON 형식**:
```json
{
  "solutions": [
    {
      "problem_number": 1,
      "correct_answer": "3", // 객관식: "1","2","3","4","5" / 주관식: 실제 답
      "explanation": "상세한 해설 및 풀이 과정 전체",
      "solution_steps": [
        "1단계: ...",
        "2단계: ...",
        "3단계: ..."
      ] // 단계별 풀이가 있는 경우
    }
  ]
}
```

중요사항:
- 수식은 LaTeX 형식으로 표현
- 해설의 모든 단계와 설명을 빠짐없이 포함
- 문제 번호를 정확히 인식하여 problem_number에 기록 (매우 중요!)
- 그래프나 도형 설명도 해설에 상세히 포함하고 "첨부된 이미지 참조"라고 명시
- 반드시 유효한 JSON 형식으로 응답
- 해설이 여러 단계로 나뉘어 있다면 solution_steps에 단계별로 정리
- 고등학교 수학 수준의 상세한 해설 제공

이제 이미지를 분석하여 해답과 해설을 추출해주세요."""

    def extract_problem_from_image(self, problem_image_path: str, math_content_images: List[str] = None) -> Dict[str, Any]:
        """
        Extract a single problem from an individual problem image.
        
        Args:
            problem_image_path: Path to individual problem image
            math_content_images: List of paths to mathematical content images within this problem
            
        Returns:
            Dictionary containing extracted problem
        """
        try:
            logger.info(f"Extracting problem from: {problem_image_path}")
            
            # Encode main problem image
            base64_image = self.encode_image(problem_image_path)
            
            # Prepare content list starting with problem image
            content_list = [
                {
                    "type": "text",
                    "text": "이 고등학교 수학 문제를 분석하여 JSON 형식으로 추출해주세요. 정답과 해설은 제외하고 문제 내용과 메타데이터만 추출해주세요."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
            
            # Add mathematical content images if provided
            if math_content_images:
                content_list.append({
                    "type": "text",
                    "text": f"다음 {len(math_content_images)}개의 추가 이미지는 이 문제에 포함된 수학적 내용(그래프, 표, 도형 등)입니다:"
                })
                
                for i, content_image_path in enumerate(math_content_images, 1):
                    try:
                        content_base64 = self.encode_image(content_image_path)
                        content_list.append({
                            "type": "text",
                            "text": f"수학적 내용 {i}:"
                        })
                        content_list.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{content_base64}"
                            }
                        })
                    except Exception as e:
                        logger.warning(f"Failed to encode math content image {content_image_path}: {e}")
            
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": self.get_single_problem_extraction_prompt()
                },
                {
                    "role": "user", 
                    "content": content_list
                }
            ]
            
            # Call GPT-4o-mini
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Parse JSON response using the helper function
            try:
                result = self._parse_gpt_json(content)
                
                # Since this is a single problem, extract it from the problems array
                if 'problems' in result and result['problems']:
                    problem_data = result['problems'][0]  # Take the first (and should be only) problem
                    logger.info(f"Successfully extracted problem: {problem_data.get('problem_number', 'unknown')}")
                    return problem_data
                else:
                    logger.error("No problem found in GPT response")
                    return {"error": "No problem found", "raw_response": content}
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {content}")
                return {"error": "JSON parsing failed", "raw_response": content}
                
        except Exception as e:
            logger.error(f"Error extracting problem from {problem_image_path}: {e}")
            return {"error": str(e)}

    def extract_answers_from_answer_key(self, answer_key_dict: Dict[int, str]) -> Dict[str, Any]:
        """
        Convert answer key dictionary to the format expected by the processor.
        
        Args:
            answer_key_dict: Dictionary mapping problem numbers to answers
            
        Returns:
            Dictionary containing solutions in expected format
        """
        try:
            logger.info(f"Converting answer key with {len(answer_key_dict)} answers")
            
            solutions = []
            for problem_number, answer in answer_key_dict.items():
                solution = {
                    "problem_number": problem_number,
                    "correct_answer": answer
                }
                solutions.append(solution)
            
            result = {"solutions": solutions}
            logger.info(f"Successfully converted {len(solutions)} answers")
            return result
            
        except Exception as e:
            logger.error(f"Error converting answer key: {e}")
            return {"solutions": [], "error": str(e)}

    def extract_from_image_list(self, image_paths: List[str], extract_type: str = "problems", delay: float = 1.0, max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Extract from multiple images with rate limiting and optional concurrency.
        
        Args:
            image_paths: List of image file paths
            extract_type: "problems" or "solutions"
            delay: Delay between API calls (seconds)
            max_concurrent: Maximum concurrent requests (1 for sequential, >1 for concurrent)
            
        Returns:
            List of extraction results
        """
        logger.info(f"Processing batch of {len(image_paths)} images for {extract_type} (concurrent: {max_concurrent})")
        
        if max_concurrent <= 1:
            # Sequential processing (original method)
            return self._extract_sequential(image_paths, extract_type, delay)
        else:
            # Concurrent processing
            return self._extract_concurrent(image_paths, extract_type, delay, max_concurrent)
    
    def _extract_sequential(self, image_paths: List[str], extract_type: str, delay: float) -> List[Dict[str, Any]]:
        """Sequential processing (original method)."""
        results = []
        extract_func = self.extract_problems_from_image if extract_type == "problems" else self.extract_solutions_from_image
        
        for i, image_path in enumerate(tqdm(image_paths, desc=f"Extracting {extract_type}")):
            try:
                result = extract_func(image_path)
                result['source_image'] = image_path
                result['batch_index'] = i
                results.append(result)
                
                # Rate limiting
                if delay > 0 and i < len(image_paths) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append({
                    f"{extract_type}": [],
                    "error": str(e),
                    "source_image": image_path,
                    "batch_index": i
                })
        
        return results
    
    def _extract_concurrent(self, image_paths: List[str], extract_type: str, delay: float, max_concurrent: int) -> List[Dict[str, Any]]:
        """Concurrent processing using ThreadPoolExecutor."""
        results = [None] * len(image_paths)  # Pre-allocate to maintain order
        extract_func = self.extract_problems_from_image if extract_type == "problems" else self.extract_solutions_from_image
        
        def process_single_image(args):
            i, image_path = args
            try:
                result = extract_func(image_path)
                result['source_image'] = image_path
                result['batch_index'] = i
                return i, result
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                return i, {
                    f"{extract_type}": [],
                    "error": str(e),
                    "source_image": image_path,
                    "batch_index": i
                }
        
        # Process in batches to respect rate limits
        batch_size = max_concurrent
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            with tqdm(total=len(image_paths), desc=f"Extracting {extract_type}") as pbar:
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(image_paths))
                    batch_items = [(i, image_paths[i]) for i in range(start_idx, end_idx)]
                    
                    # Submit batch
                    futures = [executor.submit(process_single_image, item) for item in batch_items]
                    
                    # Collect results
                    for future in futures:
                        try:
                            idx, result = future.result()
                            results[idx] = result
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Concurrent processing error: {e}")
                            pbar.update(1)
                    
                    # Rate limiting between batches
                    if delay > 0 and batch_idx < total_batches - 1:
                        time.sleep(delay)
        
        return [r for r in results if r is not None]

    def match_problems_and_solutions_scoped(self, problems_list: List[Dict[str, Any]], solutions_list: List[Dict[str, Any]], exam_metadata: Dict[str, Any], extracted_images: Dict[str, List[str]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Match problems with solutions by problem number within the same exam (scoped).
        
        Args:
            problems_list: List of problem extraction results from one exam
            solutions_list: List of solution extraction results from same exam
            exam_metadata: Metadata about this specific exam
            extracted_images: Dictionary mapping page keys to extracted image filenames
            
        Returns:
            Tuple of (matched_problems, matching_report)
        """
        logger.info(f"Matching problems with solutions for exam: {exam_metadata.get('exam_date', 'unknown')}")
        
        # Initialize extracted_images if not provided
        if extracted_images is None:
            extracted_images = {}
        
        # Flatten all problems and solutions from this exam
        all_problems = []
        all_solutions = []
        
        for result in problems_list:
            if result.get("problems"):
                all_problems.extend(result["problems"])
        
        for result in solutions_list:
            if result.get("solutions"):
                all_solutions.extend(result["solutions"])
        
        # Create solution lookup by problem number (scoped to this exam)
        solutions_by_number = {}
        for solution in all_solutions:
            problem_num = solution.get("problem_number")
            if problem_num is not None:
                solutions_by_number[problem_num] = solution
        
        # Match problems with solutions
        matched_problems = []
        matching_report = {
            "exam_info": exam_metadata,
            "total_problems": len(all_problems),
            "total_solutions": len(all_solutions),
            "matched_by_number": 0,
            "matched_by_position": 0,
            "unmatched_problems": [],
            "unmatched_solutions": list(solutions_by_number.keys()),
            "errors": []
        }
        
        for i, problem in enumerate(all_problems):
            problem_num = problem.get("problem_number")
            solution = None
            match_method = None
            
            # Try to match by problem number first (preferred method)
            if problem_num is not None and problem_num in solutions_by_number:
                solution = solutions_by_number[problem_num]
                matching_report["unmatched_solutions"].remove(problem_num)
                match_method = "number"
                matching_report["matched_by_number"] += 1
            
            # Fallback: match by position if problem number missing/failed
            elif i < len(all_solutions):
                solution = all_solutions[i]
                match_method = "position"
                matching_report["matched_by_position"] += 1
                logger.warning(f"Using position fallback for problem {i+1} (problem_number: {problem_num})")
            
            # Find images for this problem (look for images from the same page)
            problem_images = []
            for page_key, image_list in extracted_images.items():
                if f"problem" in page_key:
                    # Add all images from problem pages for now
                    # Could be more sophisticated matching based on page number
                    problem_images.extend(image_list)
            
            # Create combined problem entry
            if solution:
                # Add exam metadata to source_info
                source_info = problem.get("source_info", {})
                source_info.update({
                    "exam_date": exam_metadata.get("exam_date"),
                    "exam_year": exam_metadata.get("exam_year"),
                    "exam_month": exam_metadata.get("exam_month"),
                    "exam_day": exam_metadata.get("exam_day"),
                    "exam_type": exam_metadata.get("exam_type"),
                    "problems_file": exam_metadata.get("problems_file"),
                    "solutions_file": exam_metadata.get("solutions_file"),
                    "processed_date": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                combined_problem = {
                    **problem,  # All problem data
                    "correct_answer": solution.get("correct_answer", ""),
                    "images": problem_images,  # Add extracted images
                    "match_method": match_method,
                    "source_info": source_info
                }
                matched_problems.append(combined_problem)
            else:
                # Problem without solution
                matching_report["unmatched_problems"].append({
                    "problem_number": problem_num,
                    "position": i,
                    "content": problem.get("content", "")[:100] + "..."
                })
                
                # Still add the problem without solution data
                source_info = problem.get("source_info", {})
                source_info.update({
                    "exam_date": exam_metadata.get("exam_date"),
                    "exam_year": exam_metadata.get("exam_year"),
                    "exam_month": exam_metadata.get("exam_month"),
                    "exam_day": exam_metadata.get("exam_day"),
                    "exam_type": exam_metadata.get("exam_type"),
                    "problems_file": exam_metadata.get("problems_file"),
                    "solutions_file": exam_metadata.get("solutions_file"),
                    "processed_date": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
                combined_problem = {
                    **problem,
                    "correct_answer": "",
                    "images": problem_images,  # Add extracted images
                    "match_method": "none",
                    "source_info": source_info
                }
                matched_problems.append(combined_problem)
        
        total_matched = matching_report["matched_by_number"] + matching_report["matched_by_position"]
        matching_report["total_matched"] = total_matched
        matching_report["success_rate"] = total_matched / max(1, matching_report["total_problems"])
        
        logger.info(f"Matching completed for {exam_metadata.get('exam_date', 'unknown')}: {total_matched}/{matching_report['total_problems']} problems matched ({matching_report['success_rate']:.1%})")
        
        return matched_problems, matching_report


def main():
    """CLI interface for GPT extraction with scoped matching."""
    parser = argparse.ArgumentParser(
        description="Extract math problems and solutions from images using GPT-4o-mini with scoped matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract problems only
  python gpt_extractor.py --problems /path/to/problem/images/
  
  # Extract solutions only  
  python gpt_extractor.py --solutions /path/to/solution/images/
  
  # Extract and match both (scoped to same exam)
  python gpt_extractor.py --problems prob_images/ --solutions sol_images/ --exam-date 2024-06-06
"""
    )
    
    parser.add_argument(
        "--problems",
        help="Directory containing problem images"
    )
    
    parser.add_argument(
        "--solutions", 
        help="Directory containing solution images"
    )
    
    parser.add_argument(
        "--exam-date",
        help="Exam date (YYYY-MM-DD) for metadata"
    )
    
    parser.add_argument(
        "--exam-type",
        default="exam",
        help="Exam type (suneung, mock, school, etc.)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output JSON file (default: auto-generated)"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if not args.problems and not args.solutions:
        parser.error("Must specify either --problems or --solutions (or both)")
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize extractor
        extractor = GPTExtractor()
        
        # Create exam metadata
        exam_metadata = {
            "exam_date": args.exam_date or "unknown",
            "exam_type": args.exam_type,
            "problems_file": args.problems,
            "solutions_file": args.solutions
        }
        
        problems_results = []
        solutions_results = []
        
        # Extract problems
        if args.problems:
            if os.path.isdir(args.problems):
                image_files = [f for f in os.listdir(args.problems) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
                image_paths = [os.path.join(args.problems, f) for f in sorted(image_files)]
                problems_results = extractor.extract_from_image_list(image_paths, "problems", args.delay)
            else:
                print(f"Error: Problems directory not found: {args.problems}")
                return 1
        
        # Extract solutions
        if args.solutions:
            if os.path.isdir(args.solutions):
                image_files = [f for f in os.listdir(args.solutions) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
                image_paths = [os.path.join(args.solutions, f) for f in sorted(image_files)]
                solutions_results = extractor.extract_from_image_list(image_paths, "solutions", args.delay)
            else:
                print(f"Error: Solutions directory not found: {args.solutions}")
                return 1
        
        # Process results
        final_result = {}
        
        if problems_results and solutions_results:
            # Match problems and solutions (scoped to this exam)
            matched_problems, matching_report = extractor.match_problems_and_solutions_scoped(
                problems_results, solutions_results, exam_metadata
            )
            final_result = {
                "matched_problems": matched_problems,
                "matching_report": matching_report,
                "exam_metadata": exam_metadata
            }
        elif problems_results:
            # Problems only
            all_problems = []
            for result in problems_results:
                if result.get("problems"):
                    all_problems.extend(result["problems"])
            final_result = {
                "problems": all_problems,
                "exam_metadata": exam_metadata
            }
        elif solutions_results:
            # Solutions only
            all_solutions = []
            for result in solutions_results:
                if result.get("solutions"):
                    all_solutions.extend(result["solutions"])
            final_result = {
                "solutions": all_solutions,
                "exam_metadata": exam_metadata
            }
        
        # Save results
        if not args.output:
            timestamp = int(time.time())
            exam_date = args.exam_date or "unknown"
            args.output = f"extracted_{exam_date}_{timestamp}.json"
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print(f"\n=== Extraction Complete ===")
        print(f"Exam: {exam_metadata['exam_date']} ({exam_metadata['exam_type']})")
        print(f"Output file: {args.output}")
        
        if "matched_problems" in final_result:
            matched_count = len(final_result["matched_problems"])
            print(f"Matched problems: {matched_count}")
            
            report = final_result["matching_report"]
            print(f"Success rate: {report['total_matched']}/{report['total_problems']} ({report['success_rate']:.1%})")
            print(f"Matched by number: {report['matched_by_number']}")
            print(f"Matched by position: {report['matched_by_position']}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())