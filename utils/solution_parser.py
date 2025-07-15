"""
Solution parser module for MathRush DataProcessor.
Extracts answers from the answer key box on the first page of solutions PDF.
"""

import cv2
import numpy as np
import os
from typing import Dict, Optional, List, Tuple, Any
import logging
from PIL import Image
import pytesseract
import re

logger = logging.getLogger(__name__)


class SolutionParser:
    """Parse answer key box to extract correct answers."""
    
    def __init__(self, 
                 answer_box_region: Tuple[float, float, float, float] = (0.0, 0.0, 0.4, 0.6),
                 min_confidence: float = 0.5):
        """
        Initialize solution parser.
        
        Args:
            answer_box_region: (x_ratio, y_ratio, width_ratio, height_ratio) for answer box location
            min_confidence: Minimum confidence for answer detection
        """
        self.answer_box_region = answer_box_region
        self.min_confidence = min_confidence
        
        # Mapping for circled numbers used in multiple choice
        self.circled_numbers = {
            '①': '1', '②': '2', '③': '3', '④': '4', '⑤': '5',
            '⑥': '6', '⑦': '7', '⑧': '8', '⑨': '9', '⑩': '10',
            '⑪': '11', '⑫': '12', '⑬': '13', '⑭': '14', '⑮': '15',
            '⑯': '16', '⑰': '17', '⑱': '18', '⑲': '19', '⑳': '20'
        }
        
        logger.info(f"Solution parser initialized with answer box region: {answer_box_region}")
    
    def parse_answer_key(self, solutions_first_page_path: str) -> Dict[int, str]:
        """
        Parse answer key from the first page of solutions PDF.
        
        Args:
            solutions_first_page_path: Path to the first page image of solutions PDF
            
        Returns:
            Dictionary mapping problem number to answer (e.g., {1: '③', 2: '15', 3: '①'})
        """
        try:
            logger.info(f"Parsing answer key from: {solutions_first_page_path}")
            
            # Load image
            image = cv2.imread(solutions_first_page_path)
            if image is None:
                logger.error(f"Failed to load solutions image: {solutions_first_page_path}")
                return {}
            
            # Extract answer box region
            answer_box = self._extract_answer_box(image)
            if answer_box is None:
                logger.error("Failed to extract answer box region")
                return {}
            
            # Parse answers from the answer box
            answers = self._parse_answers_from_box(answer_box)
            
            logger.info(f"Parsed {len(answers)} answers from answer key")
            return answers
            
        except Exception as e:
            logger.error(f"Error parsing answer key from {solutions_first_page_path}: {e}")
            return {}
    
    def _extract_answer_box(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the answer box region from the solutions page.
        
        Args:
            image: Full solutions page image
            
        Returns:
            Cropped answer box image or None if not found
        """
        try:
            height, width = image.shape[:2]
            
            # Calculate answer box coordinates
            x_ratio, y_ratio, width_ratio, height_ratio = self.answer_box_region
            
            x = int(width * x_ratio)
            y = int(height * y_ratio)
            w = int(width * width_ratio)
            h = int(height * height_ratio)
            
            # Extract the region
            answer_box = image[y:y+h, x:x+w]
            
            logger.debug(f"Extracted answer box region: ({x}, {y}, {w}, {h})")
            return answer_box
            
        except Exception as e:
            logger.error(f"Error extracting answer box: {e}")
            return None
    
    def _parse_answers_from_box(self, answer_box: np.ndarray) -> Dict[int, str]:
        """
        Parse answers from the answer box image.
        
        Args:
            answer_box: Cropped answer box image
            
        Returns:
            Dictionary mapping problem number to answer
        """
        answers = {}
        
        # Method 1: OCR-based parsing
        ocr_answers = self._parse_answers_by_ocr(answer_box)
        answers.update(ocr_answers)
        
        # Method 2: Pattern-based parsing for structured layouts
        pattern_answers = self._parse_answers_by_pattern(answer_box)
        answers.update(pattern_answers)
        
        # Method 3: Contour-based parsing for circled numbers
        contour_answers = self._parse_answers_by_contour(answer_box)
        answers.update(contour_answers)
        
        # Clean and validate answers
        cleaned_answers = self._clean_and_validate_answers(answers)
        
        return cleaned_answers
    
    def _parse_answers_by_ocr(self, answer_box: np.ndarray) -> Dict[int, str]:
        """Parse answers using OCR."""
        answers = {}
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(answer_box, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            enhanced = cv2.equalizeHist(gray)
            
            # OCR configuration for Korean exam answer sheets
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'
            
            # Extract text with bounding boxes
            data = pytesseract.image_to_data(enhanced, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Group text by lines to identify problem-answer pairs
            lines = self._group_text_by_lines(data)
            
            for line_text, line_confidence in lines:
                if line_confidence > self.min_confidence * 100:
                    # Look for problem-answer patterns
                    problem_answers = self._extract_problem_answers_from_line(line_text)
                    answers.update(problem_answers)
        
        except Exception as e:
            logger.debug(f"OCR parsing failed: {e}")
        
        return answers
    
    def _parse_answers_by_pattern(self, answer_box: np.ndarray) -> Dict[int, str]:
        """Parse answers using pattern matching for structured layouts."""
        answers = {}
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(answer_box, cv2.COLOR_BGR2GRAY)
            
            # Look for structured patterns (e.g., grid layout)
            # This is a simplified approach - in practice, you'd analyze the specific layout
            
            # Divide the answer box into a grid and process each cell
            rows = self._detect_answer_rows(gray)
            
            for row_idx, row_region in enumerate(rows):
                # Process each row to find answers
                row_answers = self._process_answer_row(row_region, row_idx)
                answers.update(row_answers)
        
        except Exception as e:
            logger.debug(f"Pattern parsing failed: {e}")
        
        return answers
    
    def _parse_answers_by_contour(self, answer_box: np.ndarray) -> Dict[int, str]:
        """Parse answers by detecting contours (especially for circled numbers)."""
        answers = {}
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(answer_box, cv2.COLOR_BGR2GRAY)
            
            # Binary threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Group contours by position (left to right, top to bottom)
            contour_groups = self._group_contours_by_position(contours)
            
            for group_idx, group in enumerate(contour_groups):
                # Process each group to find problem-answer pairs
                group_answers = self._process_contour_group(gray, group, group_idx)
                answers.update(group_answers)
        
        except Exception as e:
            logger.debug(f"Contour parsing failed: {e}")
        
        return answers
    
    def _group_text_by_lines(self, ocr_data: Dict[str, List]) -> List[Tuple[str, float]]:
        """Group OCR text by lines."""
        lines = []
        
        try:
            # Group by y-coordinate (lines)
            text_items = []
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                if text:
                    text_items.append({
                        'text': text,
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'conf': ocr_data['conf'][i]
                    })
            
            # Sort by y-coordinate
            text_items.sort(key=lambda x: x['y'])
            
            # Group items with similar y-coordinates
            current_line = []
            current_y = None
            y_threshold = 20  # Pixels
            
            for item in text_items:
                if current_y is None or abs(item['y'] - current_y) <= y_threshold:
                    current_line.append(item)
                    current_y = item['y']
                else:
                    # Process current line
                    if current_line:
                        line_text = ' '.join([item['text'] for item in sorted(current_line, key=lambda x: x['x'])])
                        line_conf = sum([item['conf'] for item in current_line]) / len(current_line)
                        lines.append((line_text, line_conf))
                    
                    # Start new line
                    current_line = [item]
                    current_y = item['y']
            
            # Process last line
            if current_line:
                line_text = ' '.join([item['text'] for item in sorted(current_line, key=lambda x: x['x'])])
                line_conf = sum([item['conf'] for item in current_line]) / len(current_line)
                lines.append((line_text, line_conf))
        
        except Exception as e:
            logger.debug(f"Line grouping failed: {e}")
        
        return lines
    
    def _extract_problem_answers_from_line(self, line_text: str) -> Dict[int, str]:
        """Extract problem-answer pairs from a text line."""
        answers = {}
        
        # Common patterns for Korean exam answer sheets
        patterns = [
            r'(\d+)\s*[:\-\.]\s*([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])',  # "1: ①" or "1. ③"
            r'(\d+)\s*[:\-\.]\s*(\d+)',  # "1: 15" or "1. 23"
            r'(\d+)\s+([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])',  # "1 ①"
            r'(\d+)\s+(\d+)',  # "1 15"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, line_text)
            for match in matches:
                try:
                    problem_num = int(match[0])
                    answer = match[1]
                    
                    # Convert circled numbers to regular numbers for consistency
                    if answer in self.circled_numbers:
                        answer = self.circled_numbers[answer]
                    
                    answers[problem_num] = answer
                except ValueError:
                    continue
        
        return answers
    
    def _detect_answer_rows(self, gray: np.ndarray) -> List[np.ndarray]:
        """Detect rows in the answer box."""
        rows = []
        
        try:
            height, width = gray.shape
            
            # Simple approach: divide into equal rows
            # In practice, you'd want to detect actual row boundaries
            num_rows = max(1, height // 40)  # Assume ~40 pixels per row
            row_height = height // num_rows
            
            for i in range(num_rows):
                y_start = i * row_height
                y_end = min((i + 1) * row_height, height)
                row = gray[y_start:y_end, :]
                rows.append(row)
        
        except Exception as e:
            logger.debug(f"Row detection failed: {e}")
        
        return rows
    
    def _process_answer_row(self, row_region: np.ndarray, row_idx: int) -> Dict[int, str]:
        """Process a single answer row."""
        answers = {}
        
        try:
            # OCR on the row
            custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'
            text = pytesseract.image_to_string(row_region, config=custom_config).strip()
            
            # Extract answers from the row text
            row_answers = self._extract_problem_answers_from_line(text)
            answers.update(row_answers)
        
        except Exception as e:
            logger.debug(f"Row processing failed: {e}")
        
        return answers
    
    def _group_contours_by_position(self, contours: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Group contours by their position (for structured layouts)."""
        groups = []
        
        try:
            # Get bounding rectangles for all contours
            contour_rects = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                contour_rects.append((contour, x, y, w, h))
            
            # Sort by position (top to bottom, left to right)
            contour_rects.sort(key=lambda x: (x[2], x[1]))  # Sort by y, then x
            
            # Group by rows (similar y-coordinates)
            current_group = []
            current_y = None
            y_threshold = 30  # Pixels
            
            for contour, x, y, w, h in contour_rects:
                if current_y is None or abs(y - current_y) <= y_threshold:
                    current_group.append(contour)
                    current_y = y
                else:
                    if current_group:
                        groups.append(current_group)
                    current_group = [contour]
                    current_y = y
            
            # Add last group
            if current_group:
                groups.append(current_group)
        
        except Exception as e:
            logger.debug(f"Contour grouping failed: {e}")
        
        return groups
    
    def _process_contour_group(self, gray: np.ndarray, contour_group: List[np.ndarray], group_idx: int) -> Dict[int, str]:
        """Process a group of contours."""
        answers = {}
        
        try:
            # For each contour in the group, extract and classify
            for contour in contour_group:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract the region
                roi = gray[y:y+h, x:x+w]
                
                # Classify the region
                classification = self._classify_contour_region(roi)
                
                if classification:
                    # For now, assume sequential problem numbering
                    # In practice, you'd want more sophisticated problem number detection
                    problem_num = group_idx + 1
                    answers[problem_num] = classification
        
        except Exception as e:
            logger.debug(f"Contour group processing failed: {e}")
        
        return answers
    
    def _classify_contour_region(self, roi: np.ndarray) -> Optional[str]:
        """Classify a contour region as an answer."""
        try:
            # OCR on the small region
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'
            text = pytesseract.image_to_string(roi, config=custom_config).strip()
            
            if text:
                # Convert circled numbers
                if text in self.circled_numbers:
                    return self.circled_numbers[text]
                elif text.isdigit():
                    return text
                else:
                    # Try to extract numbers from the text
                    numbers = re.findall(r'\d+', text)
                    if numbers:
                        return numbers[0]
        
        except Exception as e:
            logger.debug(f"Region classification failed: {e}")
        
        return None
    
    def _clean_and_validate_answers(self, answers: Dict[int, str]) -> Dict[int, str]:
        """Clean and validate the extracted answers."""
        cleaned = {}
        
        for problem_num, answer in answers.items():
            # Validate problem number
            if not isinstance(problem_num, int) or problem_num < 1 or problem_num > 100:
                logger.warning(f"Invalid problem number: {problem_num}")
                continue
            
            # Clean answer
            cleaned_answer = str(answer).strip()
            
            # Validate answer format
            if self._is_valid_answer(cleaned_answer):
                cleaned[problem_num] = cleaned_answer
            else:
                logger.warning(f"Invalid answer format for problem {problem_num}: {cleaned_answer}")
        
        return cleaned
    
    def _is_valid_answer(self, answer: str) -> bool:
        """Check if an answer is in valid format."""
        # Valid answers are either:
        # 1. Single digits (1-5 for multiple choice)
        # 2. Numbers (for subjective questions)
        # 3. Circled numbers (①②③④⑤)
        
        if not answer:
            return False
        
        # Check if it's a single digit (1-5)
        if answer.isdigit() and len(answer) == 1 and '1' <= answer <= '5':
            return True
        
        # Check if it's a number (for subjective answers)
        if answer.isdigit() and len(answer) <= 4:  # Reasonable limit
            return True
        
        # Check if it's a circled number
        if answer in self.circled_numbers.values():
            return True
        
        return False