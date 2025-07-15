"""
Problem segmentation module for MathRush DataProcessor.
Segments page images into individual problems based on problem numbers.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Tuple, Optional
import logging
from PIL import Image
import pytesseract
import re

logger = logging.getLogger(__name__)


class ProblemSegmenter:
    """Segment page images into individual problems based on problem numbers."""
    
    def __init__(self, 
                 min_problem_height: int = 150,
                 max_problem_height: int = 800,
                 header_height: int = 100,
                 footer_height: int = 80,
                 margin_left: int = 50,
                 margin_right: int = 50):
        """
        Initialize problem segmenter.
        
        Args:
            min_problem_height: Minimum height for a problem section
            max_problem_height: Maximum height for a problem section
            header_height: Height to ignore at top (exam title)
            footer_height: Height to ignore at bottom (page number)
            margin_left: Left margin to maintain
            margin_right: Right margin to maintain
        """
        self.min_problem_height = min_problem_height
        self.max_problem_height = max_problem_height
        self.header_height = header_height
        self.footer_height = footer_height
        self.margin_left = margin_left
        self.margin_right = margin_right
        
        logger.info(f"Problem segmenter initialized with min_height={min_problem_height}, max_height={max_problem_height}")
    
    def segment_page_into_problems(self, page_image_path: str, output_dir: str, page_key: str) -> List[Dict[str, Any]]:
        """
        Segment a page image into individual problems.
        
        Args:
            page_image_path: Path to the page image
            output_dir: Directory to save problem images
            page_key: Unique key for this page (e.g., "page_001")
            
        Returns:
            List of problem dictionaries with image paths and metadata
        """
        try:
            logger.info(f"Segmenting page into problems: {page_image_path}")
            
            # Load image
            image = cv2.imread(page_image_path)
            if image is None:
                logger.error(f"Failed to load image: {page_image_path}")
                return []
            
            height, width = image.shape[:2]
            
            # Create working area (exclude header and footer)
            work_top = self.header_height
            work_bottom = height - self.footer_height
            work_left = self.margin_left
            work_right = width - self.margin_right
            
            # Extract working area
            working_area = image[work_top:work_bottom, work_left:work_right]
            
            # Detect problem numbers and boundaries
            problem_boundaries = self._detect_problem_boundaries(working_area)
            
            # Segment into individual problems
            problems = []
            for i, (start_y, end_y, problem_number) in enumerate(problem_boundaries):
                # Adjust coordinates back to original image
                actual_start_y = work_top + start_y
                actual_end_y = work_top + end_y
                
                # Extract problem image
                problem_image = image[actual_start_y:actual_end_y, work_left:work_right]
                
                # Save problem image
                problem_filename = f"{page_key}_problem_{problem_number:02d}.png"
                problem_path = os.path.join(output_dir, problem_filename)
                cv2.imwrite(problem_path, problem_image)
                
                problem_info = {
                    'number': problem_number,
                    'image_path': problem_path,
                    'bbox': (work_left, actual_start_y, work_right - work_left, actual_end_y - actual_start_y),
                    'page_key': page_key,
                    'filename': problem_filename
                }
                problems.append(problem_info)
                
                logger.debug(f"Extracted problem {problem_number}: {problem_filename}")
            
            logger.info(f"Successfully segmented {len(problems)} problems from {page_key}")
            return problems
            
        except Exception as e:
            logger.error(f"Error segmenting page {page_image_path}: {e}")
            return []
    
    def _detect_problem_boundaries(self, working_area: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect problem boundaries based on problem numbers.
        
        Args:
            working_area: Cropped image without header/footer
            
        Returns:
            List of tuples (start_y, end_y, problem_number)
        """
        height, width = working_area.shape[:2]
        
        # Convert to grayscale for better text detection
        gray = cv2.cvtColor(working_area, cv2.COLOR_BGR2GRAY)
        
        # Detect problem numbers using OCR and pattern matching
        problem_positions = self._find_problem_numbers(gray)
        
        if not problem_positions:
            # Fallback: if no problem numbers detected, return whole area as one problem
            logger.warning("No problem numbers detected, treating entire area as one problem")
            return [(0, height, 1)]
        
        # Sort by y-coordinate
        problem_positions.sort(key=lambda x: x[1])
        
        # Calculate boundaries
        boundaries = []
        for i, (number, y, confidence) in enumerate(problem_positions):
            start_y = y if i == 0 else max(0, y - 50)  # Start a bit before problem number
            
            # End is either next problem start or end of image
            if i < len(problem_positions) - 1:
                next_y = problem_positions[i + 1][1]
                end_y = min(height, next_y - 10)  # End a bit before next problem
            else:
                end_y = height
            
            # Validate problem height
            problem_height = end_y - start_y
            if problem_height < self.min_problem_height:
                logger.warning(f"Problem {number} height too small ({problem_height}px), skipping")
                continue
            
            if problem_height > self.max_problem_height:
                logger.warning(f"Problem {number} height too large ({problem_height}px), truncating")
                end_y = start_y + self.max_problem_height
            
            boundaries.append((start_y, end_y, number))
        
        return boundaries
    
    def _find_problem_numbers(self, gray_image: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Find problem numbers in the image using OCR and pattern matching.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            List of tuples (problem_number, y_position, confidence)
        """
        problem_numbers = []
        
        # Method 1: OCR-based detection
        ocr_numbers = self._find_numbers_by_ocr(gray_image)
        problem_numbers.extend(ocr_numbers)
        
        # Method 2: Template matching for common number patterns
        template_numbers = self._find_numbers_by_template(gray_image)
        problem_numbers.extend(template_numbers)
        
        # Method 3: Contour-based detection for bold numbers
        contour_numbers = self._find_numbers_by_contour(gray_image)
        problem_numbers.extend(contour_numbers)
        
        # Remove duplicates and sort
        unique_numbers = {}
        for number, y, confidence in problem_numbers:
            if number not in unique_numbers or confidence > unique_numbers[number][1]:
                unique_numbers[number] = (y, confidence)
        
        # Convert back to list format
        result = [(num, y, conf) for num, (y, conf) in unique_numbers.items()]
        result.sort(key=lambda x: x[0])  # Sort by problem number
        
        logger.debug(f"Found {len(result)} problem numbers: {[x[0] for x in result]}")
        return result
    
    def _find_numbers_by_ocr(self, gray_image: np.ndarray) -> List[Tuple[int, int, float]]:
        """Find problem numbers using OCR."""
        numbers = []
        
        try:
            # Focus on left side of image where problem numbers typically appear
            height, width = gray_image.shape
            left_region = gray_image[:, :width//4]  # Left 25% of image
            
            # OCR configuration for better number detection
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789().①②③④⑤⑥⑦⑧⑨⑩'
            
            # Extract text with bounding boxes
            data = pytesseract.image_to_data(left_region, config=custom_config, output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])
                
                if confidence > 30:  # Minimum confidence threshold
                    # Check for problem number patterns
                    number = self._extract_problem_number(text)
                    if number:
                        y = data['top'][i]
                        numbers.append((number, y, confidence / 100.0))
        
        except Exception as e:
            logger.debug(f"OCR detection failed: {e}")
        
        return numbers
    
    def _find_numbers_by_template(self, gray_image: np.ndarray) -> List[Tuple[int, int, float]]:
        """Find problem numbers using template matching."""
        numbers = []
        
        try:
            # Create templates for numbers 1-20 (common range)
            height, width = gray_image.shape
            left_region = gray_image[:, :width//4]
            
            # Look for bold number patterns
            for num in range(1, 21):
                # Create simple template (this is a simplified approach)
                template = self._create_number_template(num)
                if template is not None:
                    result = cv2.matchTemplate(left_region, template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= 0.6)  # Threshold for template matching
                    
                    for y, x in zip(locations[0], locations[1]):
                        confidence = result[y, x]
                        numbers.append((num, y, confidence))
        
        except Exception as e:
            logger.debug(f"Template matching failed: {e}")
        
        return numbers
    
    def _find_numbers_by_contour(self, gray_image: np.ndarray) -> List[Tuple[int, int, float]]:
        """Find problem numbers by detecting bold text contours."""
        numbers = []
        
        try:
            height, width = gray_image.shape
            left_region = gray_image[:, :width//4]
            
            # Enhance contrast for better contour detection
            enhanced = cv2.equalizeHist(left_region)
            
            # Binary threshold
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter for number-like characteristics
                aspect_ratio = w / h
                area = cv2.contourArea(contour)
                
                # Numbers are typically compact and not too wide
                if (0.2 < aspect_ratio < 1.5 and 
                    100 < area < 2000 and 
                    x < width//8):  # Must be on far left
                    
                    # Extract the region and try to classify
                    roi = left_region[y:y+h, x:x+w]
                    number = self._classify_number_region(roi)
                    
                    if number:
                        confidence = min(1.0, area / 1000.0)  # Simple confidence based on area
                        numbers.append((number, y, confidence))
        
        except Exception as e:
            logger.debug(f"Contour detection failed: {e}")
        
        return numbers
    
    def _extract_problem_number(self, text: str) -> Optional[int]:
        """Extract problem number from OCR text."""
        # Pattern matching for various number formats
        patterns = [
            r'^(\d+)\.',           # "1.", "2.", etc.
            r'^(\d+)\)',           # "1)", "2)", etc.
            r'^\((\d+)\)',         # "(1)", "(2)", etc.
            r'^(\d+)$',            # Just numbers
            r'[①②③④⑤⑥⑦⑧⑨⑩]'    # Circled numbers
        ]
        
        for pattern in patterns:
            if pattern == r'[①②③④⑤⑥⑦⑧⑨⑩]':
                # Handle circled numbers
                circled_map = {'①': 1, '②': 2, '③': 3, '④': 4, '⑤': 5, 
                              '⑥': 6, '⑦': 7, '⑧': 8, '⑨': 9, '⑩': 10}
                for char in text:
                    if char in circled_map:
                        return circled_map[char]
            else:
                match = re.search(pattern, text)
                if match:
                    try:
                        return int(match.group(1))
                    except (ValueError, IndexError):
                        continue
        
        return None
    
    def _create_number_template(self, number: int) -> Optional[np.ndarray]:
        """Create a simple template for a number (placeholder implementation)."""
        # This is a simplified template creation - in practice, you'd want 
        # to use actual font templates or trained templates
        try:
            # Create a simple template using PIL
            from PIL import Image, ImageDraw, ImageFont
            
            # Create small template image
            template_size = (40, 60)
            template = Image.new('L', template_size, 255)  # White background
            draw = ImageDraw.Draw(template)
            
            # Try to use a bold font
            try:
                font = ImageFont.truetype("Arial-Bold.ttf", 36)
            except:
                font = ImageFont.load_default()
            
            # Draw the number
            draw.text((10, 10), str(number), fill=0, font=font)  # Black text
            
            # Convert to numpy array
            template_array = np.array(template)
            return template_array
            
        except Exception as e:
            logger.debug(f"Template creation failed for number {number}: {e}")
            return None
    
    def _classify_number_region(self, roi: np.ndarray) -> Optional[int]:
        """Classify a region as a specific number (placeholder implementation)."""
        # This is a simplified classification - in practice, you'd want
        # to use a trained classifier or more sophisticated matching
        try:
            # Use OCR on the small region
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(roi, config=custom_config).strip()
            
            if text.isdigit():
                return int(text)
                
        except Exception as e:
            logger.debug(f"Region classification failed: {e}")
        
        return None
    
    def cleanup_temp_files(self, problem_list: List[Dict[str, Any]]) -> None:
        """Clean up temporary problem image files."""
        for problem in problem_list:
            if 'image_path' in problem:
                try:
                    if os.path.exists(problem['image_path']):
                        os.remove(problem['image_path'])
                        logger.debug(f"Cleaned up: {problem['image_path']}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {problem['image_path']}: {e}")