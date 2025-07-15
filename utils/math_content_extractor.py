"""
Mathematical content extractor for MathRush DataProcessor.
Extracts mathematical diagrams, graphs, and tables from individual problem images.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class MathContentExtractor:
    """Extract mathematical content (graphs, diagrams, tables) from problem images."""
    
    def __init__(self, 
                 min_area: int = 5000,
                 max_area_ratio: float = 0.6,
                 min_complexity_score: float = 0.1):
        """
        Initialize mathematical content extractor.
        
        Args:
            min_area: Minimum area for mathematical content
            max_area_ratio: Maximum area ratio relative to problem image
            min_complexity_score: Minimum complexity score for valid content
        """
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        self.min_complexity_score = min_complexity_score
        
        logger.info(f"Math content extractor initialized: min_area={min_area}, max_area_ratio={max_area_ratio}")
    
    def extract_mathematical_content(self, problem_image_path: str, output_dir: str, problem_key: str) -> List[str]:
        """
        Extract mathematical content from a problem image.
        
        Args:
            problem_image_path: Path to the problem image
            output_dir: Directory to save extracted content
            problem_key: Unique key for this problem (e.g., "page_001_problem_01")
            
        Returns:
            List of paths to extracted mathematical content images
        """
        try:
            logger.info(f"Extracting mathematical content from: {problem_image_path}")
            
            # Load image
            image = cv2.imread(problem_image_path)
            if image is None:
                logger.error(f"Failed to load problem image: {problem_image_path}")
                return []
            
            height, width = image.shape[:2]
            problem_area = height * width
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect mathematical content regions
            math_regions = self._detect_mathematical_regions(gray, problem_area)
            
            # Extract and save each region
            extracted_files = []
            for i, region in enumerate(math_regions, 1):
                x, y, w, h = region['bbox']
                region_image = image[y:y+h, x:x+w]
                
                # Save extracted content
                content_filename = f"{problem_key}_content_{i:02d}.png"
                content_path = os.path.join(output_dir, content_filename)
                cv2.imwrite(content_path, region_image)
                
                extracted_files.append(content_path)
                logger.debug(f"Extracted mathematical content: {content_filename}")
            
            logger.info(f"Extracted {len(extracted_files)} mathematical content pieces from {problem_key}")
            return extracted_files
            
        except Exception as e:
            logger.error(f"Error extracting mathematical content from {problem_image_path}: {e}")
            return []
    
    def _detect_mathematical_regions(self, gray: np.ndarray, problem_area: int) -> List[Dict[str, Any]]:
        """
        Detect regions containing mathematical content.
        
        Args:
            gray: Grayscale problem image
            problem_area: Total area of the problem image
            
        Returns:
            List of region dictionaries with bbox and metadata
        """
        regions = []
        
        # Method 1: Detect structured diagrams (graphs, charts)
        diagram_regions = self._detect_diagrams(gray, problem_area)
        regions.extend(diagram_regions)
        
        # Method 2: Detect table structures
        table_regions = self._detect_tables(gray, problem_area)
        regions.extend(table_regions)
        
        # Method 3: Detect coordinate systems
        coordinate_regions = self._detect_coordinate_systems(gray, problem_area)
        regions.extend(coordinate_regions)
        
        # Method 4: Detect geometric figures
        geometric_regions = self._detect_geometric_figures(gray, problem_area)
        regions.extend(geometric_regions)
        
        # Remove overlapping regions
        filtered_regions = self._remove_overlapping_regions(regions)
        
        # Sort by area (largest first)
        filtered_regions.sort(key=lambda r: r['area'], reverse=True)
        
        logger.debug(f"Detected {len(filtered_regions)} mathematical content regions")
        return filtered_regions
    
    def _detect_diagrams(self, gray: np.ndarray, problem_area: int) -> List[Dict[str, Any]]:
        """Detect diagram-like structures (graphs, charts)."""
        regions = []
        
        try:
            # Enhance contrast
            enhanced = cv2.equalizeHist(gray)
            
            # Edge detection
            edges = cv2.Canny(enhanced, 30, 100)
            
            # Morphological operations to connect nearby edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter by area
                if area < self.min_area or area > problem_area * self.max_area_ratio:
                    continue
                
                # Check aspect ratio (diagrams can be various shapes)
                aspect_ratio = w / h
                if aspect_ratio < 0.3 or aspect_ratio > 4.0:
                    continue
                
                # Check complexity
                roi = gray[y:y+h, x:x+w]
                complexity = self._calculate_complexity_score(roi)
                
                if complexity >= self.min_complexity_score:
                    regions.append({
                        'type': 'diagram',
                        'bbox': (x, y, w, h),
                        'area': area,
                        'complexity': complexity,
                        'aspect_ratio': aspect_ratio
                    })
        
        except Exception as e:
            logger.debug(f"Diagram detection failed: {e}")
        
        return regions
    
    def _detect_tables(self, gray: np.ndarray, problem_area: int) -> List[Dict[str, Any]]:
        """Detect table structures."""
        regions = []
        
        try:
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            # Detect horizontal lines
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Detect vertical lines
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine horizontal and vertical lines
            table_mask = cv2.add(horizontal_lines, vertical_lines)
            
            # Find contours in the table mask
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # Filter by area
                if area < self.min_area or area > problem_area * self.max_area_ratio:
                    continue
                
                # Tables typically have rectangular aspect ratios
                aspect_ratio = w / h
                if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                    continue
                
                # Check for grid-like pattern
                roi = gray[y:y+h, x:x+w]
                if self._has_grid_pattern(roi):
                    regions.append({
                        'type': 'table',
                        'bbox': (x, y, w, h),
                        'area': area,
                        'complexity': 0.8,  # Tables are inherently complex
                        'aspect_ratio': aspect_ratio
                    })
        
        except Exception as e:
            logger.debug(f"Table detection failed: {e}")
        
        return regions
    
    def _detect_coordinate_systems(self, gray: np.ndarray, problem_area: int) -> List[Dict[str, Any]]:
        """Detect coordinate systems and axes."""
        regions = []
        
        try:
            # Look for perpendicular lines that might form axes
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Group lines by orientation
                horizontal_lines = []
                vertical_lines = []
                
                for line in lines:
                    rho, theta = line[0]
                    
                    # Classify as horizontal or vertical
                    if abs(theta) < np.pi/4 or abs(theta - np.pi) < np.pi/4:  # Horizontal
                        horizontal_lines.append((rho, theta))
                    elif abs(theta - np.pi/2) < np.pi/4:  # Vertical
                        vertical_lines.append((rho, theta))
                
                # If we have both horizontal and vertical lines, likely a coordinate system
                if len(horizontal_lines) > 0 and len(vertical_lines) > 0:
                    # Find the region that contains the intersection
                    h, w = gray.shape
                    
                    # Simple heuristic: assume coordinate system is in center area
                    margin = min(w, h) // 4
                    x, y = margin, margin
                    width, height = w - 2*margin, h - 2*margin
                    area = width * height
                    
                    if area >= self.min_area:
                        regions.append({
                            'type': 'coordinate_system',
                            'bbox': (x, y, width, height),
                            'area': area,
                            'complexity': 0.9,  # Coordinate systems are complex
                            'aspect_ratio': width / height
                        })
        
        except Exception as e:
            logger.debug(f"Coordinate system detection failed: {e}")
        
        return regions
    
    def _detect_geometric_figures(self, gray: np.ndarray, problem_area: int) -> List[Dict[str, Any]]:
        """Detect geometric figures (circles, polygons, etc.)."""
        regions = []
        
        try:
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if area < self.min_area or perimeter < 100:
                    continue
                
                # Check for geometric characteristics
                x, y, w, h = cv2.boundingRect(contour)
                bbox_area = w * h
                
                # Skip if bounding box is too large
                if bbox_area > problem_area * self.max_area_ratio:
                    continue
                
                # Calculate circularity (4π*area/perimeter²)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Check if it's a reasonable geometric shape
                    if circularity > 0.3:  # Somewhat circular or regular
                        regions.append({
                            'type': 'geometric_figure',
                            'bbox': (x, y, w, h),
                            'area': bbox_area,
                            'complexity': circularity,
                            'aspect_ratio': w / h
                        })
        
        except Exception as e:
            logger.debug(f"Geometric figure detection failed: {e}")
        
        return regions
    
    def _calculate_complexity_score(self, roi: np.ndarray) -> float:
        """Calculate complexity score for a region."""
        try:
            # Edge density
            edges = cv2.Canny(roi, 50, 150)
            edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
            
            # Variance (texture complexity)
            variance = np.var(roi) / 10000.0  # Normalize
            
            # Contour complexity
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_complexity = len(contours) / 100.0  # Normalize
            
            # Combined score
            complexity = (edge_density + variance + contour_complexity) / 3.0
            return min(1.0, complexity)
            
        except Exception as e:
            logger.debug(f"Complexity calculation failed: {e}")
            return 0.0
    
    def _has_grid_pattern(self, roi: np.ndarray) -> bool:
        """Check if region has a grid-like pattern."""
        try:
            # Look for regular patterns in rows and columns
            # This is a simplified check - in practice, you'd want more sophisticated pattern detection
            
            # Check for horizontal line patterns
            horizontal_projection = np.sum(roi, axis=1)
            horizontal_peaks = len([i for i in range(1, len(horizontal_projection)-1) 
                                  if horizontal_projection[i] < horizontal_projection[i-1] and 
                                     horizontal_projection[i] < horizontal_projection[i+1]])
            
            # Check for vertical line patterns
            vertical_projection = np.sum(roi, axis=0)
            vertical_peaks = len([i for i in range(1, len(vertical_projection)-1) 
                                if vertical_projection[i] < vertical_projection[i-1] and 
                                   vertical_projection[i] < vertical_projection[i+1]])
            
            # A grid should have multiple peaks in both directions
            return horizontal_peaks >= 2 and vertical_peaks >= 2
            
        except Exception as e:
            logger.debug(f"Grid pattern check failed: {e}")
            return False
    
    def _remove_overlapping_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping regions, keeping the one with higher complexity."""
        if not regions:
            return []
        
        # Sort by complexity (descending)
        sorted_regions = sorted(regions, key=lambda r: r['complexity'], reverse=True)
        
        filtered = []
        for region in sorted_regions:
            x1, y1, w1, h1 = region['bbox']
            
            # Check if this region overlaps significantly with any already accepted region
            overlaps = False
            for accepted in filtered:
                x2, y2, w2, h2 = accepted['bbox']
                
                # Calculate overlap
                overlap_area = self._calculate_overlap_area(
                    (x1, y1, w1, h1), (x2, y2, w2, h2)
                )
                
                # If overlap is more than 30% of either region, consider it overlapping
                if overlap_area > 0.3 * min(w1*h1, w2*h2):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(region)
        
        return filtered
    
    def _calculate_overlap_area(self, bbox1: tuple, bbox2: tuple) -> float:
        """Calculate overlap area between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            return (right - left) * (bottom - top)
        else:
            return 0.0
    
    def cleanup_temp_files(self, file_list: List[str]) -> None:
        """Clean up temporary mathematical content files."""
        for file_path in file_list:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")