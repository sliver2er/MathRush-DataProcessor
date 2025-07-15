"""
Image extraction utility for math problems.
Detects and extracts graphs, diagrams, and figures from PDF page images.
"""

import cv2
import numpy as np
import os
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFilter
import logging

logger = logging.getLogger(__name__)


class ImageExtractor:
    """Extract images/diagrams from PDF page images."""
    
    def __init__(self, min_area: int = 10000, max_area_ratio: float = 0.8):
        """
        Initialize image extractor.
        
        Args:
            min_area: Minimum area for detected regions (pixels)
            max_area_ratio: Maximum area ratio relative to page size
        """
        self.min_area = min_area
        self.max_area_ratio = max_area_ratio
        
        logger.info(f"Image extractor initialized: min_area={min_area}, max_area_ratio={max_area_ratio}")
    
    def detect_image_regions(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect potential image regions in a PDF page.
        
        Args:
            image_path: Path to the PDF page image
            
        Returns:
            List of detected regions with bounding boxes and metadata
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return []
            
            height, width = image.shape[:2]
            page_area = height * width
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect different types of regions
            regions = []
            
            # Method 1: Detect enclosed rectangular regions (graphs, diagrams)
            regions.extend(self._detect_rectangular_regions(gray, page_area))
            
            # Method 2: Detect grid-like patterns (coordinate systems)
            regions.extend(self._detect_grid_patterns(gray, page_area))
            
            # Method 3: Detect table-like structures
            regions.extend(self._detect_table_structures(gray, page_area))
            
            # Remove overlapping regions
            regions = self._remove_overlapping_regions(regions)
            
            logger.info(f"Detected {len(regions)} image regions in {image_path}")
            return regions
            
        except Exception as e:
            logger.error(f"Error detecting image regions in {image_path}: {e}")
            return []
    
    def _detect_rectangular_regions(self, gray: np.ndarray, page_area: int) -> List[Dict[str, Any]]:
        """Detect rectangular regions that might contain graphs or diagrams."""
        regions = []
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by area
            if area < self.min_area or area > page_area * self.max_area_ratio:
                continue
            
            # Filter by aspect ratio (reasonable rectangles)
            aspect_ratio = w / h
            if aspect_ratio < 0.3 or aspect_ratio > 5.0:
                continue
            
            # Check if region contains significant content
            roi = gray[y:y+h, x:x+w]
            if self._has_significant_content(roi):
                regions.append({
                    'type': 'rectangular',
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': self._calculate_confidence(roi, 'rectangular')
                })
        
        return regions
    
    def _detect_grid_patterns(self, gray: np.ndarray, page_area: int) -> List[Dict[str, Any]]:
        """Detect grid-like patterns (coordinate systems, graph paper)."""
        regions = []
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        grid_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        
        # Find contours in grid mask
        contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area < self.min_area or area > page_area * self.max_area_ratio:
                continue
            
            # Check for grid-like pattern
            roi = gray[y:y+h, x:x+w]
            if self._has_grid_pattern(roi):
                regions.append({
                    'type': 'grid',
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': self._calculate_confidence(roi, 'grid')
                })
        
        return regions
    
    def _detect_table_structures(self, gray: np.ndarray, page_area: int) -> List[Dict[str, Any]]:
        """Detect table-like structures."""
        regions = []
        
        # Use morphological operations to detect table-like structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Find horizontal and vertical lines
        horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, 
                                    cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)))
        vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, 
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30)))
        
        # Combine to find table intersections
        table_mask = cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0)
        
        # Find contours
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            if area < self.min_area or area > page_area * self.max_area_ratio:
                continue
            
            roi = gray[y:y+h, x:x+w]
            if self._has_table_structure(roi):
                regions.append({
                    'type': 'table',
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': self._calculate_confidence(roi, 'table')
                })
        
        return regions
    
    def _has_significant_content(self, roi: np.ndarray) -> bool:
        """Check if region contains significant mathematical content (not page numbers, headers, etc.)."""
        h, w = roi.shape
        
        # 1. Region should be substantial enough to contain graphs/diagrams
        if h < 100 or w < 100:  # Minimum size for meaningful math diagrams
            return False
        
        # 2. Calculate variance - higher variance indicates more content
        variance = np.var(roi)
        if variance < 800:  # Higher threshold for mathematical content
            return False
        
        # 3. Check edge density - mathematical diagrams have more complex edges
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        if edge_density < 0.03:  # Higher threshold for mathematical diagrams
            return False
        
        # 4. Check for line complexity (graphs typically have multiple connected components)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Should have multiple contours for complex mathematical diagrams
        if len(contours) < 8:  # More contours expected for graphs/diagrams
            return False
        
        # 5. Check for curved/angled lines (typical in mathematical graphs)
        complex_contours = 0
        for contour in contours:
            if len(contour) > 15:  # Complex curves or detailed shapes
                complex_contours += 1
        
        return complex_contours >= 4  # Require multiple complex shapes
    
    def _has_grid_pattern(self, roi: np.ndarray) -> bool:
        """Check if region contains grid-like pattern."""
        # Look for regular patterns in horizontal and vertical directions
        horizontal_profile = np.mean(roi, axis=0)
        vertical_profile = np.mean(roi, axis=1)
        
        # Check for periodic patterns
        h_peaks = self._count_peaks(horizontal_profile)
        v_peaks = self._count_peaks(vertical_profile)
        
        return h_peaks >= 3 and v_peaks >= 3
    
    def _has_table_structure(self, roi: np.ndarray) -> bool:
        """Check if region contains table-like structure."""
        # Similar to grid pattern but more lenient
        edges = cv2.Canny(roi, 30, 100)
        
        # Look for horizontal and vertical line patterns
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, 
                                          cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1)))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, 
                                        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20)))
        
        h_line_density = np.sum(horizontal_lines > 0) / (roi.shape[0] * roi.shape[1])
        v_line_density = np.sum(vertical_lines > 0) / (roi.shape[0] * roi.shape[1])
        
        return h_line_density > 0.01 and v_line_density > 0.01
    
    def _count_peaks(self, profile: np.ndarray) -> int:
        """Count peaks in a profile (for grid detection)."""
        # Simple peak detection
        peaks = 0
        for i in range(1, len(profile) - 1):
            if profile[i] > profile[i-1] and profile[i] > profile[i+1]:
                peaks += 1
        return peaks
    
    def _calculate_confidence(self, roi: np.ndarray, region_type: str) -> float:
        """Calculate confidence score for detected region."""
        base_score = 0.5
        
        # Add variance-based confidence
        variance = np.var(roi)
        variance_score = min(variance / 1000, 0.3)
        
        # Add edge density confidence
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
        edge_score = min(edge_density * 10, 0.2)
        
        return base_score + variance_score + edge_score
    
    def _remove_overlapping_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping regions, keeping the one with higher confidence."""
        if not regions:
            return regions
        
        # Sort by confidence (descending)
        regions.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered_regions = []
        for region in regions:
            x1, y1, w1, h1 = region['bbox']
            
            # Check overlap with already selected regions
            overlap = False
            for selected in filtered_regions:
                x2, y2, w2, h2 = selected['bbox']
                
                # Calculate intersection area
                intersection_area = self._calculate_intersection_area(
                    (x1, y1, w1, h1), (x2, y2, w2, h2)
                )
                
                # If significant overlap, skip this region
                if intersection_area > min(w1 * h1, w2 * h2) * 0.3:
                    overlap = True
                    break
            
            if not overlap:
                filtered_regions.append(region)
        
        return filtered_regions
    
    def _calculate_intersection_area(self, bbox1: Tuple[int, int, int, int], 
                                   bbox2: Tuple[int, int, int, int]) -> int:
        """Calculate intersection area between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection coordinates
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        # Check if there's an intersection
        if left < right and top < bottom:
            return (right - left) * (bottom - top)
        return 0
    
    def extract_images(self, image_path: str, output_dir: str, 
                      base_name: str) -> List[str]:
        """
        Extract detected images from PDF page and save them.
        
        Args:
            image_path: Path to the PDF page image
            output_dir: Directory to save extracted images
            base_name: Base name for saved images
            
        Returns:
            List of paths to extracted image files
        """
        try:
            # Detect regions
            regions = self.detect_image_regions(image_path)
            
            if not regions:
                logger.debug(f"No image regions detected in {image_path}")
                return []
            
            # Load original image
            original = cv2.imread(image_path)
            if original is None:
                logger.error(f"Could not load original image: {image_path}")
                return []
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            extracted_paths = []
            
            for i, region in enumerate(regions, 1):
                x, y, w, h = region['bbox']
                
                # Extract region
                roi = original[y:y+h, x:x+w]
                
                # Generate filename
                filename = f"{base_name}_img_{i}.png"
                output_path = os.path.join(output_dir, filename)
                
                # Save extracted image
                cv2.imwrite(output_path, roi)
                extracted_paths.append(filename)  # Return relative filename
                
                logger.debug(f"Extracted {region['type']} image: {output_path}")
            
            logger.info(f"Extracted {len(extracted_paths)} images from {image_path}")
            return extracted_paths
            
        except Exception as e:
            logger.error(f"Error extracting images from {image_path}: {e}")
            return []


def test_image_extractor():
    """Test the image extractor with sample images."""
    extractor = ImageExtractor()
    
    # Test with sample images
    test_images = [
        "output/test_images/2606_probs_page_001.png",
        "output/test_images/2606_probs_page_002.png"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\nTesting: {image_path}")
            regions = extractor.detect_image_regions(image_path)
            
            print(f"Detected {len(regions)} regions:")
            for i, region in enumerate(regions, 1):
                print(f"  {i}. Type: {region['type']}, "
                      f"BBox: {region['bbox']}, "
                      f"Confidence: {region['confidence']:.3f}")
            
            # Extract images
            extracted = extractor.extract_images(
                image_path, 
                "output/extracted_test", 
                f"test_page_{os.path.basename(image_path).split('.')[0]}"
            )
            print(f"Extracted: {extracted}")
        else:
            print(f"Test image not found: {image_path}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_image_extractor()