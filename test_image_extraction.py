"""
Test image extraction with improved filtering.
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from utils.image_extractor import ImageExtractor
from processors.pdf_converter import PDFConverter
import shutil

def test_image_extraction():
    """Test image extraction with improved filtering."""
    print("=== Testing Image Extraction ===\n")
    
    # Test files
    problems_pdf = "samples/2020-12-03_suneung_problems.pdf"
    
    if not os.path.exists(problems_pdf):
        print(f"❌ Test file not found: {problems_pdf}")
        return
    
    try:
        # Convert PDF to images first
        print("1. Converting PDF to images...")
        converter = PDFConverter()
        
        # Convert to images
        output_dir = "output/test_image_extraction"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        problems_images = converter.convert_pdf_to_images(problems_pdf, output_dir)
        print(f"✅ Converted {len(problems_images)} pages to images")
        
        # Initialize image extractor
        extractor = ImageExtractor()
        
        # Test on first few pages
        for i, image_path in enumerate(problems_images[:3], 1):
            print(f"\n2. Testing image extraction on page {i}: {os.path.basename(image_path)}")
            
            # Extract images
            page_key = f"page_{i:03d}"
            extracted_images = extractor.extract_images(image_path, output_dir, page_key)
            
            print(f"📊 Extracted {len(extracted_images)} images from page {i}")
            
            if extracted_images:
                print("   Image details:")
                for j, img_file in enumerate(extracted_images, 1):
                    print(f"   {j}. {img_file}")
            else:
                print("   ℹ️  No mathematical diagrams/graphs detected on this page")
        
        print(f"\n✅ Image extraction test completed")
        print(f"📁 Output directory: {output_dir}")
        
    except Exception as e:
        print(f"❌ Image extraction test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_extraction()