"""
PDF to Image conversion module for MathRush DataProcessor.
Converts PDF pages to high-resolution images for GPT processing.
"""

import os
import argparse
import glob
from typing import List, Optional, Tuple
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import logging

# Import settings
try:
    from config.settings import settings
except ImportError:
    # Fallback if running directly
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from config.settings import settings

logger = logging.getLogger(__name__)


class PDFConverter:
    """Convert PDF documents to images for processing."""
    
    def __init__(self, dpi: Optional[int] = None, output_format: Optional[str] = None):
        """
        Initialize PDF converter.
        
        Args:
            dpi: Resolution for image conversion (uses settings.PDF_DPI if None)
            output_format: Output image format (uses settings.PDF_OUTPUT_FORMAT if None)
        """
        self.dpi = dpi or settings.PDF_DPI
        self.output_format = (output_format or settings.PDF_OUTPUT_FORMAT).upper()
        
    def convert_pdf_to_images(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[str]:
        """
        Convert PDF to images.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save images (optional)
            page_range: Tuple of (start_page, end_page) to convert specific pages
            
        Returns:
            List of image file paths or PIL Image objects
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            logger.info(f"Converting PDF: {pdf_path} (DPI: {self.dpi})")
            
            # Set page range if specified
            first_page = None
            last_page = None
            if page_range:
                first_page, last_page = page_range
                
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=first_page,
                last_page=last_page
            )
            
            logger.info(f"Successfully converted {len(images)} pages")
            
            # Save images if output directory is specified
            if output_dir:
                return self._save_images(images, pdf_path, output_dir)
            else:
                # Return PIL Image objects
                return images
                
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path}: {str(e)}")
            raise
    
    def convert_pdf_from_bytes(
        self,
        pdf_bytes: bytes,
        output_dir: Optional[str] = None,
        filename_prefix: str = "page"
    ) -> List[str]:
        """
        Convert PDF from bytes to images.
        
        Args:
            pdf_bytes: PDF file as bytes
            output_dir: Directory to save images (optional)
            filename_prefix: Prefix for saved image files
            
        Returns:
            List of image file paths or PIL Image objects
        """
        try:
            logger.info(f"Converting PDF from bytes (DPI: {self.dpi})")
            
            images = convert_from_bytes(pdf_bytes, dpi=self.dpi)
            
            logger.info(f"Successfully converted {len(images)} pages from bytes")
            
            # Save images if output directory is specified
            if output_dir:
                return self._save_images(images, filename_prefix, output_dir)
            else:
                return images
                
        except Exception as e:
            logger.error(f"Error converting PDF from bytes: {str(e)}")
            raise
    
    def _save_images(
        self,
        images: List[Image.Image],
        source_name: str,
        output_dir: str
    ) -> List[str]:
        """
        Save PIL images to files.
        
        Args:
            images: List of PIL Image objects
            source_name: Source filename or prefix
            output_dir: Output directory
            
        Returns:
            List of saved image file paths
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract base filename without extension
            if os.path.isfile(source_name):
                base_name = os.path.splitext(os.path.basename(source_name))[0]
            else:
                base_name = source_name
            
            saved_paths = []
            
            for i, image in enumerate(images, 1):
                filename = f"{base_name}_page_{i:03d}.{self.output_format.lower()}"
                filepath = os.path.join(output_dir, filename)
                
                # Save image
                image.save(filepath, self.output_format)
                saved_paths.append(filepath)
                
                logger.debug(f"Saved page {i} to: {filepath}")
            
            logger.info(f"Saved {len(saved_paths)} images to: {output_dir}")
            return saved_paths
            
        except Exception as e:
            logger.error(f"Error saving images: {str(e)}")
            raise
    
    def get_pdf_page_count(self, pdf_path: str) -> int:
        """
        Get the number of pages in a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Number of pages
        """
        try:
            # Convert only first page to get count efficiently
            images = convert_from_path(pdf_path, dpi=72, last_page=1)
            
            # Use a more efficient method to get page count
            from pdf2image.pdf2image import pdfinfo_from_path
            info = pdfinfo_from_path(pdf_path)
            
            return info["Pages"]
            
        except Exception as e:
            logger.error(f"Error getting page count for {pdf_path}: {str(e)}")
            # Fallback: try to convert all pages (less efficient)
            try:
                images = convert_from_path(pdf_path, dpi=72)
                return len(images)
            except:
                raise e
    
    def convert_page_batch(
        self,
        pdf_path: str,
        batch_size: Optional[int] = None,
        output_dir: Optional[str] = None
    ) -> List[List[str]]:
        """
        Convert PDF pages in batches.
        
        Args:
            pdf_path: Path to the PDF file
            batch_size: Number of pages per batch
            output_dir: Directory to save images
            
        Returns:
            List of batches, each containing image paths or PIL objects
        """
        try:
            batch_size = batch_size or settings.PDF_BATCH_SIZE
            page_count = self.get_pdf_page_count(pdf_path)
            logger.info(f"Processing {page_count} pages in batches of {batch_size}")
            
            batches = []
            
            for start_page in range(1, page_count + 1, batch_size):
                end_page = min(start_page + batch_size - 1, page_count)
                
                logger.info(f"Processing batch: pages {start_page}-{end_page}")
                
                batch_images = self.convert_pdf_to_images(
                    pdf_path,
                    output_dir,
                    page_range=(start_page, end_page)
                )
                
                batches.append(batch_images)
            
            return batches
            
        except Exception as e:
            logger.error(f"Error in batch conversion: {str(e)}")
            raise


def process_pdf_file(pdf_path: str, output_dir: Optional[str] = None, batch_mode: bool = False) -> bool:
    """Process a single PDF file."""
    try:
        converter = PDFConverter()
        
        print(f"Processing: {pdf_path}")
        print(f"Settings: DPI={settings.PDF_DPI}, Format={settings.PDF_OUTPUT_FORMAT}")
        
        # Get page count
        page_count = converter.get_pdf_page_count(pdf_path)
        print(f"PDF has {page_count} pages")
        
        if output_dir is None:
            # Create output directory based on PDF filename
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.join(settings.OUTPUT_DIR, f"{base_name}_images")
        
        # Convert PDF
        if batch_mode and page_count > settings.PDF_BATCH_SIZE:
            print(f"Using batch mode (batch size: {settings.PDF_BATCH_SIZE})")
            batches = converter.convert_page_batch(pdf_path, output_dir=output_dir)
            total_images = sum(len(batch) for batch in batches)
            print(f"Converted {total_images} pages in {len(batches)} batches to {output_dir}")
        else:
            images = converter.convert_pdf_to_images(pdf_path, output_dir=output_dir)
            print(f"Converted {len(images)} pages to {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return False


def process_directory(input_dir: str, output_dir: Optional[str] = None, batch_mode: bool = False) -> None:
    """Process all PDF files in a directory."""
    pdf_pattern = os.path.join(input_dir, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files in {input_dir}")
    
    success_count = 0
    for pdf_file in pdf_files:
        if process_pdf_file(pdf_file, output_dir, batch_mode):
            success_count += 1
        print()  # Empty line between files
    
    print(f"Successfully processed {success_count}/{len(pdf_files)} files")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Convert PDF files to images for MathRush DataProcessor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single PDF
  python pdf_converter.py input.pdf
  
  # Convert PDF with custom output directory
  python pdf_converter.py input.pdf --output /path/to/output
  
  # Convert all PDFs in directory
  python pdf_converter.py /path/to/pdfs/
  
  # Use batch mode for large PDFs
  python pdf_converter.py input.pdf --batch
  
  # Custom DPI and format
  python pdf_converter.py input.pdf --dpi 600 --format JPEG
"""
    )
    
    parser.add_argument(
        "input",
        help="PDF file or directory containing PDF files"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output directory for images (default: auto-generated)"
    )
    
    parser.add_argument(
        "-b", "--batch",
        action="store_true",
        help="Use batch processing for large PDFs"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        help=f"Image resolution DPI (default: {settings.PDF_DPI})"
    )
    
    parser.add_argument(
        "--format",
        choices=["PNG", "JPEG", "TIFF"],
        help=f"Output image format (default: {settings.PDF_OUTPUT_FORMAT})"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help=f"Pages per batch (default: {settings.PDF_BATCH_SIZE})"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Validate input
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        return 1
    
    # Override settings if provided
    if args.dpi:
        settings.PDF_DPI = args.dpi
    if args.format:
        settings.PDF_OUTPUT_FORMAT = args.format
    if args.batch_size:
        settings.PDF_BATCH_SIZE = args.batch_size
    
    try:
        if os.path.isfile(input_path):
            if not input_path.lower().endswith('.pdf'):
                print(f"Error: Input file is not a PDF: {input_path}")
                return 1
            
            success = process_pdf_file(input_path, args.output, args.batch)
            return 0 if success else 1
            
        elif os.path.isdir(input_path):
            process_directory(input_path, args.output, args.batch)
            return 0
            
        else:
            print(f"Error: Input is neither a file nor directory: {input_path}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())