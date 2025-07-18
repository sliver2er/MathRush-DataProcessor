"""
Configuration settings for MathRush DataProcessor.
Centralized configuration management using environment variables and defaults.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings with environment variable support."""
    
    # PDF Processing Settings
    PDF_DPI: int = int(os.getenv('PDF_DPI', '300'))
    PDF_OUTPUT_FORMAT: str = os.getenv('PDF_OUTPUT_FORMAT', 'PNG')
    PDF_BATCH_SIZE: int = int(os.getenv('PDF_BATCH_SIZE', '5'))
    
    # Directory Settings
    OUTPUT_DIR: str = os.getenv('OUTPUT_DIR', 'output')
    SAMPLES_DIR: str = os.getenv('SAMPLES_DIR', 'samples')
    TEMP_DIR: str = os.getenv('TEMP_DIR', 'temp')
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    OPENAI_MAX_TOKENS: int = int(os.getenv('OPENAI_MAX_TOKENS', '2000'))
    OPENAI_TEMPERATURE: float = float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
    
    # Rate limiting settings for OpenAI
    OPENAI_REQUESTS_PER_MINUTE: int = int(os.getenv('OPENAI_REQUESTS_PER_MINUTE', '50'))
    OPENAI_RETRY_ATTEMPTS: int = int(os.getenv('OPENAI_RETRY_ATTEMPTS', '5'))
    OPENAI_RETRY_BASE_DELAY: float = float(os.getenv('OPENAI_RETRY_BASE_DELAY', '2.0'))
    OPENAI_RETRY_MAX_DELAY: float = float(os.getenv('OPENAI_RETRY_MAX_DELAY', '60.0'))

    # Gemini Settings
    GOOGLE_API_KEY: str = os.getenv('GOOGLE_API_KEY', '')
    GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-2.5-pro')
    GEMINI_MAX_TOKENS: int = int(os.getenv('GEMINI_MAX_TOKENS', '4000'))
    GEMINI_TEMPERATURE: float = float(os.getenv('GEMINI_TEMPERATURE', '0.1'))

    # Rate limiting settings for Gemini
    GEMINI_REQUESTS_PER_MINUTE: int = int(os.getenv('GEMINI_REQUESTS_PER_MINUTE', '15')) # Gemini models may have specific RPM limits
    GEMINI_RETRY_ATTEMPTS: int = int(os.getenv('GEMINI_RETRY_ATTEMPTS', '5'))
    GEMINI_RETRY_BASE_DELAY: float = float(os.getenv('GEMINI_RETRY_BASE_DELAY', '5.0'))
    GEMINI_RETRY_MAX_DELAY: float = float(os.getenv('GEMINI_RETRY_MAX_DELAY', '120.0'))
    
    # Supabase Settings
    SUPABASE_URL: str = os.getenv('SUPABASE_URL', '')
    SUPABASE_KEY: str = os.getenv('SUPABASE_KEY', '')
    SUPABASE_TABLE: str = os.getenv('SUPABASE_TABLE', 'problems')
    
    # Processing Settings
    MAX_CONCURRENT_REQUESTS: int = int(os.getenv('MAX_CONCURRENT_REQUESTS', '5'))
    RETRY_ATTEMPTS: int = int(os.getenv('RETRY_ATTEMPTS', '3'))
    RETRY_DELAY: float = float(os.getenv('RETRY_DELAY', '1.0'))
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', 'processor.log')
    
    # Data Validation Settings
    MIN_PROBLEM_LENGTH: int = int(os.getenv('MIN_PROBLEM_LENGTH', '10'))
    MAX_PROBLEM_LENGTH: int = int(os.getenv('MAX_PROBLEM_LENGTH', '5000'))
    REQUIRED_FIELDS: list = ['content', 'problem_type', 'correct_answer', 'exam_name', 'problem_number']
    
    # Correct Rate Validation (for future difficulty determination)
    MIN_CORRECT_RATE: float = float(os.getenv('MIN_CORRECT_RATE', '0.0'))    # 0% correct rate
    MAX_CORRECT_RATE: float = float(os.getenv('MAX_CORRECT_RATE', '100.0'))  # 100% correct rate
    
    @classmethod
    def get_pdf_config(cls) -> Dict[str, Any]:
        """Get PDF processing configuration."""
        return {
            'dpi': cls.PDF_DPI,
            'output_format': cls.PDF_OUTPUT_FORMAT,
            'batch_size': cls.PDF_BATCH_SIZE
        }
    
    @classmethod
    def get_openai_config(cls) -> Dict[str, Any]:
        """Get OpenAI API configuration."""
        return {
            'api_key': cls.OPENAI_API_KEY,
            'model': cls.OPENAI_MODEL,
            'max_tokens': cls.OPENAI_MAX_TOKENS,
            'temperature': cls.OPENAI_TEMPERATURE,
            'requests_per_minute': cls.OPENAI_REQUESTS_PER_MINUTE,
            'retry_attempts': cls.OPENAI_RETRY_ATTEMPTS,
            'retry_base_delay': cls.OPENAI_RETRY_BASE_DELAY,
            'retry_max_delay': cls.OPENAI_RETRY_MAX_DELAY
        }

    @classmethod
    def get_gemini_config(cls) -> Dict[str, Any]:
        """Get Gemini API configuration."""
        return {
            'api_key': cls.GOOGLE_API_KEY,
            'model': cls.GEMINI_MODEL,
            'max_tokens': cls.GEMINI_MAX_TOKENS,
            'temperature': cls.GEMINI_TEMPERATURE,
            'requests_per_minute': cls.GEMINI_REQUESTS_PER_MINUTE,
            'retry_attempts': cls.GEMINI_RETRY_ATTEMPTS,
            'retry_base_delay': cls.GEMINI_RETRY_BASE_DELAY,
            'retry_max_delay': cls.GEMINI_RETRY_MAX_DELAY
        }
    
    @classmethod
    def get_supabase_config(cls) -> Dict[str, Any]:
        """Get Supabase configuration."""
        return {
            'url': cls.SUPABASE_URL,
            'key': cls.SUPABASE_KEY,
            'table': cls.SUPABASE_TABLE
        }
    
    @classmethod
    def get_directories(cls) -> Dict[str, str]:
        """Get directory paths."""
        return {
            'output': cls.OUTPUT_DIR,
            'samples': cls.SAMPLES_DIR,
            'temp': cls.TEMP_DIR
        }
    
    @classmethod
    def validate_required_settings(cls) -> bool:
        """Validate that required settings are configured."""
        missing = []
        
        if not cls.GOOGLE_API_KEY and not cls.OPENAI_API_KEY:
            missing.append('GOOGLE_API_KEY or OPENAI_API_KEY')
        
        if not cls.SUPABASE_URL:
            missing.append('SUPABASE_URL')
            
        if not cls.SUPABASE_KEY:
            missing.append('SUPABASE_KEY')
        
        if missing:
            print(f"Missing required environment variables: {', '.join(missing)}")
            return False
        
        return True
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        directories = [cls.OUTPUT_DIR, cls.SAMPLES_DIR, cls.TEMP_DIR]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_all_settings(cls) -> Dict[str, Any]:
        """Get all settings as a dictionary."""
        return {
            'pdf': cls.get_pdf_config(),
            'openai': cls.get_openai_config(),
            'gemini': cls.get_gemini_config(),
            'supabase': cls.get_supabase_config(),
            'directories': cls.get_directories(),
            'processing': {
                'max_concurrent_requests': cls.MAX_CONCURRENT_REQUESTS,
                'retry_attempts': cls.RETRY_ATTEMPTS,
                'retry_delay': cls.RETRY_DELAY
            },
            'logging': {
                'level': cls.LOG_LEVEL,
                'file': cls.LOG_FILE
            },
            'validation': {
                'min_problem_length': cls.MIN_PROBLEM_LENGTH,
                'max_problem_length': cls.MAX_PROBLEM_LENGTH,
                'required_fields': cls.REQUIRED_FIELDS
            }
        }


# Create global settings instance
settings = Settings()

# Auto-create directories on import
settings.create_directories()


def print_settings():
    """Print current settings for debugging."""
    print("=== MathRush DataProcessor Settings ===")
    all_settings = settings.get_all_settings()
    
    for category, config in all_settings.items():
        print(f"\n{category.upper()}:")
        for key, value in config.items():
            # Hide sensitive information
            if 'key' in key.lower() or 'token' in key.lower():
                value = '*' * len(str(value)) if value else 'Not set'
            print(f"  {key}: {value}")


if __name__ == "__main__":
    print_settings()
    print(f"\nValidation: {'✓ PASS' if settings.validate_required_settings() else '✗ FAIL'}")