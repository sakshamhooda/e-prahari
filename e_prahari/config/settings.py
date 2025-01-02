"""
Configuration settings for e-Prahari.
"""

from typing import Dict, Any
from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "e-Prahari"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Model Settings
    MODEL_CACHE_DIR: str = str(Path(__file__).parent.parent / "models")
    
    # Scoring Weights
    SCORING_WEIGHTS: Dict[str, float] = {
        "source_credibility": 0.3,
        "content_consistency": 0.3,
        "engagement_patterns": 0.2,
        "author_behavior": 0.2
    }
    
    # Processing Settings
    MAX_CONTENT_LENGTH: int = 1000000  # Maximum content length in bytes
    SUPPORTED_LANGUAGES: list = ["en"]  # Supported languages
    SUPPORTED_IMAGE_FORMATS: list = ["jpg", "jpeg", "png"]
    
    # External Services
    FACT_CHECK_API_URL: str = "https://factcheck-api.example.com"
    FACT_CHECK_API_KEY: str = ""
    
    # Cache Settings
    CACHE_TTL: int = 3600  # Cache time to live in seconds
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Performance
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 32
    
    # Thresholds
    CREDIBILITY_THRESHOLD: float = 0.5  # Threshold for credibility scoring
    RISK_THRESHOLD: float = 0.7  # Threshold for risk assessment
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
