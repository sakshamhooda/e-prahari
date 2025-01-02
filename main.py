"""
Main entry point for e-Prahari application.
"""

import logging
from pathlib import Path
import click
import uvicorn
from e_prahari.config.settings import settings
from e_prahari.presentation.api import app

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=settings.LOG_FORMAT
)

logger = logging.getLogger(__name__)

def init_models():
    """Initialize and cache required models."""
    try:
        # Create model cache directory if it doesn't exist
        Path(settings.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        logger.info("Model cache directory initialized")
        
        # Download and cache required models
        from e_prahari.processing.content_classifier import ContentClassifier
        classifier = ContentClassifier()
        logger.info("Models initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise

@click.group()
def cli():
    """e-Prahari: Digital Sentinel Framework for Misinformation Detection"""
    pass

@cli.command()
@click.option(
    '--host',
    default="0.0.0.0",
    help="Host to bind the server to"
)
@click.option(
    '--port',
    default=8000,
    help="Port to bind the server to"
)
@click.option(
    '--reload',
    is_flag=True,
    default=False,
    help="Enable auto-reload for development"
)
def serve(host: str, port: int, reload: bool):
    """Start the e-Prahari API server."""
    try:
        logger.info("Initializing e-Prahari server...")
        init_models()
        
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(
            "e_prahari.presentation.api:app",
            host=host,
            port=port,
            reload=reload,
            workers=settings.MAX_WORKERS
        )
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        raise

@cli.command()
def init():
    """Initialize e-Prahari system."""
    try:
        logger.info("Initializing e-Prahari system...")
        
        # Create necessary directories
        Path(settings.MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        init_models()
        
        logger.info("System initialized successfully")
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise

@cli.command()
@click.argument('content_path')
def analyze(content_path: str):
    """
    Analyze content from a file for misinformation.
    
    Args:
        content_path: Path to the content file
    """
    try:
        from e_prahari.processing.content_classifier import ContentClassifier
        classifier = ContentClassifier()
        
        # Read content
        with open(content_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Analyze content
        results = classifier.analyze_content({'text': content})
        
        # Print results
        click.echo("\nAnalysis Results:")
        click.echo("-----------------")
        click.echo(f"Risk Score: {results['risk_scores']['overall_risk']:.2f}")
        click.echo(f"Content Risk: {results['risk_scores']['content_risk']:.2f}")
        click.echo(f"Source Risk: {results['risk_scores']['source_risk']:.2f}")
        
        if results['risk_scores']['overall_risk'] > settings.RISK_THRESHOLD:
            click.echo("\n⚠️  High risk of misinformation detected!")
        else:
            click.echo("\n✅ Content appears to be legitimate.")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    cli()