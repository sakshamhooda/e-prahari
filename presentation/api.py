"""
FastAPI-based API interface for e-Prahari.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from PIL import Image
import io
import json

from ..processing.content_classifier import ContentClassifier
from ..analysis.scoring import CredibilityScorer, SourceCredibility, ContentConsistency, EngagementPatterns, AuthorBehavior

app = FastAPI(
    title="e-Prahari API",
    description="Digital Sentinel Framework for Misinformation Detection",
    version="0.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
content_classifier = ContentClassifier()
credibility_scorer = CredibilityScorer()

class ContentRequest(BaseModel):
    """Request model for content analysis."""
    text: Optional[str] = None
    title: Optional[str] = None
    source: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    """Response model for analysis results."""
    content_analysis: Dict[str, Any]
    credibility_score: float
    risk_assessment: Dict[str, float]
    recommendations: List[str]

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(
    content: ContentRequest,
    files: Optional[List[UploadFile]] = None
):
    """
    Analyze content for potential misinformation.
    
    Args:
        content: ContentRequest object containing text and metadata
        files: Optional list of image files
        
    Returns:
        AnalysisResponse containing analysis results
    """
    try:
        # Prepare content dictionary
        content_dict = content.dict()
        
        # Process uploaded images if any
        if files:
            images = []
            for file in files:
                content_type = file.content_type
                if not content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=400,
                        detail=f"File {file.filename} is not an image"
                    )
                
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                images.append(image)
            
            content_dict['images'] = images
        
        # Perform content analysis
        analysis_results = content_classifier.analyze_content(content_dict)
        
        # Calculate credibility scores
        source_cred = SourceCredibility(
            historical_accuracy=analysis_results['source_analysis'].get('historical_accuracy', 0.5),
            domain_affiliation=0.5,  # Placeholder
            fact_checker_signals=0.5  # Placeholder
        )
        
        content_consistency = ContentConsistency(
            contradiction_index=analysis_results['text_analysis']['contradiction_score'],
            image_authenticity=np.mean([img['authenticity_score'] for img in analysis_results['image_analysis']]) if 'image_analysis' in analysis_results else 0.5,
            semantic_alignment=0.5  # Placeholder
        )
        
        engagement_patterns = EngagementPatterns(
            share_velocity=0.5,  # Placeholder
            comment_uniformity=0.5,  # Placeholder
            natural_time_score=0.5  # Placeholder
        )
        
        author_behavior = AuthorBehavior(
            misinformation_history=0.5,  # Placeholder
            account_flags=0.5,  # Placeholder
            profile_authenticity=0.5  # Placeholder
        )
        
        credibility_score = credibility_scorer.calculate_score(
            source_cred,
            content_consistency,
            engagement_patterns,
            author_behavior
        )
        
        # Generate recommendations
        recommendations = _generate_recommendations(
            analysis_results,
            credibility_score
        )
        
        return AnalysisResponse(
            content_analysis=analysis_results,
            credibility_score=credibility_score,
            risk_assessment=analysis_results['risk_scores'],
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

def _generate_recommendations(
    analysis_results: Dict[str, Any],
    credibility_score: float
) -> List[str]:
    """
    Generate recommendations based on analysis results.
    
    Args:
        analysis_results: Dictionary containing analysis results
        credibility_score: Overall credibility score
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Add recommendations based on credibility score
    if credibility_score < 0.3:
        recommendations.append(
            "High risk of misinformation. Verify with trusted sources before sharing."
        )
    elif credibility_score < 0.6:
        recommendations.append(
            "Medium risk. Additional fact-checking recommended."
        )
    
    # Add specific recommendations based on analysis results
    if analysis_results['text_analysis']['contradiction_score'] > 0.5:
        recommendations.append(
            "Internal contradictions detected. Review content carefully."
        )
    
    if 'image_analysis' in analysis_results:
        low_auth_images = [
            i for i, img in enumerate(analysis_results['image_analysis'])
            if img['authenticity_score'] < 0.5
        ]
        if low_auth_images:
            recommendations.append(
                f"Potential manipulation detected in image(s) {low_auth_images}. "
                "Verify image authenticity."
            )
    
    return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)