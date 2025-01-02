"""
Content classification module for e-Prahari.
Handles deep learning model inference for content analysis.
"""

from typing import Dict, Any, List, Optional
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline
)
import numpy as np
from ..preprocessing.text_processor import TextProcessor
from ..preprocessing.media_validator import MediaValidator

class ContentClassifier:
    def __init__(self):
        """
        Initialize content classifier with required models.
        """
        # Initialize BERT-based models for different tasks
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        # Model for misinformation classification
        self.misinfo_model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2
        )
        
        # Model for stance detection
        self.stance_model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=4  # neutral, agree, disagree, discuss
        )
        
        # Initialize preprocessing components
        self.text_processor = TextProcessor()
        self.media_validator = MediaValidator()
        
        # Initialize NLI pipeline for contradiction detection
        self.nli_pipeline = pipeline("zero-shot-classification")

    def analyze_content(
        self,
        content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze content to detect potential misinformation.
        
        Args:
            content: Dictionary containing content details:
                    - text: Main text content
                    - title: Content title
                    - images: List of PIL Image objects
                    - source: Source information
                    - metadata: Additional metadata
        
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Process text content
        if 'text' in content:
            text_analysis = self._analyze_text(content['text'])
            results['text_analysis'] = text_analysis
        
        # Process images
        if 'images' in content and content['images']:
            image_analysis = [
                self.media_validator.validate_image(img)
                for img in content['images']
            ]
            results['image_analysis'] = image_analysis
        
        # Analyze source credibility
        if 'source' in content:
            source_analysis = self._analyze_source(content['source'])
            results['source_analysis'] = source_analysis
        
        # Calculate overall risk scores
        results['risk_scores'] = self._calculate_risk_scores(results)
        
        return results

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform deep learning analysis on text content.
        
        Args:
            text: Input text content
            
        Returns:
            Dictionary containing text analysis results
        """
        # Preprocess text
        normalized_text = self.text_processor.normalize_text(text)
        semantic_features = self.text_processor.get_semantic_features(normalized_text)
        
        # Prepare inputs for model
        inputs = self.tokenizer(
            normalized_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Get misinformation prediction
        with torch.no_grad():
            misinfo_outputs = self.misinfo_model(**inputs)
            misinfo_probs = torch.nn.functional.softmax(misinfo_outputs.logits, dim=-1)
            misinfo_score = misinfo_probs[0][1].item()  # Probability of being misinformation
        
        # Get stance prediction
        with torch.no_grad():
            stance_outputs = self.stance_model(**inputs)
            stance_probs = torch.nn.functional.softmax(stance_outputs.logits, dim=-1)
            stance_predictions = {
                'neutral': stance_probs[0][0].item(),
                'agree': stance_probs[0][1].item(),
                'disagree': stance_probs[0][2].item(),
                'discuss': stance_probs[0][3].item()
            }
        
        # Check for contradictions using NLI
        sentences = text.split('.')
        contradiction_scores = []
        for i in range(len(sentences)-1):
            for j in range(i+1, len(sentences)):
                if sentences[i] and sentences[j]:
                    contradiction_score = self.text_processor.get_contradiction_score(
                        sentences[i],
                        sentences[j]
                    )
                    contradiction_scores.append(contradiction_score)
        
        avg_contradiction = np.mean(contradiction_scores) if contradiction_scores else 0
        
        return {
            'misinformation_probability': misinfo_score,
            'stance_analysis': stance_predictions,
            'contradiction_score': avg_contradiction,
            'semantic_features': semantic_features
        }

    def _analyze_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze source credibility based on available information.
        
        Args:
            source: Dictionary containing source information
            
        Returns:
            Dictionary containing source analysis results
        """
        # Placeholder for source credibility analysis
        # In a full implementation, this would query external databases
        # and fact-checking services
        
        return {
            'credibility_score': 0.5,  # Placeholder score
            'verification_status': 'unknown',
            'historical_accuracy': None
        }

    def _calculate_risk_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate final risk scores based on all analysis results.
        
        Args:
            results: Dictionary containing all analysis results
            
        Returns:
            Dictionary containing risk scores
        """
        risk_scores = {
            'content_risk': 0.0,
            'source_risk': 0.0,
            'overall_risk': 0.0
        }
        
        # Calculate content risk from text and image analysis
        if 'text_analysis' in results:
            text_risk = (
                results['text_analysis']['misinformation_probability'] * 0.4 +
                results['text_analysis']['contradiction_score'] * 0.3 +
                max(results['text_analysis']['stance_analysis'].values()) * 0.3
            )
            risk_scores['content_risk'] = text_risk
        
        if 'image_analysis' in results:
            image_risks = [
                1 - analysis['authenticity_score']
                for analysis in results['image_analysis']
            ]
            if image_risks:
                risk_scores['content_risk'] = (
                    risk_scores['content_risk'] * 0.7 +
                    np.mean(image_risks) * 0.3
                )
        
        # Calculate source risk
        if 'source_analysis' in results:
            risk_scores['source_risk'] = 1 - results['source_analysis']['credibility_score']
        
        # Calculate overall risk
        risk_scores['overall_risk'] = (
            risk_scores['content_risk'] * 0.7 +
            risk_scores['source_risk'] * 0.3
        )
        
        return risk_scores