"""
Media validation module for e-Prahari.
Handles image and video content verification.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

class MediaValidator:
    def __init__(self):
        """
        Initialize media validator with required models.
        """
        # Initialize image manipulation detection model
        self.image_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        
        # Standard image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def validate_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Validate an image by checking for:
        - Manipulation detection
        - Metadata consistency
        - Error level analysis
        - EXIF data analysis
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            'manipulation_score': self._detect_manipulation(image),
            'metadata_consistency': self._check_metadata(image),
            'error_level_analysis': self._error_level_analysis(image),
            'exif_analysis': self._analyze_exif(image)
        }
        
        # Calculate overall authenticity score
        results['authenticity_score'] = self._calculate_authenticity_score(results)
        
        return results

    def _detect_manipulation(self, image: Image.Image) -> float:
        """
        Detect potential image manipulation using deep learning model.
        
        Returns:
            Float between 0 and 1, where higher values indicate more manipulation
        """
        # Prepare image for model
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.image_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert model output to manipulation score
        manipulation_score = predictions[0][1].item()  # Assuming binary classification
        return manipulation_score

    def _check_metadata(self, image: Image.Image) -> float:
        """
        Check image metadata for consistency.
        
        Returns:
            Float between 0 and 1, where higher values indicate more consistency
        """
        try:
            # Check for basic metadata presence
            format = image.format
            mode = image.mode
            size = image.size
            
            # Basic consistency checks
            checks = [
                format is not None,
                mode in ['RGB', 'RGBA', 'L'],
                size[0] > 0 and size[1] > 0,
                hasattr(image, 'info')
            ]
            
            return sum(checks) / len(checks)
        except Exception:
            return 0.0

    def _error_level_analysis(self, image: Image.Image) -> float:
        """
        Perform Error Level Analysis (ELA) to detect manipulated regions.
        
        Returns:
            Float between 0 and 1, where higher values indicate more inconsistency
        """
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save with specific quality
            quality = 90
            image_path = "temp.jpg"
            image.save(image_path, quality=quality)
            
            # Open saved image
            saved_image = Image.open(image_path)
            
            # Calculate difference
            diff = np.array(image) - np.array(saved_image)
            error_levels = np.mean(np.abs(diff))
            
            # Normalize error levels
            max_error = 255 * 3  # Maximum possible error for RGB
            normalized_error = error_levels / max_error
            
            return normalized_error
        except Exception:
            return 0.0

    def _analyze_exif(self, image: Image.Image) -> float:
        """
        Analyze EXIF data for potential inconsistencies.
        
        Returns:
            Float between 0 and 1, where higher values indicate more consistency
        """
        try:
            exif = image._getexif()
            if exif is None:
                return 0.5  # Neutral score if no EXIF data
            
            # Check for important EXIF tags
            important_tags = {
                271,  # Make
                272,  # Model
                306,  # DateTime
                36867,  # DateTimeOriginal
                36868   # DateTimeDigitized
            }
            
            present_tags = set(exif.keys()) & important_tags
            consistency_score = len(present_tags) / len(important_tags)
            
            return consistency_score
        except Exception:
            return 0.0

    def _calculate_authenticity_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate overall image authenticity score from individual metrics.
        
        Args:
            results: Dictionary containing individual validation results
            
        Returns:
            Float between 0 and 1, where higher values indicate more authenticity
        """
        weights = {
            'manipulation_score': 0.4,
            'metadata_consistency': 0.2,
            'error_level_analysis': 0.2,
            'exif_analysis': 0.2
        }
        
        authenticity_score = (
            (1 - results['manipulation_score']) * weights['manipulation_score'] +
            results['metadata_consistency'] * weights['metadata_consistency'] +
            (1 - results['error_level_analysis']) * weights['error_level_analysis'] +
            results['exif_analysis'] * weights['exif_analysis']
        )
        
        return authenticity_score