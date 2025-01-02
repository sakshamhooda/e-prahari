"""
Implementation of the e-Prahari scoring system as described in Section 4 of the paper.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class SourceCredibility:
    historical_accuracy: float
    domain_affiliation: float
    fact_checker_signals: float
    weights: Dict[str, float] = None

    def calculate(self) -> float:
        if self.weights is None:
            self.weights = {
                'historical_accuracy': 0.4,
                'domain_affiliation': 0.3,
                'fact_checker_signals': 0.3
            }
        
        return (
            self.weights['historical_accuracy'] * self.historical_accuracy +
            self.weights['domain_affiliation'] * self.domain_affiliation +
            self.weights['fact_checker_signals'] * self.fact_checker_signals
        )


@dataclass
class ContentConsistency:
    contradiction_index: float
    image_authenticity: float
    semantic_alignment: float
    weights: Dict[str, float] = None

    def calculate(self) -> float:
        if self.weights is None:
            self.weights = {
                'contradiction_index': 0.4,
                'image_authenticity': 0.3,
                'semantic_alignment': 0.3
            }
        
        return (
            self.weights['contradiction_index'] * (1 - self.contradiction_index) +
            self.weights['image_authenticity'] * self.image_authenticity +
            self.weights['semantic_alignment'] * self.semantic_alignment
        )


@dataclass
class EngagementPatterns:
    share_velocity: float
    comment_uniformity: float
    natural_time_score: float
    weights: Dict[str, float] = None

    def calculate(self) -> float:
        if self.weights is None:
            self.weights = {
                'share_velocity': 0.4,
                'comment_uniformity': 0.3,
                'natural_time_score': 0.3
            }
        
        return (
            self.weights['share_velocity'] * (1 - self.share_velocity) +
            self.weights['comment_uniformity'] * (1 - self.comment_uniformity) +
            self.weights['natural_time_score'] * self.natural_time_score
        )


@dataclass
class AuthorBehavior:
    misinformation_history: float
    account_flags: float
    profile_authenticity: float
    weights: Dict[str, float] = None

    def calculate(self) -> float:
        if self.weights is None:
            self.weights = {
                'misinformation_history': 0.4,
                'account_flags': 0.3,
                'profile_authenticity': 0.3
            }
        
        return (
            self.weights['misinformation_history'] * (1 - self.misinformation_history) +
            self.weights['account_flags'] * (1 - self.account_flags) +
            self.weights['profile_authenticity'] * self.profile_authenticity
        )


class CredibilityScorer:
    def __init__(
        self,
        component_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the credibility scorer with optional component weights.
        
        Args:
            component_weights: Dictionary with weights for each scoring component.
                            If None, default weights will be used.
        """
        self.component_weights = component_weights or {
            'source_credibility': 0.3,
            'content_consistency': 0.3,
            'engagement_patterns': 0.2,
            'author_behavior': 0.2
        }
        
        # Validate weights sum to 1
        if not np.isclose(sum(self.component_weights.values()), 1.0):
            raise ValueError("Component weights must sum to 1")

    def calculate_score(
        self,
        source_credibility: SourceCredibility,
        content_consistency: ContentConsistency,
        engagement_patterns: EngagementPatterns,
        author_behavior: AuthorBehavior
    ) -> float:
        """
        Calculate the final credibility score using the formula from the paper.
        
        Score = α·CS + β·CC + γ·EP + δ·AB
        
        Returns:
            float: Final credibility score between 0 and 1
        """
        cs_score = source_credibility.calculate()
        cc_score = content_consistency.calculate()
        ep_score = engagement_patterns.calculate()
        ab_score = author_behavior.calculate()
        
        final_score = (
            self.component_weights['source_credibility'] * cs_score +
            self.component_weights['content_consistency'] * cc_score +
            self.component_weights['engagement_patterns'] * ep_score +
            self.component_weights['author_behavior'] * ab_score
        )
        
        return final_score

    def is_misinformation(self, score: float, threshold: float = 0.5) -> bool:
        """
        Determine if content is likely misinformation based on the credibility score.
        
        Args:
            score: Calculated credibility score
            threshold: Score threshold below which content is considered misinformation
            
        Returns:
            bool: True if the content is likely misinformation, False otherwise
        """
        return score < threshold