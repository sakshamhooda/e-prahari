"""
Unit tests for the scoring system.
"""

import unittest
import numpy as np
from e_prahari.analysis.scoring import (
    SourceCredibility,
    ContentConsistency,
    EngagementPatterns,
    AuthorBehavior,
    CredibilityScorer
)

class TestScoring(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.scorer = CredibilityScorer()
        
        # Create test components
        self.source_cred = SourceCredibility(
            historical_accuracy=0.8,
            domain_affiliation=0.7,
            fact_checker_signals=0.9
        )
        
        self.content_consist = ContentConsistency(
            contradiction_index=0.2,
            image_authenticity=0.85,
            semantic_alignment=0.75
        )
        
        self.engagement = EngagementPatterns(
            share_velocity=0.3,
            comment_uniformity=0.4,
            natural_time_score=0.8
        )
        
        self.author = AuthorBehavior(
            misinformation_history=0.1,
            account_flags=0.2,
            profile_authenticity=0.9
        )

    def test_source_credibility(self):
        """Test source credibility calculation"""
        score = self.source_cred.calculate()
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        
        # Test with custom weights
        self.source_cred.weights = {
            'historical_accuracy': 0.5,
            'domain_affiliation': 0.3,
            'fact_checker_signals': 0.2
        }
        score = self.source_cred.calculate()
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_content_consistency(self):
        """Test content consistency calculation"""
        score = self.content_consist.calculate()
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_engagement_patterns(self):
        """Test engagement patterns calculation"""
        score = self.engagement.calculate()
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_author_behavior(self):
        """Test author behavior calculation"""
        score = self.author.calculate()
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_final_score(self):
        """Test final credibility score calculation"""
        score = self.scorer.calculate_score(
            self.source_cred,
            self.content_consist,
            self.engagement,
            self.author
        )
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_invalid_weights(self):
        """Test handling of invalid weights"""
        with self.assertRaises(ValueError):
            CredibilityScorer(component_weights={
                'source_credibility': 0.5,
                'content_consistency': 0.5,
                'engagement_patterns': 0.5,
                'author_behavior': 0.5
            })

    def test_misinformation_detection(self):
        """Test misinformation detection thresholds"""
        # Test high credibility case
        score = 0.8
        self.assertFalse(self.scorer.is_misinformation(score))
        
        # Test low credibility case
        score = 0.3
        self.assertTrue(self.scorer.is_misinformation(score))
        
        # Test threshold edge case
        score = 0.5
        self.assertFalse(self.scorer.is_misinformation(score))

if __name__ == '__main__':
    unittest.main()