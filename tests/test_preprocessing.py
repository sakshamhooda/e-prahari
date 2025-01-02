"""
Unit tests for text and media preprocessing.
"""

import unittest
from PIL import Image
import io
import numpy as np
from e_prahari.preprocessing.text_processor import TextProcessor
from e_prahari.preprocessing.media_validator import MediaValidator

class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.processor = TextProcessor()
        self.test_text = """
        This is a test article! It contains some entities like Microsoft and Google.
        It also has some URLs like https://example.com and formatting issues...
        """

    def test_text_normalization(self):
        """Test text normalization"""
        normalized = self.processor.normalize_text(self.test_text)
        
        # Check whitespace normalization
        self.assertFalse('\n' in normalized)
        self.assertFalse('  ' in normalized)
        
        # Check URL removal
        self.assertFalse('https://' in normalized)
        
        # Check punctuation standardization
        self.assertFalse('...' in normalized)

    def test_entity_extraction(self):
        """Test named entity extraction"""
        entities = self.processor.extract_entities(self.test_text)
        
        # Check if entities were found
        self.assertGreater(len(entities), 0)
        
        # Check entity structure
        for entity in entities:
            self.assertIn('text', entity)
            self.assertIn('label', entity)
            self.assertIn('start', entity)
            self.assertIn('end', entity)
        
        # Check if common entities were found
        entity_texts = [e['text'] for e in entities]
        self.assertTrue(any('Microsoft' in text or 'Google' in text 
                          for text in entity_texts))

    def test_semantic_features(self):
        """Test semantic feature extraction"""
        features = self.processor.get_semantic_features(self.test_text)
        
        # Check feature presence
        self.assertIn('sentence_count', features)
        self.assertIn('word_count', features)
        self.assertIn('avg_word_length', features)
        self.assertIn('semantic_embedding', features)
        
        # Check feature values
        self.assertGreater(features['sentence_count'], 0)
        self.assertGreater(features['word_count'], 0)
        self.assertGreater(features['avg_word_length'], 0)
        
        # Check embedding shape
        self.assertTrue(isinstance(features['semantic_embedding'], np.ndarray))

    def test_contradiction_detection(self):
        """Test contradiction detection between texts"""
        text1 = "The sky is blue."
        text2 = "The sky is red."
        
        score = self.processor.get_contradiction_score(text1, text2)
        
        # Check score bounds
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        
        # Check if contradictory statements get high scores
        self.assertGreater(score, 0.5)

class TestMediaValidator(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.validator = MediaValidator()
        
        # Create a test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        
        # Create a manipulated version
        self.manipulated_image = self.test_image.copy()
        pixels = self.manipulated_image.load()
        for i in range(50):
            for j in range(50):
                pixels[i, j] = (0, 255, 0)

    def test_image_validation(self):
        """Test image validation"""
        results = self.validator.validate_image(self.test_image)
        
        # Check result structure
        self.assertIn('manipulation_score', results)
        self.assertIn('metadata_consistency', results)
        self.assertIn('error_level_analysis', results)
        self.assertIn('authenticity_score', results)
        
        # Check score bounds
        for score in results.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_manipulation_detection(self):
        """Test manipulation detection"""
        # Test original image
        score1 = self.validator._detect_manipulation(self.test_image)
        
        # Test manipulated image
        score2 = self.validator._detect_manipulation(self.manipulated_image)
        
        # Manipulated image should have higher manipulation score
        self.assertGreater(score2, score1)

    def test_metadata_checking(self):
        """Test metadata consistency checking"""
        score = self.validator._check_metadata(self.test_image)
        
        # Check score bounds
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_error_level_analysis(self):
        """Test error level analysis"""
        score = self.validator._error_level_analysis(self.test_image)
        
        # Check score bounds
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

if __name__ == '__main__':
    unittest.main()