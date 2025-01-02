"""
Text preprocessing module for e-Prahari.
"""

import re
from typing import List, Optional, Dict, Any
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from collections import Counter

class TextProcessor:
    def __init__(self, language: str = "en"):
        """
        Initialize the text processor.
        
        Args:
            language: Language code (default: "en" for English)
        """
        self.language = language
        try:
            self.nlp = spacy.load(f"{language}_core_web_sm")
        except OSError:
            spacy.cli.download(f"{language}_core_web_sm")
            self.nlp = spacy.load(f"{language}_core_web_sm")
        
        # Initialize BERT tokenizer and model for semantic analysis
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by removing extra whitespace, converting to lowercase,
        and standardizing punctuation.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Standardize punctuation
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r'[''']', "'", text)
        text = re.sub(r'…', '...', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            List of dictionaries containing entity text, label, and position
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities

    def get_semantic_features(self, text: str) -> Dict[str, Any]:
        """
        Extract semantic features from text including:
        - Sentence count
        - Word count
        - Average word length
        - Part of speech distribution
        - Named entity counts
        - Text complexity metrics
        - Sentiment scores
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing extracted features
        """
        doc = self.nlp(text)
        
        # Basic counts
        sentences = list(doc.sents)
        words = [token.text for token in doc if not token.is_punct and not token.is_space]
        
        # POS distribution
        pos_counts = Counter([token.pos_ for token in doc])
        
        # Entity type distribution
        entity_counts = Counter([ent.label_ for ent in doc.ents])
        
        # Calculate text complexity metrics
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Get BERT embeddings for semantic richness
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        features = {
            'sentence_count': len(sentences),
            'word_count': len(words),
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'pos_distribution': dict(pos_counts),
            'entity_distribution': dict(entity_counts),
            'semantic_embedding': embeddings,
            'unique_words_ratio': len(set(words)) / len(words) if words else 0,
            'sentiment': self._get_sentiment(doc),
            'complexity_score': self._calculate_complexity(doc)
        }
        
        return features
    
    def _get_sentiment(self, doc: spacy.tokens.Doc) -> Dict[str, float]:
        """
        Calculate sentiment scores for the document.
        """
        return {
            'positive': sum(1 for token in doc if token.sentiment > 0) / len(doc),
            'negative': sum(1 for token in doc if token.sentiment < 0) / len(doc),
            'neutral': sum(1 for token in doc if token.sentiment == 0) / len(doc)
        }
    
    def _calculate_complexity(self, doc: spacy.tokens.Doc) -> float:
        """
        Calculate text complexity score based on various metrics.
        """
        features = []
        
        # Vocabulary diversity
        words = [token.text.lower() for token in doc if not token.is_punct and not token.is_space]
        vocab_diversity = len(set(words)) / len(words) if words else 0
        features.append(vocab_diversity)
        
        # Syntactic complexity (dependency tree depth)
        depths = [len(list(token.ancestors)) for token in doc]
        avg_depth = np.mean(depths) if depths else 0
        features.append(min(avg_depth / 5, 1))  # Normalize to [0,1]
        
        # Named entity density
        entity_density = len(doc.ents) / len(doc) if len(doc) > 0 else 0
        features.append(entity_density)
        
        return np.mean(features)

    def get_contradiction_score(self, text1: str, text2: str) -> float:
        """
        Calculate a contradiction score between two pieces of text.
        Uses BERT embeddings to compute semantic similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Float between 0 and 1, where higher values indicate more contradiction
        """
        # Get BERT embeddings
        inputs1 = self.tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
        inputs2 = self.tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            embeddings1 = self.model(**inputs1).last_hidden_state.mean(dim=1)
            embeddings2 = self.model(**inputs2).last_hidden_state.mean(dim=1)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        
        # Convert similarity to contradiction score (1 - similarity)
        contradiction_score = 1 - similarity.item()
        
        return contradiction_score