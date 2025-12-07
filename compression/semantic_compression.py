"""
Semantic Compression Module

Implements more aggressive compression through:
- Extractive summarization
- Sentence fusion (combining similar sentences)
- Redundancy removal
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class SemanticCompressor:
    """
    Semantic-aware compression that goes beyond simple sentence selection.
    
    Uses sentence similarity and information overlap to produce
    more compressed outputs while preserving meaning.
    """
    
    def __init__(self, similarity_threshold: float = 0.6):
        """
        Initialize semantic compressor.
        
        Args:
            similarity_threshold: Threshold for considering sentences similar
        """
        self.similarity_threshold = similarity_threshold
    
    def compress_semantically(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        target_ratio: float = 0.5
    ) -> str:
        """
        Compress text using semantic understanding.
        
        Args:
            text: Input text to compress
            metadata: Optional metadata (question, etc.)
            target_ratio: Target compression ratio
            
        Returns:
            Compressed text
        """
        if metadata is None:
            metadata = {}
        
        question = metadata.get('question', '')
        
        # Step 1: Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= 2:
            return text
        
        # Step 2: Remove redundant sentences
        unique_sentences = self._remove_redundancy(sentences)
        
        # Step 3: Score sentences by relevance
        scored_sentences = self._score_sentences(unique_sentences, question)
        
        # Step 4: Select top sentences
        keep_count = max(2, int(len(scored_sentences) * target_ratio))
        top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:keep_count]
        
        # Step 5: Sort by original order and join
        top_sentences_ordered = sorted(top_sentences, key=lambda x: x[2])  # Sort by original index
        result = ' '.join([s[0] for s in top_sentences_ordered])
        
        # Step 6: Apply sentence fusion if beneficial
        result = self._fuse_sentences(result)
        
        logger.info(
            f"Semantic compression: {len(sentences)} -> {len(self._split_sentences(result))} sentences"
        )
        
        return result
    
    def _remove_redundancy(self, sentences: List[str]) -> List[str]:
        """
        Remove redundant/duplicate sentences.
        
        Args:
            sentences: List of sentences
            
        Returns:
            List with redundancy removed
        """
        unique = []
        seen_content = set()
        
        for sent in sentences:
            # Normalize for comparison
            normalized = self._normalize_for_comparison(sent)
            
            # Check for high overlap with seen content
            is_redundant = False
            for seen in seen_content:
                similarity = self._compute_word_overlap(normalized, seen)
                if similarity > self.similarity_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                unique.append(sent)
                seen_content.add(normalized)
        
        return unique
    
    def _score_sentences(
        self,
        sentences: List[str],
        question: str
    ) -> List[Tuple[str, float, int]]:
        """
        Score sentences by relevance to question.
        
        Args:
            sentences: List of sentences
            question: The question
            
        Returns:
            List of (sentence, score, original_index)
        """
        keywords = self._extract_keywords(question)
        scored = []
        
        for idx, sent in enumerate(sentences):
            score = 0.0
            sent_lower = sent.lower()
            
            # Keyword match score
            for kw in keywords:
                if kw.lower() in sent_lower:
                    score += 1.0
            
            # Entity presence score
            entities = len(re.findall(r'\b[A-Z][a-z]+', sent))
            numbers = len(re.findall(r'\b\d+', sent))
            score += entities * 0.3 + numbers * 0.5
            
            # Position score (first and last sentences often important)
            if idx == 0:
                score += 0.5
            elif idx == len(sentences) - 1:
                score += 0.3
            
            # Length score (moderate length preferred)
            word_count = len(sent.split())
            if 10 <= word_count <= 30:
                score += 0.2
            
            scored.append((sent, score, idx))
        
        return scored
    
    def _fuse_sentences(self, text: str) -> str:
        """
        Fuse related sentences to reduce length.
        
        Args:
            text: Input text
            
        Returns:
            Text with fused sentences
        """
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return text
        
        fused = []
        i = 0
        
        while i < len(sentences):
            current = sentences[i]
            
            # Check if next sentence can be fused
            if i + 1 < len(sentences):
                next_sent = sentences[i + 1]
                
                # Check for fusion patterns
                fused_result = self._try_fuse_pair(current, next_sent)
                if fused_result:
                    fused.append(fused_result)
                    i += 2
                    continue
            
            fused.append(current)
            i += 1
        
        return ' '.join(fused)
    
    def _try_fuse_pair(self, sent1: str, sent2: str) -> Optional[str]:
        """
        Try to fuse two sentences.
        
        Args:
            sent1: First sentence
            sent2: Second sentence
            
        Returns:
            Fused sentence or None if fusion not beneficial
        """
        # Pattern 1: Second sentence starts with pronoun referring to first
        pronoun_starts = ['it ', 'he ', 'she ', 'they ', 'this ', 'that ', 'these ']
        
        for pronoun in pronoun_starts:
            if sent2.lower().startswith(pronoun):
                # Simple fusion by connecting with comma or conjunction
                # Remove the pronoun and combine
                rest = sent2[len(pronoun):].strip()
                if rest:
                    # Check if this creates a valid sentence
                    if len(sent1.split()) + len(rest.split()) < 40:  # Not too long
                        return f"{sent1.rstrip('.')}, {rest}"
        
        # Pattern 2: Both sentences about same entity
        entities1 = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', sent1))
        entities2 = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', sent2))
        
        common_entities = entities1 & entities2
        if common_entities and len(sent1.split()) + len(sent2.split()) < 35:
            # Can potentially combine information about same entity
            return f"{sent1.rstrip('.')}, and {sent2[0].lower()}{sent2[1:]}"
        
        return None
    
    def _compute_word_overlap(self, text1: str, text2: str) -> float:
        """Compute word overlap between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _normalize_for_comparison(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase, remove punctuation, extra whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    @staticmethod
    def _extract_keywords(question: str) -> List[str]:
        """Extract keywords from question."""
        stopwords = {
            'what', 'when', 'where', 'who', 'why', 'how',
            'is', 'are', 'was', 'were', 'the', 'a', 'an',
            'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'
        }
        words = question.lower().split()
        return [w.strip('?.,!') for w in words 
                if w.lower() not in stopwords and len(w) > 2]
    
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]


def compress_with_semantics(
    text: str,
    question: str = "",
    target_ratio: float = 0.5
) -> str:
    """
    Convenience function for semantic compression.
    
    Args:
        text: Text to compress
        question: Optional question for relevance scoring
        target_ratio: Target compression ratio
        
    Returns:
        Compressed text
    """
    compressor = SemanticCompressor()
    return compressor.compress_semantically(
        text,
        metadata={'question': question},
        target_ratio=target_ratio
    )
