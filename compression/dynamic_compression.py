"""
Dynamic Compression Ratio Selection

Automatically adjusts compression ratio based on:
- Question complexity (simple factoid vs multi-hop)
- Document information density
- Content relevance to question
"""

import re
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DynamicRatioSelector:
    """
    Dynamically selects compression ratio based on question and document properties.
    
    Key insight: Simple questions with high-density documents need less context,
    while complex questions with sparse documents need more context.
    """
    
    def __init__(
        self,
        base_ratio: float = 0.5,
        min_ratio: float = 0.2,
        max_ratio: float = 0.9
    ):
        """
        Initialize the dynamic ratio selector.
        
        Args:
            base_ratio: Default compression ratio
            min_ratio: Minimum allowed ratio (most aggressive compression)
            max_ratio: Maximum allowed ratio (least compression)
        """
        self.base_ratio = base_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    
    def compute_optimal_ratio(
        self,
        question: str,
        document: str,
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Compute optimal compression ratio for given question and document.
        
        Args:
            question: The question to answer
            document: Source document/passage
            metadata: Optional additional metadata
            
        Returns:
            Optimal compression ratio (0.0 to 1.0)
        """
        # Estimate question complexity (0-1, higher = more complex)
        complexity = self.estimate_question_complexity(question)
        
        # Estimate document density (0-1, higher = more dense/informative)
        density = self.estimate_document_density(document)
        
        # Estimate relevance (how much of document is relevant)
        relevance = self.estimate_relevance(question, document)
        
        # Compute adjusted ratio
        # Higher complexity -> need more context -> higher ratio
        # Higher density -> can compress more -> lower ratio
        # Higher relevance -> need more context -> higher ratio
        
        complexity_factor = 1.0 + (complexity - 0.5) * 0.4  # 0.8 to 1.2
        density_factor = 1.0 - (density - 0.5) * 0.3  # 0.85 to 1.15
        relevance_factor = 1.0 + (relevance - 0.5) * 0.3  # 0.85 to 1.15
        
        adjusted_ratio = self.base_ratio * complexity_factor * density_factor * relevance_factor
        
        # Clamp to valid range
        optimal_ratio = max(self.min_ratio, min(self.max_ratio, adjusted_ratio))
        
        logger.info(
            f"Dynamic ratio: base={self.base_ratio:.2f} -> optimal={optimal_ratio:.2f} "
            f"(complexity={complexity:.2f}, density={density:.2f}, relevance={relevance:.2f})"
        )
        
        return optimal_ratio
    
    def estimate_question_complexity(self, question: str) -> float:
        """
        Estimate question complexity.
        
        Args:
            question: The question text
            
        Returns:
            Complexity score (0.0 to 1.0)
        """
        score = 0.5  # Base score
        
        question_lower = question.lower()
        
        # Multi-hop indicators (complex)
        multi_hop_words = ['both', 'and', 'compare', 'difference', 'relationship', 
                          'how many', 'all', 'each', 'between']
        for word in multi_hop_words:
            if word in question_lower:
                score += 0.1
        
        # Reasoning indicators (complex)
        reasoning_words = ['why', 'how', 'explain', 'cause', 'reason', 
                          'because', 'result', 'effect', 'impact']
        for word in reasoning_words:
            if word in question_lower:
                score += 0.08
        
        # Simple factoid indicators (simple)
        factoid_words = ['what is', 'who is', 'when did', 'where is', 
                        'name of', 'which']
        for word in factoid_words:
            if word in question_lower:
                score -= 0.05
        
        # Question length factor (longer usually more complex)
        word_count = len(question.split())
        if word_count > 15:
            score += 0.1
        elif word_count < 6:
            score -= 0.1
        
        # Clamp to 0-1
        return max(0.0, min(1.0, score))
    
    def estimate_document_density(self, document: str) -> float:
        """
        Estimate document information density.
        
        Args:
            document: Source document text
            
        Returns:
            Density score (0.0 to 1.0)
        """
        if not document:
            return 0.5
        
        score = 0.5  # Base score
        
        words = document.split()
        word_count = len(words)
        
        # Entity density (capitalized words, numbers, dates)
        capitalized = len(re.findall(r'\b[A-Z][a-z]+', document))
        numbers = len(re.findall(r'\b\d+', document))
        years = len(re.findall(r'\b(19|20)\d{2}\b', document))
        
        entity_density = (capitalized + numbers * 1.5 + years * 2) / max(word_count, 1)
        score += entity_density * 2  # Scale appropriately
        
        # Sentence count (more sentences = potentially more redundancy)
        sentences = re.split(r'[.!?]+\s+', document)
        sentence_count = len([s for s in sentences if s.strip()])
        
        if sentence_count > 10:
            score -= 0.1  # More redundancy expected
        elif sentence_count < 3:
            score += 0.1  # Less redundancy, need more care
        
        # Average sentence length (longer = more dense)
        avg_sentence_len = word_count / max(sentence_count, 1)
        if avg_sentence_len > 25:
            score += 0.1
        elif avg_sentence_len < 10:
            score -= 0.1
        
        # Clamp to 0-1
        return max(0.0, min(1.0, score))
    
    def estimate_relevance(self, question: str, document: str) -> float:
        """
        Estimate how much of the document is relevant to the question.
        
        Args:
            question: The question
            document: Source document
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        # Extract question keywords
        keywords = self._extract_keywords(question)
        if not keywords:
            return 0.5
        
        # Split document into sentences
        sentences = re.split(r'[.!?]+\s+', document)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.5
        
        # Count sentences with keyword matches
        relevant_count = 0
        for sent in sentences:
            sent_lower = sent.lower()
            if any(kw.lower() in sent_lower for kw in keywords):
                relevant_count += 1
        
        relevance = relevant_count / len(sentences)
        
        return relevance
    
    @staticmethod
    def _extract_keywords(question: str) -> List[str]:
        """Extract keywords from question."""
        stopwords = {
            'what', 'when', 'where', 'who', 'why', 'how',
            'is', 'are', 'was', 'were', 'the', 'a', 'an',
            'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or',
            'did', 'does', 'do', 'it', 'this', 'that'
        }
        words = question.lower().split()
        return [w.strip('?.,!') for w in words 
                if w.lower() not in stopwords and len(w) > 2]


class AdaptiveClarifier:
    """
    Enhanced Clarifier with dynamic ratio selection.
    
    Wraps the original Clarifier and adds dynamic ratio computation.
    """
    
    def __init__(
        self,
        current_role: str,
        next_role: str,
        base_ratio: float = 0.5
    ):
        """
        Initialize adaptive clarifier.
        
        Args:
            current_role: Current agent's role
            next_role: Next agent's role
            base_ratio: Base compression ratio
        """
        self.current_role = current_role
        self.next_role = next_role
        self.ratio_selector = DynamicRatioSelector(base_ratio=base_ratio)
        
        # Import here to avoid circular imports
        from compression.role_specific import Clarifier
        self.clarifier = Clarifier(current_role, next_role)
    
    def clarify(
        self,
        context: str,
        metadata: Optional[Dict] = None,
        target_compression: Optional[float] = None,
        min_sentences: int = 2
    ) -> str:
        """
        Compress context with dynamic ratio selection.
        
        Args:
            context: Input context to compress
            metadata: Metadata including question
            target_compression: Override for dynamic ratio (if None, compute dynamically)
            min_sentences: Minimum sentences to keep
            
        Returns:
            Compressed context
        """
        if metadata is None:
            metadata = {}
        
        question = metadata.get('question', '')
        
        # Compute dynamic ratio if not overridden
        if target_compression is None:
            target_compression = self.ratio_selector.compute_optimal_ratio(
                question=question,
                document=context,
                metadata=metadata
            )
        
        # Use the clarifier with computed ratio
        return self.clarifier.clarify(
            context=context,
            metadata=metadata,
            target_compression=target_compression,
            min_sentences=min_sentences
        )
    
    def get_compression_stats(self, original: str, compressed: str) -> Dict:
        """Get compression statistics."""
        return self.clarifier.get_compression_stats(original, compressed)
