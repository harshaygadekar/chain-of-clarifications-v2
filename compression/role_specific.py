"""
Role-Specific Compression Strategies

Implements compression strategies tailored to each agent's role and information needs.
This is the core innovation of the Chain of Clarifications approach.
"""

from typing import Dict, List, Optional, Tuple
from compression.naive_compression import SentenceScorer
import re
import logging

logger = logging.getLogger(__name__)


class RoleSpecificScorer:
    """
    Scores content importance based on the next agent's role and needs.

    Each role has different information requirements:
    - Reasoner needs: relevant facts, entities, supporting evidence
    - Verifier needs: final answer, reasoning chain, justification
    """

    def __init__(self, current_role: str, next_role: str):
        """
        Initialize the role-specific scorer.

        Args:
            current_role: Current agent's role
            next_role: Next agent's role (who will receive the compressed context)
        """
        self.current_role = current_role
        self.next_role = next_role
        self.sentence_scorer = SentenceScorer()

    def score_tokens(
        self,
        context: str,
        metadata: Optional[Dict] = None
    ) -> List[float]:
        """
        Score tokens by importance for next role.

        Args:
            context: Context text to score
            metadata: Additional information (question, etc.)

        Returns:
            List of importance scores per sentence
        """
        if metadata is None:
            metadata = {}

        if self.next_role == "reasoner":
            return self._score_for_reasoner(context, metadata)
        elif self.next_role == "verifier":
            return self._score_for_verifier(context, metadata)
        else:
            # Fallback to position-based scoring
            sentences = self._split_sentences(context)
            return self.sentence_scorer.score_by_position(sentences)

    def _score_for_reasoner(
        self,
        context: str,
        metadata: Dict
    ) -> List[float]:
        """
        Score content for Reasoner agent.

        Reasoner needs:
        - Sentences relevant to the question
        - Entities (names, dates, numbers)
        - Supporting facts
        - Less: background info, redundant content
        """
        sentences = self._split_sentences(context)
        question = metadata.get('question', '')

        # Extract question keywords
        question_words = self._extract_keywords(question)

        # Multiple scoring criteria
        score_lists = []
        weights = []

        # 1. Keyword overlap with question (high weight)
        if question_words:
            keyword_scores = self.sentence_scorer.score_by_keywords(
                sentences,
                question_words
            )
            score_lists.append(keyword_scores)
            weights.append(3.0)

        # 2. Entity presence (dates, names, numbers)
        entity_scores = self.sentence_scorer.score_by_entities(sentences)
        score_lists.append(entity_scores)
        weights.append(2.5)

        # 3. Position (earlier sentences often have context)
        position_scores = self.sentence_scorer.score_by_position(sentences)
        score_lists.append(position_scores)
        weights.append(1.0)

        # 4. Length (longer sentences may have more info)
        length_scores = self.sentence_scorer.score_by_length(sentences)
        score_lists.append(length_scores)
        weights.append(0.5)

        # Combine scores
        combined_scores = self.sentence_scorer.combine_scores(
            score_lists,
            weights
        )

        return combined_scores

    def _score_for_verifier(
        self,
        context: str,
        metadata: Dict
    ) -> List[float]:
        """
        Score content for Verifier agent.

        Verifier needs:
        - Final answer and conclusion
        - Main reasoning steps
        - Supporting evidence
        - Less: exploratory reasoning, rejected alternatives
        """
        sentences = self._split_sentences(context)

        score_lists = []
        weights = []

        # 1. Answer indicators (very high weight)
        answer_scores = self._score_answer_sentences(sentences)
        score_lists.append(answer_scores)
        weights.append(5.0)

        # 2. Reasoning markers (therefore, because, thus)
        reasoning_scores = self._score_reasoning_sentences(sentences)
        score_lists.append(reasoning_scores)
        weights.append(3.0)

        # 3. Entities and facts
        entity_scores = self.sentence_scorer.score_by_entities(sentences)
        score_lists.append(entity_scores)
        weights.append(2.0)

        # 4. Position (final sentences often have conclusions)
        # Reverse position: later = more important for verifier
        position_scores = self._score_reverse_position(sentences)
        score_lists.append(position_scores)
        weights.append(1.5)

        # Combine scores
        combined_scores = self.sentence_scorer.combine_scores(
            score_lists,
            weights
        )

        return combined_scores

    def _score_answer_sentences(self, sentences: List[str]) -> List[float]:
        """Score sentences that likely contain answers."""
        answer_markers = [
            'answer', 'conclusion', 'result', 'therefore',
            'finally', 'in summary', 'thus', 'hence'
        ]

        scores = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(2.0 if marker in sent_lower else 0.0
                       for marker in answer_markers)
            scores.append(score)

        # Normalize
        max_score = max(scores) if scores and max(scores) > 0 else 1
        scores = [s / max_score for s in scores]

        return scores

    def _score_reasoning_sentences(self, sentences: List[str]) -> List[float]:
        """Score sentences that show reasoning."""
        reasoning_markers = [
            'because', 'since', 'therefore', 'thus', 'hence',
            'as a result', 'consequently', 'this means',
            'which indicates', 'suggesting that'
        ]

        scores = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(1.0 if marker in sent_lower else 0.0
                       for marker in reasoning_markers)
            scores.append(score)

        # Normalize
        max_score = max(scores) if scores and max(scores) > 0 else 1
        scores = [s / max_score for s in scores]

        return scores

    def _score_reverse_position(self, sentences: List[str]) -> List[float]:
        """Score with later sentences getting higher scores."""
        n = len(sentences)
        scores = [i / n for i in range(n)]
        return scores

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _extract_keywords(question: str) -> List[str]:
        """Extract important keywords from question."""
        # Remove question words and common words
        stopwords = {
            'what', 'when', 'where', 'who', 'why', 'how',
            'is', 'are', 'was', 'were', 'the', 'a', 'an',
            'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'
        }

        words = question.lower().split()
        keywords = [w.strip('?.,!') for w in words
                   if w.lower() not in stopwords and len(w) > 2]

        return keywords


class Clarifier:
    """
    Adaptive compression module that clarifies context for next agent.

    Uses role-specific strategies to compress context while preserving
    information critical for the next agent's task.
    """

    def __init__(self, current_role: str, next_role: str):
        """
        Initialize the Clarifier.

        Args:
            current_role: Current agent's role
            next_role: Next agent's role
        """
        self.current_role = current_role
        self.next_role = next_role
        self.scorer = RoleSpecificScorer(current_role, next_role)

    def clarify(
        self,
        context: str,
        metadata: Optional[Dict] = None,
        target_compression: float = 0.5,
        min_sentences: int = 2
    ) -> str:
        """
        Compress context adaptively for next role.

        Args:
            context: Input context to compress
            metadata: Additional metadata (question, etc.)
            target_compression: Target compression ratio (0.0 to 1.0)
            min_sentences: Minimum sentences to keep

        Returns:
            Compressed context optimized for next role
        """
        if metadata is None:
            metadata = {}

        # Split into sentences
        sentences = self._split_sentences(context)

        if len(sentences) <= min_sentences:
            return context

        # Score sentences for next role
        importance_scores = self.scorer.score_tokens(context, metadata)

        # Determine adaptive compression ratio
        actual_ratio = self._adaptive_ratio(
            importance_scores,
            target=target_compression
        )

        # Select sentences to keep
        keep_count = max(min_sentences, int(len(sentences) * actual_ratio))

        # Get top sentences by score
        indexed_scores = [(i, score) for i, score in enumerate(importance_scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        top_indices = sorted([i for i, _ in indexed_scores[:keep_count]])

        # Reconstruct context in original order
        compressed_context = ' '.join([sentences[i] for i in top_indices])

        logger.info(
            f"Clarifier ({self.current_role}→{self.next_role}): "
            f"{len(sentences)} → {keep_count} sentences "
            f"({actual_ratio:.2%} retention)"
        )

        return compressed_context

    def _adaptive_ratio(
        self,
        importance_scores: List[float],
        target: float
    ) -> float:
        """
        Adjust compression ratio based on importance distribution.

        If many high-importance sentences → compress less
        If mostly low-importance → compress more

        Args:
            importance_scores: Sentence importance scores
            target: Target compression ratio

        Returns:
            Adjusted compression ratio
        """
        if not importance_scores or len(importance_scores) == 0:
            return target

        # Count high-importance sentences (score > 0.7)
        high_importance_count = sum(
            1 for s in importance_scores if s > 0.7
        )
        high_importance_fraction = high_importance_count / len(importance_scores)

        # Adjust ratio
        if high_importance_fraction > 0.6:
            # Many important sentences → compress less aggressively
            adjusted = min(target * 1.3, 0.9)
        elif high_importance_fraction < 0.3:
            # Few important sentences → can compress more
            adjusted = max(target * 0.7, 0.2)
        else:
            adjusted = target

        return adjusted

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def get_compression_stats(
        self,
        original: str,
        compressed: str
    ) -> Dict:
        """Get compression statistics."""
        return {
            'current_role': self.current_role,
            'next_role': self.next_role,
            'original_length': len(original),
            'compressed_length': len(compressed),
            'compression_ratio': len(compressed) / len(original) if len(original) > 0 else 0,
            'original_sentences': len(self._split_sentences(original)),
            'compressed_sentences': len(self._split_sentences(compressed))
        }
