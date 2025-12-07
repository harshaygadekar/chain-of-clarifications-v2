"""
Naive Compression Module

Implements fixed-ratio compression strategies as baselines.
These serve as comparison points for the role-specific compression.
"""

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class NaiveCompressor:
    """
    Fixed-ratio compression that keeps a specified percentage of content.

    Strategies:
    - first_n: Keep first N% of tokens
    - last_n: Keep last N% of tokens
    - random: Randomly sample N% of tokens
    - sentence_first: Keep first N% of sentences
    """

    def __init__(self, compression_ratio: float = 0.5, strategy: str = "first_n"):
        """
        Initialize the naive compressor.

        Args:
            compression_ratio: Fraction of content to keep (0.0 to 1.0)
            strategy: Compression strategy to use
        """
        assert 0.0 < compression_ratio <= 1.0, "Ratio must be between 0 and 1"
        self.compression_ratio = compression_ratio
        self.strategy = strategy

    def compress(self, text: str) -> str:
        """
        Compress text using the specified strategy.

        Args:
            text: Input text to compress

        Returns:
            Compressed text
        """
        if self.compression_ratio >= 1.0:
            return text

        if self.strategy == "first_n":
            return self._compress_first_n(text)
        elif self.strategy == "last_n":
            return self._compress_last_n(text)
        elif self.strategy == "random":
            return self._compress_random(text)
        elif self.strategy == "sentence_first":
            return self._compress_sentence_first(text)
        else:
            logger.warning(f"Unknown strategy: {self.strategy}, using first_n")
            return self._compress_first_n(text)

    def _compress_first_n(self, text: str) -> str:
        """Keep first N% of tokens."""
        words = text.split()
        keep_count = max(1, int(len(words) * self.compression_ratio))
        compressed = ' '.join(words[:keep_count])
        return compressed

    def _compress_last_n(self, text: str) -> str:
        """Keep last N% of tokens."""
        words = text.split()
        keep_count = max(1, int(len(words) * self.compression_ratio))
        compressed = ' '.join(words[-keep_count:])
        return compressed

    def _compress_random(self, text: str) -> str:
        """Randomly sample N% of tokens."""
        import random
        words = text.split()
        keep_count = max(1, int(len(words) * self.compression_ratio))

        # Set seed for reproducibility
        random.seed(42)
        sampled_indices = sorted(random.sample(range(len(words)), keep_count))
        compressed = ' '.join([words[i] for i in sampled_indices])
        return compressed

    def _compress_sentence_first(self, text: str) -> str:
        """Keep first N% of sentences."""
        sentences = self._split_sentences(text)
        keep_count = max(1, int(len(sentences) * self.compression_ratio))
        compressed = ' '.join(sentences[:keep_count])
        return compressed

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        import re
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def get_compression_stats(self, original: str, compressed: str) -> dict:
        """
        Get compression statistics.

        Args:
            original: Original text
            compressed: Compressed text

        Returns:
            Dictionary with statistics
        """
        orig_words = len(original.split())
        comp_words = len(compressed.split())
        orig_chars = len(original)
        comp_chars = len(compressed)

        return {
            'original_words': orig_words,
            'compressed_words': comp_words,
            'word_reduction': 1 - (comp_words / orig_words) if orig_words > 0 else 0,
            'original_chars': orig_chars,
            'compressed_chars': comp_chars,
            'char_reduction': 1 - (comp_chars / orig_chars) if orig_chars > 0 else 0,
            'target_ratio': self.compression_ratio,
            'actual_word_ratio': comp_words / orig_words if orig_words > 0 else 0
        }


class SentenceScorer:
    """
    Scores sentences by importance for more intelligent compression.
    Used as a building block for role-specific strategies.
    """

    def __init__(self):
        """Initialize the sentence scorer."""
        pass

    def score_by_position(self, sentences: List[str]) -> List[float]:
        """
        Score sentences by position (earlier = more important).

        Args:
            sentences: List of sentences

        Returns:
            List of scores (0.0 to 1.0)
        """
        n = len(sentences)
        scores = [1.0 - (i / n) for i in range(n)]
        return scores

    def score_by_length(self, sentences: List[str]) -> List[float]:
        """
        Score sentences by length (longer = more informative).

        Args:
            sentences: List of sentences

        Returns:
            List of scores (0.0 to 1.0)
        """
        lengths = [len(s.split()) for s in sentences]
        max_len = max(lengths) if lengths else 1
        scores = [l / max_len for l in lengths]
        return scores

    def score_by_keywords(
        self,
        sentences: List[str],
        keywords: List[str]
    ) -> List[float]:
        """
        Score sentences by keyword presence.

        Args:
            sentences: List of sentences
            keywords: Keywords to look for

        Returns:
            List of scores (0.0 to 1.0)
        """
        scores = []
        keywords_lower = [k.lower() for k in keywords]

        for sent in sentences:
            sent_lower = sent.lower()
            count = sum(1 for kw in keywords_lower if kw in sent_lower)
            scores.append(count)

        # Normalize
        max_score = max(scores) if scores else 1
        if max_score > 0:
            scores = [s / max_score for s in scores]

        return scores

    def score_by_entities(self, sentences: List[str]) -> List[float]:
        """
        Score sentences by presence of entities (capitalized words, numbers, dates).

        Args:
            sentences: List of sentences

        Returns:
            List of scores (0.0 to 1.0)
        """
        import re
        scores = []

        for sent in sentences:
            score = 0
            # Count capitalized words (potential entities)
            capitalized = len(re.findall(r'\b[A-Z][a-z]+', sent))
            score += capitalized

            # Count numbers
            numbers = len(re.findall(r'\b\d+', sent))
            score += numbers * 1.5  # Numbers often important

            # Count years
            years = len(re.findall(r'\b(19|20)\d{2}\b', sent))
            score += years * 2  # Dates very important

            scores.append(score)

        # Normalize
        max_score = max(scores) if scores else 1
        if max_score > 0:
            scores = [s / max_score for s in scores]

        return scores

    def combine_scores(
        self,
        score_lists: List[List[float]],
        weights: Optional[List[float]] = None
    ) -> List[float]:
        """
        Combine multiple score lists with weights.

        Args:
            score_lists: List of score lists
            weights: Weights for each score list (None for equal weights)

        Returns:
            Combined scores
        """
        if not score_lists:
            return []

        n = len(score_lists[0])

        if weights is None:
            weights = [1.0] * len(score_lists)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Combine
        combined = [0.0] * n
        for scores, weight in zip(score_lists, weights):
            for i in range(n):
                combined[i] += scores[i] * weight

        return combined


def compress_with_sentence_scoring(
    text: str,
    compression_ratio: float,
    score_method: str = "position",
    keywords: Optional[List[str]] = None
) -> str:
    """
    Compress text using sentence scoring.

    Args:
        text: Input text
        compression_ratio: Fraction to keep
        score_method: Scoring method (position, length, keywords, entities)
        keywords: Keywords for keyword-based scoring

    Returns:
        Compressed text
    """
    compressor = NaiveCompressor(compression_ratio)
    scorer = SentenceScorer()

    sentences = compressor._split_sentences(text)

    if len(sentences) <= 1:
        return text

    # Score sentences
    if score_method == "position":
        scores = scorer.score_by_position(sentences)
    elif score_method == "length":
        scores = scorer.score_by_length(sentences)
    elif score_method == "keywords" and keywords:
        scores = scorer.score_by_keywords(sentences, keywords)
    elif score_method == "entities":
        scores = scorer.score_by_entities(sentences)
    else:
        scores = scorer.score_by_position(sentences)

    # Select top sentences
    keep_count = max(1, int(len(sentences) * compression_ratio))
    indexed_scores = [(i, score) for i, score in enumerate(scores)]
    indexed_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted([i for i, _ in indexed_scores[:keep_count]])

    # Reconstruct text in original order
    compressed = ' '.join([sentences[i] for i in top_indices])

    return compressed
