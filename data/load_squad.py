"""
SQuAD Dataset Loader

This module provides utilities for loading and preprocessing the SQuAD 1.1 dataset
for use in the Chain of Clarifications experiments.
"""

from datasets import load_dataset
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SQuADLoader:
    """
    Loader for SQuAD 1.1 dataset.

    Handles downloading, caching, and preprocessing the dataset
    for question answering experiments.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the SQuAD loader.

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        self.dataset = None
        self.train_data = None
        self.validation_data = None

    def load(self, split: str = "validation") -> None:
        """
        Load the SQuAD dataset.

        Args:
            split: Which split to load ('train' or 'validation')
        """
        try:
            logger.info(f"Loading SQuAD 1.1 dataset ({split} split)...")
            self.dataset = load_dataset(
                "squad",
                split=split,
                cache_dir=self.cache_dir
            )
            logger.info(f"Loaded {len(self.dataset)} examples")

            if split == "train":
                self.train_data = self.dataset
            else:
                self.validation_data = self.dataset

        except Exception as e:
            logger.error(f"Failed to load SQuAD dataset: {e}")
            raise

    def get_examples(
        self,
        num_examples: Optional[int] = None,
        start_idx: int = 0,
        split: str = "validation"
    ) -> List[Dict]:
        """
        Get examples from the dataset.

        Args:
            num_examples: Number of examples to return (None for all)
            start_idx: Starting index
            split: Dataset split to use

        Returns:
            List of example dictionaries
        """
        # Load dataset if not already loaded
        if self.dataset is None:
            self.load(split)

        # Determine dataset to use
        if split == "train" and self.train_data is not None:
            data = self.train_data
        elif split == "validation" and self.validation_data is not None:
            data = self.validation_data
        else:
            data = self.dataset

        # Extract examples
        if num_examples is None:
            end_idx = len(data)
        else:
            end_idx = min(start_idx + num_examples, len(data))

        examples = []
        for idx in range(start_idx, end_idx):
            example = data[idx]
            examples.append(self._preprocess_example(example))

        logger.info(f"Retrieved {len(examples)} examples from {split} split")
        return examples

    def _preprocess_example(self, example: Dict) -> Dict:
        """
        Preprocess a single example.

        Args:
            example: Raw example from dataset

        Returns:
            Preprocessed example dictionary
        """
        # Extract answer text and position
        answers = example.get('answers', {})
        answer_text = answers.get('text', [''])[0] if answers.get('text') else ''
        answer_start = answers.get('answer_start', [0])[0] if answers.get('answer_start') else 0

        processed = {
            'id': example.get('id', ''),
            'question': example.get('question', ''),
            'context': example.get('context', ''),
            'answer_text': answer_text,
            'answer_start': answer_start,
            'title': example.get('title', ''),
        }

        return processed

    def get_sample_for_testing(self, num_samples: int = 10) -> List[Dict]:
        """
        Get a small sample for testing.

        Args:
            num_samples: Number of samples to return

        Returns:
            List of sample examples
        """
        return self.get_examples(num_examples=num_samples, split="validation")

    def compute_f1(self, prediction: str, ground_truth: str) -> float:
        """
        Compute F1 score between prediction and ground truth.

        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            F1 score (0.0 to 1.0)
        """
        # Normalize strings
        pred_tokens = self._normalize_answer(prediction).split()
        truth_tokens = self._normalize_answer(ground_truth).split()

        # Handle empty cases
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return float(len(pred_tokens) == len(truth_tokens))

        # Compute token overlap
        common_tokens = set(pred_tokens) & set(truth_tokens)
        num_common = len(common_tokens)

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    def compute_exact_match(self, prediction: str, ground_truth: str) -> bool:
        """
        Compute exact match between prediction and ground truth.

        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            True if exact match, False otherwise
        """
        return self._normalize_answer(prediction) == self._normalize_answer(ground_truth)

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """
        Normalize answer text for comparison.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        import re
        import string

        # Lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.strip()

    def get_statistics(self, split: str = "validation") -> Dict:
        """
        Get dataset statistics.

        Args:
            split: Dataset split

        Returns:
            Dictionary with statistics
        """
        if self.dataset is None:
            self.load(split)

        examples = self.get_examples(split=split)

        # Compute statistics
        context_lengths = [len(ex['context']) for ex in examples]
        question_lengths = [len(ex['question']) for ex in examples]
        answer_lengths = [len(ex['answer_text']) for ex in examples]

        stats = {
            'num_examples': len(examples),
            'avg_context_length': sum(context_lengths) / len(context_lengths),
            'avg_question_length': sum(question_lengths) / len(question_lengths),
            'avg_answer_length': sum(answer_lengths) / len(answer_lengths),
            'max_context_length': max(context_lengths),
            'max_question_length': max(question_lengths),
        }

        return stats
