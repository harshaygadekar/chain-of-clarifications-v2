"""
Metrics Module

Provides comprehensive metrics tracking for the Chain of Clarifications system.
Tracks accuracy, context sizes, latency, and other key performance indicators.
"""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks and computes various metrics for evaluating the system.

    Metrics include:
    - F1 scores
    - Exact Match (EM) accuracy
    - Context sizes at each agent
    - Token usage
    - Latency
    - Success rate
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.predictions = []
        self.ground_truths = []
        self.f1_scores = []
        self.em_scores = []
        self.context_sizes = defaultdict(list)  # By agent role
        self.token_counts = defaultdict(list)
        self.latencies = []
        self.success_flags = []
        self.errors = []
        self.memory_usage = []

    def add_result(
        self,
        prediction: str,
        ground_truth: str,
        context_sizes: Optional[Dict[str, int]] = None,
        latency: Optional[float] = None,
        success: bool = True,
        error: Optional[str] = None,
        memory_info: Optional[Dict] = None,
        **kwargs
    ):
        """
        Add a result for tracking.

        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            context_sizes: Dictionary of context sizes by agent role
            latency: Time taken for this example
            success: Whether the processing succeeded
            error: Error message if failed
            memory_info: Memory usage information
            **kwargs: Additional metrics
        """
        self.predictions.append(prediction)
        self.ground_truths.append(ground_truth)

        # Compute and store F1 and EM
        f1 = self.compute_f1(prediction, ground_truth)
        em = self.compute_em(prediction, ground_truth)
        self.f1_scores.append(f1)
        self.em_scores.append(em)

        # Store context sizes
        if context_sizes:
            for role, size in context_sizes.items():
                self.context_sizes[role].append(size)

        # Store latency
        if latency is not None:
            self.latencies.append(latency)

        # Store success flag
        self.success_flags.append(success)

        # Store error if any
        if error:
            self.errors.append(error)

        # Store memory info
        if memory_info:
            self.memory_usage.append(memory_info)

    def compute_f1(self, prediction: str, ground_truth: str) -> float:
        """
        Compute F1 score between prediction and ground truth.

        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            F1 score (0.0 to 1.0)
        """
        pred_tokens = self._normalize_answer(prediction).split()
        truth_tokens = self._normalize_answer(ground_truth).split()

        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return float(len(pred_tokens) == len(truth_tokens))

        common_tokens = set(pred_tokens) & set(truth_tokens)
        num_common = len(common_tokens)

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

    def compute_em(self, prediction: str, ground_truth: str) -> float:
        """
        Compute exact match score.

        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return float(
            self._normalize_answer(prediction) == self._normalize_answer(ground_truth)
        )

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalize answer text for comparison."""
        import re
        import string

        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = ' '.join(text.split())
        return text.strip()

    def get_summary(self) -> Dict:
        """
        Get summary statistics of all tracked metrics.

        Returns:
            Dictionary with summary statistics
        """
        summary = {}

        # F1 and EM scores
        if self.f1_scores:
            summary['f1_mean'] = np.mean(self.f1_scores)
            summary['f1_std'] = np.std(self.f1_scores)
            summary['f1_median'] = np.median(self.f1_scores)

        if self.em_scores:
            summary['em_mean'] = np.mean(self.em_scores)
            summary['em_std'] = np.std(self.em_scores)

        # Context sizes by agent
        for role, sizes in self.context_sizes.items():
            if sizes:
                summary[f'{role}_context_mean'] = np.mean(sizes)
                summary[f'{role}_context_std'] = np.std(sizes)
                summary[f'{role}_context_max'] = np.max(sizes)

        # Latency
        if self.latencies:
            summary['latency_mean'] = np.mean(self.latencies)
            summary['latency_std'] = np.std(self.latencies)
            summary['latency_median'] = np.median(self.latencies)

        # Success rate
        if self.success_flags:
            summary['success_rate'] = np.mean(self.success_flags)
            summary['num_examples'] = len(self.success_flags)
            summary['num_failures'] = len([x for x in self.success_flags if not x])

        # Memory usage
        if self.memory_usage:
            summary['avg_memory_mb'] = np.mean([
                m.get('allocated_mb', 0) for m in self.memory_usage
            ])
            summary['max_memory_mb'] = np.max([
                m.get('max_allocated_mb', 0) for m in self.memory_usage
            ])

        return summary

    def print_summary(self):
        """Print a formatted summary of metrics."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("METRICS SUMMARY")
        print("=" * 60)

        if 'num_examples' in summary:
            print(f"\nTotal Examples: {summary['num_examples']}")
            print(f"Success Rate: {summary['success_rate']:.2%}")

        if 'f1_mean' in summary:
            print(f"\nF1 Score:")
            print(f"  Mean: {summary['f1_mean']:.4f}")
            print(f"  Std:  {summary['f1_std']:.4f}")
            print(f"  Median: {summary['f1_median']:.4f}")

        if 'em_mean' in summary:
            print(f"\nExact Match:")
            print(f"  Mean: {summary['em_mean']:.4f}")
            print(f"  Percentage: {summary['em_mean']*100:.2f}%")

        # Context sizes
        print(f"\nContext Sizes:")
        for key in summary:
            if '_context_mean' in key:
                role = key.replace('_context_mean', '')
                print(f"  {role.capitalize()}:")
                print(f"    Mean: {summary[key]:.1f} tokens")
                if f'{role}_context_max' in summary:
                    print(f"    Max:  {summary[f'{role}_context_max']:.1f} tokens")

        if 'latency_mean' in summary:
            print(f"\nLatency:")
            print(f"  Mean: {summary['latency_mean']:.2f}s")
            print(f"  Median: {summary['latency_median']:.2f}s")

        if 'avg_memory_mb' in summary:
            print(f"\nMemory Usage:")
            print(f"  Average: {summary['avg_memory_mb']:.1f} MB")
            print(f"  Peak: {summary['max_memory_mb']:.1f} MB")

        print("=" * 60 + "\n")

    def export_to_dict(self) -> Dict:
        """
        Export all raw data to dictionary.

        Returns:
            Dictionary with all tracked data
        """
        return {
            'predictions': self.predictions,
            'ground_truths': self.ground_truths,
            'f1_scores': self.f1_scores,
            'em_scores': self.em_scores,
            'context_sizes': dict(self.context_sizes),
            'latencies': self.latencies,
            'success_flags': self.success_flags,
            'errors': self.errors,
            'memory_usage': self.memory_usage,
            'summary': self.get_summary()
        }


def compute_confidence_interval(scores: List[float], confidence: float = 0.95) -> tuple:
    """
    Compute confidence interval for scores.

    Args:
        scores: List of scores
        confidence: Confidence level (default 0.95 for 95%)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    from scipy import stats

    mean = np.mean(scores)
    sem = stats.sem(scores)
    ci = stats.t.interval(confidence, len(scores)-1, loc=mean, scale=sem)

    return mean, ci[0], ci[1]


def paired_t_test(scores1: List[float], scores2: List[float]) -> Dict:
    """
    Perform paired t-test between two sets of scores.

    Args:
        scores1: First set of scores
        scores2: Second set of scores

    Returns:
        Dictionary with test results
    """
    from scipy import stats

    t_stat, p_value = stats.ttest_rel(scores1, scores2)

    # Compute Cohen's d (effect size)
    diff = np.array(scores1) - np.array(scores2)
    cohens_d = np.mean(diff) / np.std(diff)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05,
        'mean_diff': np.mean(diff)
    }
