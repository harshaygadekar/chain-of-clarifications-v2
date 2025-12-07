"""
Error Analysis Pipeline

Categorizes and analyzes prediction errors for deeper insights
into system behavior and failure modes.
"""

import re
from typing import Dict, List, Optional, Tuple
from enum import Enum
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of prediction errors."""
    CORRECT = "correct"
    PARTIAL_MATCH = "partial_match"
    WRONG_ENTITY = "wrong_entity"
    INCOMPLETE = "incomplete"
    HALLUCINATION = "hallucination"
    EMPTY = "empty"
    PARAPHRASE = "paraphrase"
    WRONG_TYPE = "wrong_type"
    OFF_TOPIC = "off_topic"


class ErrorAnalyzer:
    """
    Analyzes prediction errors and categorizes them.
    
    Provides insights into:
    - What types of errors the system makes
    - Which questions are harder
    - Where compression may be losing information
    """
    
    def __init__(self, f1_threshold: float = 0.5, em_threshold: float = 0.9):
        """
        Initialize error analyzer.
        
        Args:
            f1_threshold: F1 threshold for partial match
            em_threshold: F1 threshold for considering near-exact
        """
        self.f1_threshold = f1_threshold
        self.em_threshold = em_threshold
    
    def categorize_error(
        self,
        prediction: str,
        ground_truth: str,
        context: Optional[str] = None,
        question: Optional[str] = None
    ) -> Tuple[ErrorType, Dict]:
        """
        Categorize a single prediction error.
        
        Args:
            prediction: Model's prediction
            ground_truth: Correct answer
            context: Original context/document
            question: The question asked
            
        Returns:
            Tuple of (ErrorType, details dict)
        """
        details = {
            'prediction': prediction,
            'ground_truth': ground_truth,
            'f1_score': 0.0,
            'reason': ''
        }
        
        # Handle empty predictions
        if not prediction or prediction.strip() == '':
            details['reason'] = 'Model produced empty output'
            return ErrorType.EMPTY, details
        
        # Check for error markers from model
        if prediction.startswith('[ERROR:'):
            details['reason'] = 'Model encountered an error'
            return ErrorType.EMPTY, details
        
        # Normalize for comparison
        pred_norm = self._normalize(prediction)
        truth_norm = self._normalize(ground_truth)
        
        # Compute F1
        f1 = self._compute_f1(pred_norm, truth_norm)
        details['f1_score'] = f1
        
        # Exact or near-exact match
        if f1 >= self.em_threshold:
            details['reason'] = 'Correct or near-correct answer'
            return ErrorType.CORRECT, details
        
        # High F1 but not exact - likely paraphrase or partial
        if f1 >= self.f1_threshold:
            # Check if it's a paraphrase (similar meaning, different words)
            if self._is_paraphrase(prediction, ground_truth):
                details['reason'] = 'Semantically similar but different wording'
                return ErrorType.PARAPHRASE, details
            else:
                details['reason'] = 'Partial overlap with ground truth'
                return ErrorType.PARTIAL_MATCH, details
        
        # Check if prediction is incomplete (subset of answer)
        if self._is_incomplete(pred_norm, truth_norm):
            details['reason'] = 'Answer is incomplete'
            return ErrorType.INCOMPLETE, details
        
        # Check for hallucination (answer not in context)
        if context and self._is_hallucination(prediction, context):
            details['reason'] = 'Answer not found in context'
            return ErrorType.HALLUCINATION, details
        
        # Check for wrong entity type
        if self._is_wrong_type(prediction, ground_truth):
            details['reason'] = 'Wrong type of entity'
            return ErrorType.WRONG_TYPE, details
        
        # Check for wrong entity (same type, wrong instance)
        if self._is_wrong_entity(prediction, ground_truth, context):
            details['reason'] = 'Wrong entity of same type'
            return ErrorType.WRONG_ENTITY, details
        
        # Off-topic answer
        details['reason'] = 'Answer is off-topic or unrelated'
        return ErrorType.OFF_TOPIC, details
    
    def analyze_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
        contexts: Optional[List[str]] = None,
        questions: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze a batch of predictions.
        
        Args:
            predictions: List of predictions
            ground_truths: List of ground truths
            contexts: Optional list of contexts
            questions: Optional list of questions
            
        Returns:
            Analysis results dictionary
        """
        if contexts is None:
            contexts = [None] * len(predictions)
        if questions is None:
            questions = [None] * len(predictions)
        
        error_counts = Counter()
        examples_by_type = {et: [] for et in ErrorType}
        all_details = []
        
        for i, (pred, truth, ctx, q) in enumerate(zip(
            predictions, ground_truths, contexts, questions
        )):
            error_type, details = self.categorize_error(pred, truth, ctx, q)
            error_counts[error_type] += 1
            details['index'] = i
            details['question'] = q
            all_details.append(details)
            
            # Store example (limit to 5 per type)
            if len(examples_by_type[error_type]) < 5:
                examples_by_type[error_type].append({
                    'index': i,
                    'prediction': pred,
                    'ground_truth': truth,
                    'question': q
                })
        
        # Compute statistics
        total = len(predictions)
        correct = error_counts[ErrorType.CORRECT]
        
        analysis = {
            'total_examples': total,
            'correct_count': correct,
            'accuracy': correct / total if total > 0 else 0,
            'error_distribution': {
                et.value: count for et, count in error_counts.items()
            },
            'error_percentages': {
                et.value: count / total * 100 if total > 0 else 0
                for et, count in error_counts.items()
            },
            'examples_by_type': {
                et.value: examples for et, examples in examples_by_type.items()
                if examples
            },
            'all_details': all_details
        }
        
        return analysis
    
    def generate_error_report(
        self,
        analysis: Dict,
        include_examples: bool = True
    ) -> str:
        """
        Generate human-readable error report.
        
        Args:
            analysis: Analysis results from analyze_batch
            include_examples: Whether to include example errors
            
        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("ERROR ANALYSIS REPORT")
        lines.append("=" * 60)
        
        lines.append(f"\nTotal Examples: {analysis['total_examples']}")
        lines.append(f"Correct: {analysis['correct_count']} ({analysis['accuracy']:.2%})")
        
        lines.append("\n--- Error Distribution ---")
        for error_type, count in sorted(
            analysis['error_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            pct = analysis['error_percentages'][error_type]
            if count > 0:
                lines.append(f"  {error_type}: {count} ({pct:.1f}%)")
        
        if include_examples:
            lines.append("\n--- Example Errors ---")
            for error_type, examples in analysis.get('examples_by_type', {}).items():
                if error_type != 'correct' and examples:
                    lines.append(f"\n{error_type.upper()}:")
                    for ex in examples[:3]:  # Show max 3 per type
                        lines.append(f"  Q: {ex.get('question', 'N/A')[:60]}...")
                        lines.append(f"  Pred: {ex['prediction'][:50]}...")
                        lines.append(f"  GT: {ex['ground_truth'][:50]}...")
                        lines.append("")
        
        lines.append("=" * 60)
        
        return '\n'.join(lines)
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        import string
        
        if not text:
            return ""
        
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = ' '.join(text.split())
        return text.strip()
    
    def _compute_f1(self, pred: str, truth: str) -> float:
        """Compute F1 score between normalized strings."""
        pred_tokens = pred.split()
        truth_tokens = truth.split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return float(len(pred_tokens) == len(truth_tokens))
        
        common = set(pred_tokens) & set(truth_tokens)
        num_common = len(common)
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    def _is_paraphrase(self, pred: str, truth: str) -> bool:
        """Check if prediction is a paraphrase of truth."""
        # Simple heuristic: similar length, different words, some overlap
        pred_words = set(self._normalize(pred).split())
        truth_words = set(self._normalize(truth).split())
        
        overlap = len(pred_words & truth_words)
        total = len(pred_words | truth_words)
        
        # Some overlap but not all same words
        jaccard = overlap / total if total > 0 else 0
        return 0.3 <= jaccard <= 0.7
    
    def _is_incomplete(self, pred: str, truth: str) -> bool:
        """Check if prediction is incomplete version of truth."""
        pred_words = set(pred.split())
        truth_words = set(truth.split())
        
        # Pred is subset of truth
        if pred_words <= truth_words and len(pred_words) < len(truth_words):
            return True
        
        # Pred is much shorter but overlaps
        if len(pred_words) < len(truth_words) * 0.5:
            overlap = len(pred_words & truth_words)
            if overlap > 0 and overlap == len(pred_words):
                return True
        
        return False
    
    def _is_hallucination(self, pred: str, context: str) -> bool:
        """Check if prediction is hallucinated (not in context)."""
        if not context:
            return False
        
        pred_norm = self._normalize(pred)
        context_norm = self._normalize(context)
        
        # Check if main prediction words appear in context
        pred_words = pred_norm.split()
        if len(pred_words) == 0:
            return False
        
        # At least half of prediction words should be in context
        in_context = sum(1 for w in pred_words if w in context_norm)
        return in_context < len(pred_words) * 0.5
    
    def _is_wrong_type(self, pred: str, truth: str) -> bool:
        """Check if prediction is wrong type (e.g., date vs person)."""
        # Simple type detection
        def get_type(text):
            if re.match(r'^[\d,]+$', text.strip()):
                return 'number'
            if re.match(r'^\d{4}$', text.strip()):
                return 'year'
            if re.match(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', text):
                return 'date'
            if text.strip() and text.strip()[0].isupper():
                return 'entity'
            return 'other'
        
        pred_type = get_type(pred)
        truth_type = get_type(truth)
        
        return pred_type != truth_type and pred_type != 'other' and truth_type != 'other'
    
    def _is_wrong_entity(
        self,
        pred: str,
        truth: str,
        context: Optional[str]
    ) -> bool:
        """Check if prediction is wrong entity of same type."""
        # Both should be entities (capitalized)
        pred_stripped = pred.strip()
        truth_stripped = truth.strip()
        
        if not pred_stripped or not truth_stripped:
            return False
        
        pred_is_entity = pred_stripped[0].isupper()
        truth_is_entity = truth_stripped[0].isupper()
        
        if pred_is_entity and truth_is_entity:
            # Different entities
            if self._normalize(pred) != self._normalize(truth):
                # Check if pred is also in context (valid entity, just wrong)
                if context and pred_stripped in context:
                    return True
        
        return False


def analyze_experiment_results(results_file: str) -> Dict:
    """
    Analyze results from an experiment file.
    
    Args:
        results_file: Path to results JSON file
        
    Returns:
        Error analysis dictionary
    """
    import json
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    metrics = results.get('metrics', {})
    predictions = metrics.get('predictions', [])
    ground_truths = metrics.get('ground_truths', [])
    
    if not predictions or not ground_truths:
        logger.warning("No predictions found in results file")
        return {}
    
    analyzer = ErrorAnalyzer()
    analysis = analyzer.analyze_batch(predictions, ground_truths)
    
    return analysis
