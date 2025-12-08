"""
Error Analysis Pipeline

Categorizes prediction failures to understand model weaknesses:
- wrong_entity: Correct type but wrong instance
- incomplete_extraction: Partial answer
- hallucination: Facts not in source
- format_error: Junk output
- semantic_drift: Meaning changed
"""

import re
from typing import Dict, List, Optional, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Analyzes prediction errors and categorizes failure types.
    
    Helps identify systematic issues in the agent pipeline
    to guide improvements.
    """
    
    ERROR_CATEGORIES = [
        'correct',           # Prediction matches ground truth
        'wrong_entity',      # Wrong but related entity
        'incomplete',        # Partial answer, missing key info
        'hallucination',     # Contains facts not in source
        'format_error',      # Junk output (Confidence:, High, etc.)
        'semantic_drift',    # Answer about wrong topic
        'too_short',         # Answer too brief
        'too_long',          # Answer overly verbose
    ]
    
    def __init__(self):
        """Initialize the error analyzer."""
        self.results: List[Dict] = []
        self.category_counts = Counter()
    
    def analyze(
        self,
        prediction: str,
        ground_truth: str,
        source_context: str,
        question: str = ""
    ) -> Dict:
        """
        Analyze a single prediction and categorize any error.
        
        Args:
            prediction: Model's prediction
            ground_truth: Expected answer
            source_context: Original source document
            question: The question asked
            
        Returns:
            Analysis dict with category and details
        """
        result = {
            'prediction': prediction[:200],
            'ground_truth': ground_truth[:200],
            'category': 'correct',
            'issues': [],
            'f1_score': self._compute_f1(prediction, ground_truth)
        }
        
        # Check for format errors first (junk output)
        if self._is_format_error(prediction):
            result['category'] = 'format_error'
            result['issues'].append('Output contains meta-commentary or junk')
            self._record(result)
            return result
        
        # Check if too short
        if len(prediction.split()) < 5:
            result['category'] = 'too_short'
            result['issues'].append(f'Only {len(prediction.split())} words')
            self._record(result)
            return result
        
        # High F1 = likely correct
        if result['f1_score'] > 0.5:
            result['category'] = 'correct'
            self._record(result)
            return result
        
        # Check for hallucination
        hallucination_score = self._check_hallucination(prediction, source_context)
        if hallucination_score > 0.5:
            result['category'] = 'hallucination'
            result['issues'].append(f'Hallucination score: {hallucination_score:.2f}')
            self._record(result)
            return result
        
        # Check for incomplete extraction
        if self._is_incomplete(prediction, ground_truth):
            result['category'] = 'incomplete'
            result['issues'].append('Missing key information from ground truth')
            self._record(result)
            return result
        
        # Check for wrong entity
        if self._is_wrong_entity(prediction, ground_truth):
            result['category'] = 'wrong_entity'
            result['issues'].append('Mentions different entity than expected')
            self._record(result)
            return result
        
        # Check for semantic drift
        if self._is_semantic_drift(prediction, ground_truth, question):
            result['category'] = 'semantic_drift'
            result['issues'].append('Answer seems off-topic')
            self._record(result)
            return result
        
        # Default to incomplete if low F1
        result['category'] = 'incomplete'
        result['issues'].append(f'Low F1 score: {result["f1_score"]:.2f}')
        self._record(result)
        return result
    
    def _record(self, result: Dict) -> None:
        """Record analysis result."""
        self.results.append(result)
        self.category_counts[result['category']] += 1
    
    def _is_format_error(self, text: str) -> bool:
        """Check if output is junk/meta-commentary."""
        junk_patterns = [
            r'^Confidence:\s*(High|Medium|Low)',
            r'^(High|Medium|Low)\s*$',
            r'^A\.\s*$',
            r'^[A-D]\.\s*To\s+',
            r'^Summary:\s*$',
            r'^Answer:\s*$',
        ]
        
        for pattern in junk_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True
        
        # Single word answers that are meta
        if text.strip().lower() in ['high', 'medium', 'low', 'yes', 'no']:
            return True
        
        return False
    
    def _check_hallucination(self, prediction: str, source: str) -> float:
        """
        Check for hallucinated facts.
        
        Returns score 0-1 where higher = more hallucination.
        """
        pred_words = set(self._normalize(prediction).split())
        source_words = set(self._normalize(source).split())
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
        pred_words -= stopwords
        source_words -= stopwords
        
        if not pred_words:
            return 0.0
        
        # Words in prediction but not in source
        hallucinated = pred_words - source_words
        
        # Filter out common words
        hallucinated = {w for w in hallucinated if len(w) > 3}
        
        return len(hallucinated) / len(pred_words)
    
    def _is_incomplete(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction is incomplete."""
        gt_words = set(self._normalize(ground_truth).split())
        pred_words = set(self._normalize(prediction).split())
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at'}
        gt_words -= stopwords
        pred_words -= stopwords
        
        if not gt_words:
            return False
        
        # Check coverage
        covered = len(gt_words & pred_words) / len(gt_words)
        return covered < 0.3
    
    def _is_wrong_entity(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction mentions wrong entity."""
        # Extract capitalized words (likely entities)
        pred_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', prediction))
        gt_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', ground_truth))
        
        if not gt_entities:
            return False
        
        # If entities exist but don't overlap
        if pred_entities and not (pred_entities & gt_entities):
            return True
        
        return False
    
    def _is_semantic_drift(self, prediction: str, ground_truth: str, question: str) -> bool:
        """Check if prediction is semantically off-topic."""
        # Simple check: does prediction relate to question keywords?
        question_words = set(self._normalize(question).split())
        pred_words = set(self._normalize(prediction).split())
        
        # Remove stopwords
        stopwords = {'what', 'who', 'where', 'when', 'how', 'the', 'a', 'is', 'are'}
        question_words -= stopwords
        
        if not question_words:
            return False
        
        overlap = len(question_words & pred_words) / len(question_words)
        return overlap < 0.1
    
    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = set(self._normalize(prediction).split())
        gt_tokens = set(self._normalize(ground_truth).split())
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        common = len(pred_tokens & gt_tokens)
        if common == 0:
            return 0.0
        
        precision = common / len(pred_tokens)
        recall = common / len(gt_tokens)
        
        return 2 * precision * recall / (precision + recall)
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        import string
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(text.split())
    
    def get_summary(self) -> Dict:
        """Get error analysis summary."""
        total = len(self.results)
        if total == 0:
            return {'total': 0, 'categories': {}}
        
        return {
            'total': total,
            'categories': dict(self.category_counts),
            'accuracy': self.category_counts.get('correct', 0) / total,
            'top_issues': self._get_top_issues()
        }
    
    def _get_top_issues(self) -> List[str]:
        """Get most common issues."""
        all_issues = []
        for r in self.results:
            all_issues.extend(r.get('issues', []))
        return [issue for issue, _ in Counter(all_issues).most_common(5)]
    
    def print_report(self) -> None:
        """Print error analysis report."""
        summary = self.get_summary()
        
        print("\n" + "=" * 60)
        print("ERROR ANALYSIS REPORT")
        print("=" * 60)
        print(f"\nTotal predictions: {summary['total']}")
        print(f"Accuracy (correct): {summary.get('accuracy', 0):.1%}")
        
        print("\nCategory breakdown:")
        for cat in self.ERROR_CATEGORIES:
            count = self.category_counts.get(cat, 0)
            pct = count / summary['total'] * 100 if summary['total'] > 0 else 0
            bar = '█' * int(pct / 5)
            print(f"  {cat:20s}: {count:3d} ({pct:5.1f}%) {bar}")
        
        print("\nTop issues:")
        for issue in summary.get('top_issues', [])[:5]:
            print(f"  - {issue}")
        
        print("=" * 60 + "\n")
    
    def export_results(self) -> List[Dict]:
        """Export all results for further analysis."""
        return self.results
