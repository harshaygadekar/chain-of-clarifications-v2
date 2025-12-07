"""
DROP Dataset Loader

Loads DROP discrete reasoning dataset for evaluation.
DROP requires numerical reasoning and counting over text.
"""

from datasets import load_dataset
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class DROPLoader:
    """
    Loader for DROP (Discrete Reasoning Over Paragraphs) dataset.
    
    DROP requires:
    - Numerical reasoning (addition, subtraction, counting)
    - Date comparisons
    - Sorting and ordering
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the DROP loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        self.dataset = None
    
    def load(self, split: str = "validation") -> None:
        """
        Load the DROP dataset.
        
        Args:
            split: Which split to load ('train' or 'validation')
        """
        try:
            logger.info(f"Loading DROP dataset ({split} split)...")
            
            self.dataset = load_dataset(
                "drop",
                split=split,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Loaded {len(self.dataset)} examples")
            
        except Exception as e:
            logger.error(f"Failed to load DROP dataset: {e}")
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
            num_examples: Number of examples to return
            start_idx: Starting index
            split: Dataset split
            
        Returns:
            List of example dictionaries
        """
        if self.dataset is None:
            self.load(split)
        
        if num_examples is None:
            end_idx = len(self.dataset)
        else:
            end_idx = min(start_idx + num_examples, len(self.dataset))
        
        examples = []
        for idx in range(start_idx, end_idx):
            example = self.dataset[idx]
            examples.append(self._preprocess_example(example))
        
        logger.info(f"Retrieved {len(examples)} examples from DROP")
        return examples
    
    def _preprocess_example(self, example: Dict) -> Dict:
        """
        Preprocess a single DROP example.
        
        Args:
            example: Raw example from dataset
            
        Returns:
            Preprocessed example matching SQuAD format
        """
        answer_text, answer_type, spans = self._extract_answer(example)
        
        return {
            'id': example.get('query_id', ''),
            'question': example.get('question', ''),
            'context': example.get('passage', ''),
            'answer_text': answer_text,
            'answer_start': 0,
            'title': '',
            'answer_type': answer_type,
            'all_answers': spans,
            'dataset': 'drop'
        }
    
    def _extract_answer(self, example: Dict) -> tuple:
        """Extract answer from DROP example."""
        answers_spans = example.get('answers_spans', {})
        spans = answers_spans.get('spans', [])
        
        # Try span answer first
        if spans:
            return spans[0], "span", spans
        
        # Try number answer
        number = example.get('answer', {}).get('number', '')
        if number:
            return str(number), "number", spans
        
        # Try date answer
        date_text = self._extract_date(example)
        if date_text:
            return date_text, "date", spans
        
        return "", "unknown", spans
    
    def _extract_date(self, example: Dict) -> str:
        """Extract date answer from DROP example."""
        date = example.get('answer', {}).get('date', {})
        if not date:
            return ""
        
        parts = []
        if date.get('day'):
            parts.append(str(date['day']))
        if date.get('month'):
            parts.append(date['month'])
        if date.get('year'):
            parts.append(str(date['year']))
        
        return ' '.join(parts) if parts else ""
    
    def get_sample_for_testing(self, num_samples: int = 10) -> List[Dict]:
        """
        Get a small sample for testing.
        
        Args:
            num_samples: Number of samples
            
        Returns:
            List of sample examples
        """
        return self.get_examples(num_examples=num_samples, split="validation")
    
    def compute_f1(self, prediction: str, ground_truth: str) -> float:
        """
        Compute F1 score - adapted for DROP's numeric answers.
        
        Args:
            prediction: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score
        """
        # Normalize both
        pred_norm = self._normalize_answer(prediction)
        truth_norm = self._normalize_answer(ground_truth)
        
        # For numeric answers, check exact match first
        try:
            pred_num = float(pred_norm.replace(',', ''))
            truth_num = float(truth_norm.replace(',', ''))
            if pred_num == truth_num:
                return 1.0
        except (ValueError, AttributeError):
            pass
        
        # Standard token F1
        pred_tokens = pred_norm.split()
        truth_tokens = truth_norm.split()
        
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
    
    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalize answer for comparison."""
        import re
        import string
        
        if not text:
            return ""
        
        text = str(text).lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
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
        
        context_lengths = [len(ex['context']) for ex in examples]
        question_lengths = [len(ex['question']) for ex in examples]
        
        # Count answer types
        answer_types = {}
        for ex in examples:
            t = ex.get('answer_type', 'unknown')
            answer_types[t] = answer_types.get(t, 0) + 1
        
        stats = {
            'num_examples': len(examples),
            'avg_context_length': sum(context_lengths) / len(context_lengths),
            'avg_question_length': sum(question_lengths) / len(question_lengths),
            'max_context_length': max(context_lengths),
            'answer_types': answer_types,
            'dataset': 'drop'
        }
        
        return stats
