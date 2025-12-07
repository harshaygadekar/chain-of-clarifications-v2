"""
CNN/DailyMail Dataset Loader

Loads CNN/DailyMail summarization dataset for evaluation.
Articles are summarized into 2-4 sentence highlights.
"""

from datasets import load_dataset
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class CNNDailyMailLoader:
    """
    Loader for CNN/DailyMail summarization dataset.
    
    This dataset provides:
    - Long articles (500-1000 words)
    - Multi-sentence summaries (2-4 sentences)
    - Ideal for testing context compression
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the CNN/DailyMail loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        self.dataset = None
    
    def load(self, split: str = "validation") -> None:
        """
        Load the CNN/DailyMail dataset.
        
        Args:
            split: Which split to load ('train', 'validation', or 'test')
        """
        try:
            logger.info(f"Loading CNN/DailyMail dataset ({split} split)...")
            
            self.dataset = load_dataset(
                "cnn_dailymail",
                "3.0.0",
                split=split,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Loaded {len(self.dataset)} examples")
            
        except Exception as e:
            logger.error(f"Failed to load CNN/DailyMail dataset: {e}")
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
        
        logger.info(f"Retrieved {len(examples)} examples from CNN/DailyMail")
        return examples
    
    def _preprocess_example(self, example: Dict) -> Dict:
        """
        Preprocess a single CNN/DailyMail example.
        
        Converts to format compatible with agent chain:
        - question -> "Summarize this article"
        - context -> article text
        - answer_text -> highlights (summary)
        """
        article = example.get('article', '')
        highlights = example.get('highlights', '')
        
        # Truncate very long articles to ~1000 words
        words = article.split()
        if len(words) > 1000:
            article = ' '.join(words[:1000]) + '...'
        
        processed = {
            'id': example.get('id', ''),
            'question': 'Summarize the key points of this article in 2-3 sentences.',
            'context': article,
            'answer_text': highlights,
            'answer_start': 0,
            'title': '',
            'type': 'summarization',
            'dataset': 'cnn_dailymail',
            'article_length': len(words)
        }
        
        return processed
    
    def get_sample_for_testing(self, num_samples: int = 10) -> List[Dict]:
        """Get a small sample for testing."""
        return self.get_examples(num_examples=num_samples, split="validation")
    
    def get_statistics(self, split: str = "validation", sample_size: int = 100) -> Dict:
        """
        Get dataset statistics.
        
        Args:
            split: Dataset split
            sample_size: Number of examples to sample for stats
        """
        examples = self.get_examples(num_examples=sample_size, split=split)
        
        article_lengths = [len(ex['context'].split()) for ex in examples]
        summary_lengths = [len(ex['answer_text'].split()) for ex in examples]
        
        stats = {
            'num_examples_sampled': len(examples),
            'avg_article_words': sum(article_lengths) / len(article_lengths),
            'avg_summary_words': sum(summary_lengths) / len(summary_lengths),
            'max_article_words': max(article_lengths),
            'max_summary_words': max(summary_lengths),
            'dataset': 'cnn_dailymail'
        }
        
        return stats
