"""
NarrativeQA Dataset Loader

Loads NarrativeQA - question answering on long narratives (books/stories).
Perfect for testing context compression on very long documents.
"""

from datasets import load_dataset
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class NarrativeQALoader:
    """
    Loader for NarrativeQA dataset - QA on long narratives.
    
    Features:
    - Long story contexts (summaries ~700 words)
    - Abstractive answers requiring comprehension
    - Ideal for testing compression on narrative text
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the loader."""
        self.cache_dir = cache_dir
        self.dataset = None
    
    def load(self, split: str = "validation") -> None:
        """Load the NarrativeQA dataset."""
        try:
            logger.info(f"Loading NarrativeQA dataset ({split} split)...")
            
            self.dataset = load_dataset(
                "deepmind/narrativeqa",
                split=split,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Loaded {len(self.dataset)} examples")
            
        except Exception as e:
            logger.error(f"Failed to load NarrativeQA dataset: {e}")
            raise
    
    def get_examples(
        self,
        num_examples: Optional[int] = None,
        start_idx: int = 0,
        split: str = "validation"
    ) -> List[Dict]:
        """Get examples from dataset."""
        if self.dataset is None:
            self.load(split)
        
        examples = []
        
        for idx, item in enumerate(self.dataset):
            if num_examples and len(examples) >= num_examples:
                break
            
            if idx < start_idx:
                continue
            
            # Use summary as context (full documents too long)
            document = item.get('document', {})
            summary = document.get('summary', {}).get('text', '')
            
            question = item.get('question', {}).get('text', '')
            
            # Get answers
            answers = item.get('answers', [])
            answer_text = answers[0].get('text', '') if answers else ''
            
            if question and summary and answer_text:
                examples.append({
                    'id': item.get('document', {}).get('id', str(idx)),
                    'question': question,
                    'context': summary[:6000],  # Truncate if needed
                    'answer_text': answer_text,
                    'answer_start': 0,
                    'title': document.get('title', ''),
                    'type': 'narrative_qa',
                    'dataset': 'narrativeqa'
                })
        
        logger.info(f"Retrieved {len(examples)} examples from NarrativeQA")
        return examples
    
    def get_sample_for_testing(self, num_samples: int = 10) -> List[Dict]:
        """Get a small sample for testing."""
        return self.get_examples(num_examples=num_samples, split="validation")
    
    def get_statistics(self, split: str = "validation", sample_size: int = 50) -> Dict:
        """Get dataset statistics."""
        examples = self.get_examples(num_examples=sample_size, split=split)
        
        context_lengths = [len(ex['context'].split()) for ex in examples]
        answer_lengths = [len(ex['answer_text'].split()) for ex in examples]
        
        return {
            'num_examples_sampled': len(examples),
            'avg_context_words': sum(context_lengths) / len(context_lengths) if context_lengths else 0,
            'avg_answer_words': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
            'max_context_words': max(context_lengths) if context_lengths else 0,
            'dataset': 'narrativeqa'
        }
