"""
ELI5 Dataset Loader

Loads ELI5 (Explain Like I'm 5) long-form QA dataset.
Questions require explanatory multi-sentence answers.
"""

from datasets import load_dataset
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ELI5Loader:
    """
    Loader for ELI5 long-form QA dataset.
    
    ELI5 provides:
    - Complex questions requiring explanation
    - Long-form answers (2-5 sentences)
    - Ideal for testing compression on reasoning chains
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the ELI5 loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        self.dataset = None
    
    def load(self, split: str = "validation") -> None:
        """
        Load the ELI5 dataset.
        
        Args:
            split: Which split to load
        """
        try:
            logger.info(f"Loading ELI5 dataset ({split} split)...")
            
            # Map validation to validation_asks for ELI5
            eli5_split = "validation_asks" if split == "validation" else split
            
            self.dataset = load_dataset(
                "eli5",
                split=eli5_split,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            logger.info(f"Loaded {len(self.dataset)} examples")
            
        except Exception as e:
            logger.error(f"Failed to load ELI5 dataset: {e}")
            # Fallback: try alternative loading
            try:
                logger.info("Trying alternative ELI5 loading...")
                self.dataset = load_dataset(
                    "eli5_category",
                    split="validation" if split == "validation" else split,
                    cache_dir=self.cache_dir,
                    trust_remote_code=True
                )
                logger.info(f"Loaded {len(self.dataset)} examples (alternative)")
            except Exception as e2:
                logger.error(f"Alternative loading also failed: {e2}")
                raise
    
    def get_examples(
        self,
        num_examples: Optional[int] = None,
        start_idx: int = 0,
        split: str = "validation"
    ) -> List[Dict]:
        """
        Get examples from the dataset.
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
            processed = self._preprocess_example(example)
            if processed:  # Skip if preprocessing fails
                examples.append(processed)
        
        logger.info(f"Retrieved {len(examples)} examples from ELI5")
        return examples
    
    def _preprocess_example(self, example: Dict) -> Optional[Dict]:
        """
        Preprocess a single ELI5 example.
        
        Converts to format compatible with agent chain.
        """
        try:
            title = example.get('title', '')
            selftext = example.get('selftext', '')
            
            # Question is the title
            question = title
            
            # Context is the selftext (additional details)
            context = selftext if selftext else "Please provide a detailed explanation."
            
            # Get the best answer (highest score)
            answers = example.get('answers', {})
            answer_texts = answers.get('text', [])
            answer_scores = answers.get('score', [])
            
            if answer_texts:
                # Get highest scored answer
                if answer_scores:
                    best_idx = answer_scores.index(max(answer_scores))
                    best_answer = answer_texts[best_idx]
                else:
                    best_answer = answer_texts[0]
            else:
                return None  # Skip examples without answers
            
            # Truncate very long answers
            words = best_answer.split()
            if len(words) > 200:
                best_answer = ' '.join(words[:200]) + '...'
            
            processed = {
                'id': example.get('q_id', ''),
                'question': question,
                'context': context,
                'answer_text': best_answer,
                'answer_start': 0,
                'title': title,
                'type': 'long_form_qa',
                'dataset': 'eli5',
                'answer_length': len(words)
            }
            
            return processed
            
        except Exception as e:
            logger.warning(f"Error preprocessing example: {e}")
            return None
    
    def get_sample_for_testing(self, num_samples: int = 10) -> List[Dict]:
        """Get a small sample for testing."""
        return self.get_examples(num_examples=num_samples, split="validation")
    
    def get_statistics(self, split: str = "validation", sample_size: int = 100) -> Dict:
        """Get dataset statistics."""
        examples = self.get_examples(num_examples=sample_size, split=split)
        
        if not examples:
            return {'error': 'No examples loaded', 'dataset': 'eli5'}
        
        question_lengths = [len(ex['question'].split()) for ex in examples]
        answer_lengths = [len(ex['answer_text'].split()) for ex in examples]
        
        stats = {
            'num_examples_sampled': len(examples),
            'avg_question_words': sum(question_lengths) / len(question_lengths),
            'avg_answer_words': sum(answer_lengths) / len(answer_lengths),
            'max_answer_words': max(answer_lengths),
            'dataset': 'eli5'
        }
        
        return stats
