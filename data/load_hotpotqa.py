"""
HotpotQA Dataset Loader

Loads HotpotQA multi-hop reasoning dataset for evaluation.
HotpotQA requires reasoning over multiple documents to answer questions.
"""

from datasets import load_dataset
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class HotpotQALoader:
    """
    Loader for HotpotQA dataset.
    
    HotpotQA is a multi-hop QA dataset that requires:
    - Finding and reasoning over multiple supporting facts
    - Cross-document reasoning
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the HotpotQA loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        self.dataset = None
    
    def load(self, split: str = "validation") -> None:
        """
        Load the HotpotQA dataset.
        
        Args:
            split: Which split to load ('train' or 'validation')
        """
        try:
            logger.info(f"Loading HotpotQA dataset ({split} split)...")
            
            # Load distractor setting (more challenging)
            self.dataset = load_dataset(
                "hotpot_qa",
                "distractor",
                split=split,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Loaded {len(self.dataset)} examples")
            
        except Exception as e:
            logger.error(f"Failed to load HotpotQA dataset: {e}")
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
        
        logger.info(f"Retrieved {len(examples)} examples from HotpotQA")
        return examples
    
    def _preprocess_example(self, example: Dict) -> Dict:
        """
        Preprocess a single HotpotQA example.
        
        Args:
            example: Raw example from dataset
            
        Returns:
            Preprocessed example matching SQuAD format
        """
        # Combine all context paragraphs
        context_parts = []
        context_data = example.get('context', {})
        
        titles = context_data.get('title', [])
        sentences_list = context_data.get('sentences', [])
        
        for title, sentences in zip(titles, sentences_list):
            para = ' '.join(sentences)
            context_parts.append(f"{title}: {para}")
        
        combined_context = '\n\n'.join(context_parts)
        
        # Get supporting facts info
        supporting_facts = example.get('supporting_facts', {})
        support_titles = supporting_facts.get('title', [])
        support_sent_ids = supporting_facts.get('sent_id', [])
        
        processed = {
            'id': example.get('id', ''),
            'question': example.get('question', ''),
            'context': combined_context,
            'answer_text': example.get('answer', ''),
            'answer_start': 0,  # Not applicable for HotpotQA
            'title': ', '.join(titles) if titles else '',
            'type': example.get('type', ''),
            'level': example.get('level', ''),
            'supporting_facts': {
                'titles': support_titles,
                'sent_ids': support_sent_ids
            },
            'dataset': 'hotpotqa'
        }
        
        return processed
    
    def get_sample_for_testing(self, num_samples: int = 10) -> List[Dict]:
        """
        Get a small sample for testing.
        
        Args:
            num_samples: Number of samples
            
        Returns:
            List of sample examples
        """
        return self.get_examples(num_examples=num_samples, split="validation")
    
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
        answer_lengths = [len(ex['answer_text']) for ex in examples]
        
        # Count question types
        types = {}
        levels = {}
        for ex in examples:
            t = ex.get('type', 'unknown')
            l = ex.get('level', 'unknown')
            types[t] = types.get(t, 0) + 1
            levels[l] = levels.get(l, 0) + 1
        
        stats = {
            'num_examples': len(examples),
            'avg_context_length': sum(context_lengths) / len(context_lengths),
            'avg_question_length': sum(question_lengths) / len(question_lengths),
            'avg_answer_length': sum(answer_lengths) / len(answer_lengths),
            'max_context_length': max(context_lengths),
            'question_types': types,
            'difficulty_levels': levels,
            'dataset': 'hotpotqa'
        }
        
        return stats
