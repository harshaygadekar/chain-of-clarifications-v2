"""
QASPER Dataset Loader

Loads QASPER (Question Answering on Scientific Papers) dataset.
Features scientific paper QA with long contexts - ideal for compression research.
"""

from datasets import load_dataset
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class QASPERLoader:
    """
    Loader for QASPER dataset - Question Answering on Scientific Papers.
    
    This dataset provides:
    - Long scientific paper contexts (1000-5000 words)
    - Multi-hop reasoning questions
    - Extractive and abstractive answers
    - Ideal for demonstrating context compression benefits
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the QASPER loader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir
        self.dataset = None
    
    def load(self, split: str = "validation") -> None:
        """
        Load the QASPER dataset.
        
        Args:
            split: Which split to load ('train', 'validation', or 'test')
        """
        try:
            logger.info(f"Loading QASPER dataset ({split} split)...")
            
            self.dataset = load_dataset(
                "allenai/qasper",
                split=split,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Loaded {len(self.dataset)} papers")
            
        except Exception as e:
            logger.error(f"Failed to load QASPER dataset: {e}")
            raise
    
    def get_examples(
        self,
        num_examples: Optional[int] = None,
        start_idx: int = 0,
        split: str = "validation"
    ) -> List[Dict]:
        """
        Get examples from the dataset.
        
        QASPER has nested structure: papers contain multiple QA pairs.
        We flatten these into individual examples.
        
        Args:
            num_examples: Number of examples to return
            start_idx: Starting index
            split: Dataset split
            
        Returns:
            List of example dictionaries
        """
        if self.dataset is None:
            self.load(split)
        
        examples = []
        example_count = 0
        
        for paper in self.dataset:
            if num_examples and example_count >= num_examples:
                break
            
            # Extract paper context (full text from sections)
            full_text = self._extract_full_text(paper)
            title = paper.get('title', '')
            
            # Process each QA pair
            qas = paper.get('qas', [])
            for qa in qas:
                if num_examples and example_count >= num_examples:
                    break
                
                question = qa.get('question', '')
                answers = qa.get('answers', [])
                
                # Get the best answer (first extractive or abstractive)
                answer_text = self._get_best_answer(answers)
                
                if question and answer_text and full_text:
                    if example_count >= start_idx:
                        examples.append({
                            'id': f"{paper.get('id', '')}_{example_count}",
                            'question': question,
                            'context': full_text[:8000],  # Truncate very long papers
                            'answer_text': answer_text,
                            'answer_start': 0,
                            'title': title,
                            'type': 'scientific_qa',
                            'dataset': 'qasper'
                        })
                    example_count += 1
        
        logger.info(f"Retrieved {len(examples)} QA examples from QASPER")
        return examples
    
    def _extract_full_text(self, paper: Dict) -> str:
        """Extract full text from paper sections."""
        sections = []
        
        # Add abstract
        abstract = paper.get('abstract', '')
        if abstract:
            sections.append(f"Abstract: {abstract}")
        
        # Add section paragraphs
        full_text_list = paper.get('full_text', {})
        if isinstance(full_text_list, dict):
            section_names = full_text_list.get('section_name', [])
            paragraphs = full_text_list.get('paragraphs', [])
            
            for name, paras in zip(section_names, paragraphs):
                if name:
                    sections.append(f"\n{name}:")
                if isinstance(paras, list):
                    sections.extend(paras)
                elif paras:
                    sections.append(str(paras))
        
        return ' '.join(sections)
    
    def _get_best_answer(self, answers: List[Dict]) -> str:
        """Get the best answer from answer list."""
        if not answers:
            return ""
        
        for ans in answers:
            answer_obj = ans.get('answer', {})
            
            # Prefer extractive answers
            extractive = answer_obj.get('extractive_spans', [])
            if extractive:
                return ' '.join(extractive)
            
            # Fall back to free-form answer
            free_form = answer_obj.get('free_form_answer', '')
            if free_form and free_form.lower() not in ['yes', 'no', 'unanswerable']:
                return free_form
            
            # Yes/No answers
            yes_no = answer_obj.get('yes_no')
            if yes_no is not None:
                return 'Yes' if yes_no else 'No'
        
        return ""
    
    def get_sample_for_testing(self, num_samples: int = 10) -> List[Dict]:
        """Get a small sample for testing."""
        return self.get_examples(num_examples=num_samples, split="validation")
    
    def get_statistics(self, split: str = "validation", sample_size: int = 50) -> Dict:
        """Get dataset statistics."""
        examples = self.get_examples(num_examples=sample_size, split=split)
        
        context_lengths = [len(ex['context'].split()) for ex in examples]
        answer_lengths = [len(ex['answer_text'].split()) for ex in examples]
        question_lengths = [len(ex['question'].split()) for ex in examples]
        
        return {
            'num_examples_sampled': len(examples),
            'avg_context_words': sum(context_lengths) / len(context_lengths) if context_lengths else 0,
            'avg_answer_words': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
            'avg_question_words': sum(question_lengths) / len(question_lengths) if question_lengths else 0,
            'max_context_words': max(context_lengths) if context_lengths else 0,
            'dataset': 'qasper'
        }
