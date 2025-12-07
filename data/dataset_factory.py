"""
Dataset Factory

Provides a unified interface for loading different QA datasets.
Supports SQuAD, HotpotQA, and DROP.
"""

from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

# Import loaders
from data.load_squad import SQuADLoader
from data.load_hotpotqa import HotpotQALoader
from data.load_drop import DROPLoader


# Type alias for any loader
DatasetLoader = Union[SQuADLoader, HotpotQALoader, DROPLoader]


def get_loader(
    dataset_name: str,
    cache_dir: Optional[str] = None
) -> DatasetLoader:
    """
    Get a dataset loader by name.
    
    Args:
        dataset_name: Name of the dataset (squad, hotpotqa, drop)
        cache_dir: Optional cache directory
        
    Returns:
        Appropriate dataset loader instance
        
    Raises:
        ValueError: If dataset name is not recognized
    """
    dataset_name = dataset_name.lower().strip()
    
    loaders = {
        'squad': SQuADLoader,
        'squad1.1': SQuADLoader,
        'squad-1.1': SQuADLoader,
        'hotpotqa': HotpotQALoader,
        'hotpot_qa': HotpotQALoader,
        'hotpot': HotpotQALoader,
        'drop': DROPLoader,
    }
    
    if dataset_name not in loaders:
        available = ['squad', 'hotpotqa', 'drop']
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {available}"
        )
    
    logger.info(f"Creating loader for dataset: {dataset_name}")
    return loaders[dataset_name](cache_dir=cache_dir)


def get_available_datasets() -> list:
    """
    Get list of available dataset names.
    
    Returns:
        List of dataset names
    """
    return ['squad', 'hotpotqa', 'drop']


class MultiDatasetLoader:
    """
    Loader that can combine examples from multiple datasets.
    
    Useful for comprehensive evaluation across different QA types.
    """
    
    def __init__(
        self,
        datasets: list = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize multi-dataset loader.
        
        Args:
            datasets: List of dataset names to include
            cache_dir: Cache directory
        """
        if datasets is None:
            datasets = ['squad', 'hotpotqa', 'drop']
        
        self.loaders = {}
        for ds_name in datasets:
            try:
                self.loaders[ds_name] = get_loader(ds_name, cache_dir)
            except Exception as e:
                logger.warning(f"Failed to create loader for {ds_name}: {e}")
    
    def get_examples(
        self,
        num_examples_per_dataset: int = 10,
        split: str = "validation"
    ) -> dict:
        """
        Get examples from all loaded datasets.
        
        Args:
            num_examples_per_dataset: Examples per dataset
            split: Dataset split
            
        Returns:
            Dictionary mapping dataset name to examples
        """
        all_examples = {}
        
        for ds_name, loader in self.loaders.items():
            try:
                examples = loader.get_examples(
                    num_examples=num_examples_per_dataset,
                    split=split
                )
                all_examples[ds_name] = examples
                logger.info(f"Loaded {len(examples)} examples from {ds_name}")
            except Exception as e:
                logger.error(f"Failed to load examples from {ds_name}: {e}")
                all_examples[ds_name] = []
        
        return all_examples
    
    def get_combined_examples(
        self,
        num_examples_per_dataset: int = 10,
        split: str = "validation"
    ) -> list:
        """
        Get combined list of examples from all datasets.
        
        Args:
            num_examples_per_dataset: Examples per dataset
            split: Dataset split
            
        Returns:
            Combined list of examples with dataset tag
        """
        all_examples = self.get_examples(num_examples_per_dataset, split)
        
        combined = []
        for ds_name, examples in all_examples.items():
            for ex in examples:
                ex['source_dataset'] = ds_name
                combined.append(ex)
        
        return combined
    
    def get_statistics(self, split: str = "validation") -> dict:
        """
        Get statistics for all datasets.
        
        Args:
            split: Dataset split
            
        Returns:
            Dictionary mapping dataset name to statistics
        """
        stats = {}
        
        for ds_name, loader in self.loaders.items():
            try:
                stats[ds_name] = loader.get_statistics(split)
            except Exception as e:
                logger.error(f"Failed to get stats for {ds_name}: {e}")
        
        return stats
