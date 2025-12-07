"""
Baseline Experiment Runner

Runs experiments for the Chain of Clarifications system.
Supports multiple compression strategies and comprehensive metric tracking.
"""

import argparse
import json
import os
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent_chain import AgentChain
from data.dataset_factory import get_loader, get_available_datasets
from utils.metrics import MetricsTracker
from utils.memory_tracker import MemoryTracker
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ExperimentRunner:
    """
    Runs experiments with different configurations.

    Configurations:
    - No compression (baseline)
    - Fixed compression (various ratios)
    - Role-specific compression
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: Optional[str] = None,
        output_dir: str = "results",
        dataset: str = "squad"
    ):
        """
        Initialize the experiment runner.

        Args:
            model_name: Language model to use
            device: Computing device
            output_dir: Directory to save results
            dataset: Dataset to use (squad, hotpotqa, drop)
        """
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_name = dataset

        # Initialize data loader using factory
        self.data_loader = get_loader(dataset)

        # Initialize tracking
        self.memory_tracker = MemoryTracker()

    def run_experiment(
        self,
        num_examples: int = 10,
        compression_type: str = "none",
        compression_ratio: float = 0.5,
        experiment_name: Optional[str] = None
    ) -> Dict:
        """
        Run an experiment with specified configuration.

        Args:
            num_examples: Number of examples to process
            compression_type: Type of compression (none, fixed, role_specific)
            compression_ratio: Compression ratio
            experiment_name: Name for this experiment

        Returns:
            Dictionary with results
        """
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{compression_type}_{compression_ratio}_{timestamp}"

        logger.info("=" * 80)
        logger.info(f"EXPERIMENT: {experiment_name}")
        logger.info(f"Compression: {compression_type} ({compression_ratio})")
        logger.info(f"Examples: {num_examples}")
        logger.info("=" * 80)

        # Initialize metrics
        metrics_tracker = MetricsTracker()

        # Initialize agent chain
        logger.info("Initializing agent chain...")
        agent_chain = AgentChain(
            model_name=self.model_name,
            device=self.device,
            compression_type=compression_type,
            compression_ratio=compression_ratio
        )

        # Load examples
        logger.info("Loading dataset...")
        examples = self.data_loader.get_examples(
            num_examples=num_examples,
            split="validation"
        )

        # Process examples
        logger.info(f"Processing {len(examples)} examples...")
        successful = 0
        failed = 0

        for idx, example in enumerate(examples):
            logger.info(f"\n{'='*60}")
            logger.info(f"Example {idx + 1}/{len(examples)}")
            logger.info(f"{'='*60}")

            question = example['question']
            context = example['context']
            ground_truth = example['answer_text']

            logger.info(f"Q: {question}")
            logger.info(f"GT: {ground_truth}")

            # Track memory before processing
            self.memory_tracker.log_memory(f"before_ex_{idx}")

            try:
                # Process through agent chain
                result = agent_chain.process(
                    question=question,
                    document=context,
                    track_metrics=True
                )

                # Track memory after processing
                self.memory_tracker.log_memory(f"after_ex_{idx}")

                if result['success']:
                    prediction = result['final_answer']
                    logger.info(f"Prediction: {prediction}")

                    # Add to metrics
                    metrics_tracker.add_result(
                        prediction=prediction,
                        ground_truth=ground_truth,
                        context_sizes=result.get('context_sizes', {}),
                        latency=result.get('latency', 0),
                        success=True,
                        memory_info=result.get('memory_usage', {})
                    )
                    successful += 1
                else:
                    logger.error(f"Failed: {result.get('error', 'Unknown error')}")
                    metrics_tracker.add_result(
                        prediction="",
                        ground_truth=ground_truth,
                        success=False,
                        error=result.get('error', 'Unknown')
                    )
                    failed += 1

            except Exception as e:
                logger.error(f"Exception processing example: {e}", exc_info=True)
                metrics_tracker.add_result(
                    prediction="",
                    ground_truth=ground_truth,
                    success=False,
                    error=str(e)
                )
                failed += 1

            # Clear GPU cache periodically
            if (idx + 1) % 5 == 0:
                self.memory_tracker.clear_gpu_cache()

        # Print summaries
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("=" * 80)

        metrics_tracker.print_summary()
        self.memory_tracker.print_memory_summary()

        logger.info(f"\nSuccessful: {successful}/{len(examples)}")
        logger.info(f"Failed: {failed}/{len(examples)}")

        # Cleanup
        agent_chain.cleanup()

        # Save results
        results = {
            'experiment_name': experiment_name,
            'configuration': {
                'model_name': self.model_name,
                'compression_type': compression_type,
                'compression_ratio': compression_ratio,
                'num_examples': num_examples,
            },
            'metrics': metrics_tracker.export_to_dict(),
            'memory': {
                'peak': self.memory_tracker.get_peak_memory(),
                'measurements': self.memory_tracker.get_all_measurements()
            },
            'counts': {
                'successful': successful,
                'failed': failed,
                'total': len(examples)
            }
        }

        self._save_results(results, experiment_name)

        return results

    def run_compression_comparison(
        self,
        num_examples: int = 100,
        compression_ratios: List[float] = [0.25, 0.5, 0.75, 1.0]
    ) -> Dict:
        """
        Run comparison across multiple compression ratios.

        Args:
            num_examples: Number of examples per configuration
            compression_ratios: List of compression ratios to test

        Returns:
            Dictionary with all results
        """
        all_results = {}

        # No compression baseline
        logger.info("\n" + "="*80)
        logger.info("BASELINE: No Compression")
        logger.info("="*80)
        all_results['no_compression'] = self.run_experiment(
            num_examples=num_examples,
            compression_type="none",
            experiment_name="baseline_no_compression"
        )

        # Fixed compression at different ratios
        for ratio in compression_ratios:
            if ratio >= 1.0:
                continue  # Skip 100% (same as no compression)

            logger.info("\n" + "="*80)
            logger.info(f"FIXED COMPRESSION: {ratio*100}%")
            logger.info("="*80)

            all_results[f'fixed_{ratio}'] = self.run_experiment(
                num_examples=num_examples,
                compression_type="fixed",
                compression_ratio=ratio,
                experiment_name=f"fixed_compression_{int(ratio*100)}"
            )

        # Role-specific compression
        for ratio in compression_ratios:
            if ratio >= 1.0:
                continue

            logger.info("\n" + "="*80)
            logger.info(f"ROLE-SPECIFIC COMPRESSION: {ratio*100}%")
            logger.info("="*80)

            all_results[f'role_specific_{ratio}'] = self.run_experiment(
                num_examples=num_examples,
                compression_type="role_specific",
                compression_ratio=ratio,
                experiment_name=f"role_specific_{int(ratio*100)}"
            )

        # Save comparison results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = self.output_dir / f"comparison_{timestamp}.json"

        with open(comparison_file, 'w') as f:
            json.dump(all_results, f, indent=2, cls=NumpyEncoder)

        logger.info(f"\nComparison results saved to: {comparison_file}")

        return all_results

    def _save_results(self, results: Dict, experiment_name: str):
        """Save experiment results to JSON file."""
        output_file = self.output_dir / f"{experiment_name}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        # Also save a human-readable summary
        self._save_readable_summary(results, experiment_name)
    
    def _save_readable_summary(self, results: Dict, experiment_name: str):
        """Save a human-readable summary of results."""
        summary_file = self.output_dir / f"{experiment_name}_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"EXPERIMENT SUMMARY: {experiment_name}\n")
            f.write("=" * 60 + "\n\n")
            
            # Configuration
            config = results.get('configuration', {})
            f.write("CONFIGURATION:\n")
            f.write(f"  Model: {config.get('model_name', 'N/A')}\n")
            f.write(f"  Compression Type: {config.get('compression_type', 'N/A')}\n")
            f.write(f"  Compression Ratio: {config.get('compression_ratio', 'N/A')}\n")
            f.write(f"  Number of Examples: {config.get('num_examples', 'N/A')}\n\n")
            
            # Counts
            counts = results.get('counts', {})
            f.write("RESULTS:\n")
            f.write(f"  Successful: {counts.get('successful', 0)}/{counts.get('total', 0)}\n")
            f.write(f"  Failed: {counts.get('failed', 0)}/{counts.get('total', 0)}\n\n")
            
            # Metrics
            metrics = results.get('metrics', {})
            f.write("METRICS:\n")
            
            if 'f1' in metrics:
                f1 = metrics['f1']
                f.write(f"  F1 Score:\n")
                f.write(f"    Mean: {f1.get('mean', 0):.4f}\n")
                f.write(f"    Std:  {f1.get('std', 0):.4f}\n")
            
            if 'exact_match' in metrics:
                em = metrics['exact_match']
                f.write(f"  Exact Match:\n")
                f.write(f"    Mean: {em.get('mean', 0):.4f}\n")
                f.write(f"    Percentage: {em.get('mean', 0) * 100:.2f}%\n")
            
            if 'latency' in metrics:
                lat = metrics['latency']
                f.write(f"  Latency:\n")
                f.write(f"    Mean: {lat.get('mean', 0):.2f}s\n")
            
            # Memory
            memory = results.get('memory', {})
            if memory:
                f.write(f"\nMEMORY:\n")
                peak = memory.get('peak', {})
                f.write(f"  Peak GPU: {peak.get('gpu_allocated_mb', 0):.1f} MB\n")
                f.write(f"  Peak RAM: {peak.get('ram_used_mb', 0):.1f} MB\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        logger.info(f"Summary saved to: {summary_file}")

        logger.info(f"\nResults saved to: {output_file}")


def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(
        description="Run Chain of Clarifications experiments"
    )

    parser.add_argument(
        '--num_examples',
        type=int,
        default=10,
        help='Number of examples to process'
    )

    parser.add_argument(
        '--compression_type',
        type=str,
        default='none',
        choices=['none', 'fixed', 'role_specific', 'dynamic', 'semantic'],
        help='Type of compression to use'
    )

    parser.add_argument(
        '--compression_ratio',
        type=float,
        default=0.5,
        help='Compression ratio (0.0 to 1.0)'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default='google/flan-t5-base',
        help='Model name from Hugging Face'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results'
    )

    parser.add_argument(
        '--comparison',
        action='store_true',
        help='Run full compression comparison'
    )

    parser.add_argument(
        '--track_memory',
        action='store_true',
        help='Enable detailed memory tracking'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='squad',
        choices=['squad', 'hotpotqa', 'drop'],
        help='Dataset to use for evaluation'
    )

    args = parser.parse_args()

    # Initialize runner
    runner = ExperimentRunner(
        model_name=args.model_name,
        output_dir=args.output_dir,
        dataset=args.dataset
    )

    # Run experiment(s)
    if args.comparison:
        logger.info("Running full compression comparison...")
        runner.run_compression_comparison(
            num_examples=args.num_examples
        )
    else:
        logger.info("Running single experiment...")
        runner.run_experiment(
            num_examples=args.num_examples,
            compression_type=args.compression_type,
            compression_ratio=args.compression_ratio
        )


if __name__ == "__main__":
    main()
