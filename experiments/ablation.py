"""
Ablation Studies Framework

Systematically tests the contribution of different components
to overall system performance.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from copy import deepcopy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.agent_chain import AgentChain
from data.dataset_factory import get_loader
from utils.metrics import MetricsTracker
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Ablation configurations
ABLATION_CONFIGS = {
    'full': {
        'description': 'Full system with all components',
        'disable_keywords': False,
        'disable_entities': False,
        'disable_position': False,
        'use_role_specific': True,
    },
    'no_keyword_scoring': {
        'description': 'Disable keyword overlap scoring',
        'disable_keywords': True,
        'disable_entities': False,
        'disable_position': False,
        'use_role_specific': True,
    },
    'no_entity_scoring': {
        'description': 'Disable entity-based scoring',
        'disable_keywords': False,
        'disable_entities': True,
        'disable_position': False,
        'use_role_specific': True,
    },
    'no_position_scoring': {
        'description': 'Disable position-based scoring',
        'disable_keywords': False,
        'disable_entities': False,
        'disable_position': True,
        'use_role_specific': True,
    },
    'no_role_awareness': {
        'description': 'Use same scoring for all roles (no role-specific)',
        'disable_keywords': False,
        'disable_entities': False,
        'disable_position': False,
        'use_role_specific': False,
    },
    'only_keywords': {
        'description': 'Only use keyword scoring',
        'disable_keywords': False,
        'disable_entities': True,
        'disable_position': True,
        'use_role_specific': True,
    },
    'only_entities': {
        'description': 'Only use entity scoring',
        'disable_keywords': True,
        'disable_entities': False,
        'disable_position': True,
        'use_role_specific': True,
    },
    'only_position': {
        'description': 'Only use position scoring',
        'disable_keywords': True,
        'disable_entities': True,
        'disable_position': False,
        'use_role_specific': True,
    },
    'fixed_compression': {
        'description': 'Fixed compression (no adaptive)',
        'disable_keywords': False,
        'disable_entities': False,
        'disable_position': False,
        'use_role_specific': False,
        'compression_type': 'fixed',
    },
}


class AblationRunner:
    """
    Runs ablation studies to measure component contributions.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-mini-instruct",
        output_dir: str = "results/ablations",
        dataset: str = "squad"
    ):
        """
        Initialize ablation runner.
        
        Args:
            model_name: Model to use
            output_dir: Directory for results
            dataset: Dataset to use
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        
        # Load data
        self.data_loader = get_loader(dataset)
    
    def run_ablation(
        self,
        config_name: str,
        num_examples: int = 50,
        compression_ratio: float = 0.5
    ) -> Dict:
        """
        Run a single ablation configuration.
        
        Args:
            config_name: Name of ablation config
            num_examples: Number of examples to process
            compression_ratio: Compression ratio
            
        Returns:
            Results dictionary
        """
        if config_name not in ABLATION_CONFIGS:
            raise ValueError(f"Unknown config: {config_name}. Available: {list(ABLATION_CONFIGS.keys())}")
        
        config = ABLATION_CONFIGS[config_name]
        
        logger.info("=" * 80)
        logger.info(f"ABLATION: {config_name}")
        logger.info(f"Description: {config['description']}")
        logger.info("=" * 80)
        
        # Determine compression type
        compression_type = config.get('compression_type', 'role_specific')
        if not config.get('use_role_specific', True):
            compression_type = 'fixed'
        
        # Initialize agent chain
        # Note: To fully implement ablation, we'd need to modify RoleSpecificScorer
        # to accept disable_* flags. For now, we use compression_type as proxy.
        agent_chain = AgentChain(
            model_name=self.model_name,
            compression_type=compression_type,
            compression_ratio=compression_ratio
        )
        
        # Initialize metrics
        metrics_tracker = MetricsTracker()
        
        # Load examples
        examples = self.data_loader.get_examples(
            num_examples=num_examples,
            split="validation"
        )
        
        # Process examples
        successful = 0
        failed = 0
        
        for idx, example in enumerate(examples):
            logger.info(f"Processing {idx + 1}/{len(examples)}")
            
            try:
                result = agent_chain.process(
                    question=example['question'],
                    document=example['context'],
                    track_metrics=True
                )
                
                if result['success']:
                    metrics_tracker.add_result(
                        prediction=result['final_answer'],
                        ground_truth=example['answer_text'],
                        context_sizes=result.get('context_sizes', {}),
                        latency=result.get('latency', 0),
                        success=True
                    )
                    successful += 1
                else:
                    metrics_tracker.add_result(
                        prediction="",
                        ground_truth=example['answer_text'],
                        success=False,
                        error=result.get('error', 'Unknown')
                    )
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error: {e}")
                failed += 1
        
        # Cleanup
        agent_chain.cleanup()
        
        # Compile results
        results = {
            'ablation_name': config_name,
            'description': config['description'],
            'config': config,
            'dataset': self.dataset,
            'num_examples': num_examples,
            'compression_ratio': compression_ratio,
            'metrics': metrics_tracker.export_to_dict(),
            'counts': {
                'successful': successful,
                'failed': failed,
                'total': len(examples)
            }
        }
        
        # Save results
        self._save_results(results, config_name)
        
        return results
    
    def run_all_ablations(
        self,
        num_examples: int = 50,
        compression_ratio: float = 0.5,
        configs: Optional[List[str]] = None
    ) -> Dict:
        """
        Run all ablation configurations.
        
        Args:
            num_examples: Examples per configuration
            compression_ratio: Compression ratio
            configs: Specific configs to run (None for all)
            
        Returns:
            Dictionary with all results
        """
        if configs is None:
            configs = list(ABLATION_CONFIGS.keys())
        
        all_results = {}
        
        for config_name in configs:
            try:
                results = self.run_ablation(
                    config_name=config_name,
                    num_examples=num_examples,
                    compression_ratio=compression_ratio
                )
                all_results[config_name] = results
            except Exception as e:
                logger.error(f"Failed ablation {config_name}: {e}")
                all_results[config_name] = {'error': str(e)}
        
        # Save combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = self.output_dir / f"all_ablations_{timestamp}.json"
        
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"All ablation results saved to: {combined_file}")
        
        # Print summary
        self._print_ablation_summary(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict, config_name: str):
        """Save ablation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"ablation_{config_name}_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
    
    def _print_ablation_summary(self, all_results: Dict):
        """Print summary of ablation results."""
        print("\n" + "=" * 80)
        print("ABLATION STUDY SUMMARY")
        print("=" * 80)
        
        print(f"\n{'Configuration':<25} {'F1 Mean':<10} {'EM Mean':<10} {'Success':<10}")
        print("-" * 55)
        
        # Sort by F1 score
        sorted_results = []
        for name, result in all_results.items():
            if 'error' not in result:
                metrics = result.get('metrics', {}).get('summary', {})
                f1 = metrics.get('f1_mean', 0)
                em = metrics.get('em_mean', 0)
                success = metrics.get('success_rate', 0)
                sorted_results.append((name, f1, em, success))
        
        sorted_results.sort(key=lambda x: x[1], reverse=True)
        
        for name, f1, em, success in sorted_results:
            print(f"{name:<25} {f1:<10.4f} {em:<10.4f} {success:<10.2%}")
        
        print("=" * 80)
    
    def generate_ablation_report(self, results: Dict) -> str:
        """
        Generate text report of ablation study.
        
        Args:
            results: Ablation results dictionary
            
        Returns:
            Report string
        """
        lines = []
        lines.append("# Ablation Study Report\n")
        lines.append(f"Dataset: {self.dataset}")
        lines.append(f"Model: {self.model_name}\n")
        
        # Find baseline (full config)
        baseline_f1 = 0
        if 'full' in results and 'error' not in results['full']:
            baseline_f1 = results['full'].get('metrics', {}).get('summary', {}).get('f1_mean', 0)
            lines.append(f"Baseline F1 (full): {baseline_f1:.4f}\n")
        
        lines.append("## Component Contributions\n")
        lines.append("| Component Removed | F1 Score | Δ F1 | Impact |")
        lines.append("|-------------------|----------|------|--------|")
        
        for name, result in results.items():
            if name == 'full' or 'error' in result:
                continue
            
            f1 = result.get('metrics', {}).get('summary', {}).get('f1_mean', 0)
            delta = f1 - baseline_f1
            impact = "Positive" if delta > 0 else "Negative" if delta < 0 else "Neutral"
            
            lines.append(f"| {name:<17} | {f1:.4f}   | {delta:+.4f} | {impact} |")
        
        return '\n'.join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run ablation studies")
    
    parser.add_argument(
        '--num_examples',
        type=int,
        default=50,
        help='Number of examples per ablation'
    )
    
    parser.add_argument(
        '--ablation',
        type=str,
        default=None,
        help='Specific ablation to run'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all ablations'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='squad',
        help='Dataset to use'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default='microsoft/Phi-3.5-mini-instruct',
        help='Model to use'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/ablations',
        help='Output directory'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available ablation configs'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable ablation configurations:")
        for name, config in ABLATION_CONFIGS.items():
            print(f"  {name}: {config['description']}")
        return
    
    runner = AblationRunner(
        model_name=args.model_name,
        output_dir=args.output_dir,
        dataset=args.dataset
    )
    
    if args.all:
        runner.run_all_ablations(num_examples=args.num_examples)
    elif args.ablation:
        runner.run_ablation(
            config_name=args.ablation,
            num_examples=args.num_examples
        )
    else:
        print("Specify --ablation <name> or --all. Use --list to see options.")


if __name__ == "__main__":
    main()
