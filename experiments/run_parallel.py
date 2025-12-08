"""
Multi-GPU Experiment Runner

Runs experiments in parallel across multiple GPUs.
Usage: python experiments/run_parallel.py --num_examples 50
"""

import subprocess
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_experiment_on_gpu(args: tuple) -> dict:
    """Run a single experiment on specified GPU."""
    gpu_id, compression_type, dataset, num_examples = args
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd = [
        sys.executable, 'experiments/baseline.py',
        '--dataset', dataset,
        '--compression_type', compression_type,
        '--num_examples', str(num_examples)
    ]
    
    logger.info(f"GPU {gpu_id}: Starting {compression_type}...")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        success = result.returncode == 0
        logger.info(f"GPU {gpu_id}: {compression_type} {'✅' if success else '❌'}")
        
        return {
            'gpu': gpu_id,
            'compression_type': compression_type,
            'success': success,
            'stdout': result.stdout[-500:] if result.stdout else '',
            'stderr': result.stderr[-500:] if result.stderr else ''
        }
    except Exception as e:
        logger.error(f"GPU {gpu_id}: {compression_type} failed: {e}")
        return {
            'gpu': gpu_id,
            'compression_type': compression_type,
            'success': False,
            'error': str(e)
        }


def run_parallel_experiments(
    compression_types: list,
    dataset: str = 'narrativeqa',
    num_examples: int = 50,
    num_gpus: int = 2
):
    """Run experiments in parallel across GPUs."""
    
    logger.info(f"Running {len(compression_types)} experiments on {num_gpus} GPUs")
    logger.info(f"Dataset: {dataset}, Examples: {num_examples}")
    
    # Create tasks - alternate GPUs
    tasks = []
    for i, comp_type in enumerate(compression_types):
        gpu_id = i % num_gpus
        tasks.append((gpu_id, comp_type, dataset, num_examples))
    
    results = []
    
    # Run in parallel (2 at a time for 2 GPUs)
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {executor.submit(run_experiment_on_gpu, task): task for task in tasks}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            if result['success']:
                logger.info(f"✅ {result['compression_type']} completed on GPU {result['gpu']}")
            else:
                logger.error(f"❌ {result['compression_type']} failed on GPU {result['gpu']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PARALLEL EXECUTION SUMMARY")
    print("=" * 60)
    
    for r in results:
        status = "✅ Success" if r['success'] else "❌ Failed"
        print(f"GPU {r['gpu']}: {r['compression_type']:20s} - {status}")
    
    success_count = sum(1 for r in results if r['success'])
    print(f"\nCompleted: {success_count}/{len(results)}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run experiments in parallel on multiple GPUs")
    
    parser.add_argument('--dataset', type=str, default='narrativeqa')
    parser.add_argument('--num_examples', type=int, default=50)
    parser.add_argument('--num_gpus', type=int, default=2)
    parser.add_argument('--compression_types', type=str, nargs='+',
                       default=['none', 'fixed', 'role_specific', 'dynamic', 'semantic'])
    
    args = parser.parse_args()
    
    run_parallel_experiments(
        compression_types=args.compression_types,
        dataset=args.dataset,
        num_examples=args.num_examples,
        num_gpus=args.num_gpus
    )


if __name__ == "__main__":
    main()
