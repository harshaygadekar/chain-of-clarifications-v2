"""
Results Analysis Script

Analyzes experimental results and generates visualizations.
Supports statistical testing and comparative analysis.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ResultsAnalyzer:
    """Analyzes and visualizes experimental results."""

    def __init__(self, results_dir: str = "results"):
        """
        Initialize the analyzer.

        Args:
            results_dir: Directory containing result JSON files
        """
        self.results_dir = Path(results_dir)
        self.results = {}

    def load_results(self, pattern: str = "*.json"):
        """
        Load all result files.

        Args:
            pattern: File pattern to match
        """
        result_files = list(self.results_dir.glob(pattern))
        print(f"Found {len(result_files)} result files")

        for file_path in result_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Handle both single and comparison results
                if 'experiment_name' in data:
                    self.results[data['experiment_name']] = data
                else:
                    # Comparison file
                    for key, value in data.items():
                        if isinstance(value, dict) and 'experiment_name' in value:
                            self.results[value['experiment_name']] = value

        print(f"Loaded {len(self.results)} experiments")

    def create_comparison_dataframe(self) -> pd.DataFrame:
        """
        Create DataFrame comparing all experiments.

        Returns:
            DataFrame with comparison metrics
        """
        rows = []

        for exp_name, exp_data in self.results.items():
            config = exp_data.get('configuration', {})
            metrics = exp_data.get('metrics', {})
            summary = metrics.get('summary', {})

            row = {
                'experiment': exp_name,
                'compression_type': config.get('compression_type', 'unknown'),
                'compression_ratio': config.get('compression_ratio', 1.0),
                'f1_mean': summary.get('f1_mean', 0),
                'f1_std': summary.get('f1_std', 0),
                'em_mean': summary.get('em_mean', 0),
                'success_rate': summary.get('success_rate', 0),
                'latency_mean': summary.get('latency_mean', 0),
            }

            # Add context sizes if available
            for key in summary:
                if 'context' in key:
                    row[key] = summary[key]

            rows.append(row)

        return pd.DataFrame(rows)

    def plot_f1_comparison(self, save_path: Optional[str] = None):
        """
        Plot F1 score comparison across methods.

        Args:
            save_path: Path to save figure (optional)
        """
        df = self.create_comparison_dataframe()

        if df.empty:
            print("No data to plot")
            return

        plt.figure(figsize=(12, 6))

        # Group by compression type
        df_sorted = df.sort_values('f1_mean', ascending=False)

        colors = {
            'none': 'skyblue',
            'fixed': 'orange',
            'role_specific': 'green'
        }

        bar_colors = [colors.get(ct, 'gray')
                     for ct in df_sorted['compression_type']]

        plt.bar(
            range(len(df_sorted)),
            df_sorted['f1_mean'],
            yerr=df_sorted['f1_std'],
            color=bar_colors,
            capsize=5
        )

        plt.xticks(
            range(len(df_sorted)),
            df_sorted['experiment'],
            rotation=45,
            ha='right'
        )
        plt.ylabel('F1 Score')
        plt.title('F1 Score Comparison Across Methods')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def plot_compression_tradeoff(self, save_path: Optional[str] = None):
        """
        Plot accuracy vs compression tradeoff.

        Args:
            save_path: Path to save figure
        """
        df = self.create_comparison_dataframe()

        if df.empty:
            print("No data to plot")
            return

        # Calculate average context reduction
        if 'retriever_context_mean' in df.columns:
            df['avg_context'] = df.get('retriever_context_mean', 0)
        else:
            df['avg_context'] = 1000  # placeholder

        plt.figure(figsize=(10, 6))

        compression_types = df['compression_type'].unique()
        markers = {'none': 'o', 'fixed': 's', 'role_specific': '^'}
        colors = {'none': 'skyblue', 'fixed': 'orange', 'role_specific': 'green'}

        for comp_type in compression_types:
            subset = df[df['compression_type'] == comp_type]
            plt.scatter(
                subset['avg_context'],
                subset['f1_mean'],
                label=comp_type.replace('_', ' ').title(),
                marker=markers.get(comp_type, 'o'),
                s=150,
                color=colors.get(comp_type, 'gray'),
                alpha=0.7
            )

        plt.xlabel('Average Context Size (tokens)')
        plt.ylabel('F1 Score')
        plt.title('Accuracy vs Context Size Tradeoff')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def statistical_comparison(
        self,
        method1: str,
        method2: str
    ) -> Dict:
        """
        Perform statistical comparison between two methods.

        Args:
            method1: First method name
            method2: Second method name

        Returns:
            Dictionary with test results
        """
        # Find experiments
        exp1 = None
        exp2 = None

        for exp_name, exp_data in self.results.items():
            if method1 in exp_name.lower():
                exp1 = exp_data
            if method2 in exp_name.lower():
                exp2 = exp_data

        if exp1 is None or exp2 is None:
            print(f"Could not find experiments for {method1} and {method2}")
            return {}

        # Get F1 scores
        f1_scores_1 = exp1['metrics']['f1_scores']
        f1_scores_2 = exp2['metrics']['f1_scores']

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(f1_scores_1, f1_scores_2)

        # Effect size (Cohen's d)
        diff = np.array(f1_scores_1) - np.array(f1_scores_2)
        cohens_d = np.mean(diff) / np.std(diff)

        results = {
            'method1': method1,
            'method2': method2,
            'mean_f1_1': np.mean(f1_scores_1),
            'mean_f1_2': np.mean(f1_scores_2),
            'mean_difference': np.mean(diff),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }

        print("\n" + "="*60)
        print("STATISTICAL COMPARISON")
        print("="*60)
        print(f"Method 1: {method1}")
        print(f"  Mean F1: {results['mean_f1_1']:.4f}")
        print(f"\nMethod 2: {method2}")
        print(f"  Mean F1: {results['mean_f1_2']:.4f}")
        print(f"\nDifference: {results['mean_difference']:.4f}")
        print(f"t-statistic: {results['t_statistic']:.4f}")
        print(f"p-value: {results['p_value']:.4f}")
        print(f"Cohen's d: {results['cohens_d']:.4f}")
        print(f"\nSignificant (p < 0.05): {results['significant']}")
        print("="*60)

        return results

    def generate_report(self, output_file: str = "analysis_report.txt"):
        """
        Generate a comprehensive analysis report.

        Args:
            output_file: Path to save report
        """
        df = self.create_comparison_dataframe()

        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CHAIN OF CLARIFICATIONS - ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")

            f.write("EXPERIMENT SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Experiments: {len(self.results)}\n\n")

            f.write(df.to_string() + "\n\n")

            # Best performing method
            best_f1_idx = df['f1_mean'].idxmax()
            best_method = df.iloc[best_f1_idx]

            f.write("\nBEST PERFORMING METHOD\n")
            f.write("-"*80 + "\n")
            f.write(f"Experiment: {best_method['experiment']}\n")
            f.write(f"Compression Type: {best_method['compression_type']}\n")
            f.write(f"F1 Score: {best_method['f1_mean']:.4f} ± {best_method['f1_std']:.4f}\n")
            f.write(f"EM Score: {best_method['em_mean']:.4f}\n")
            f.write(f"Success Rate: {best_method['success_rate']:.2%}\n")

        print(f"Report saved to {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze experimental results")

    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory containing results'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate plots'
    )

    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('METHOD1', 'METHOD2'),
        help='Compare two methods statistically'
    )

    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate analysis report'
    )
    
    parser.add_argument(
        '--error_analysis',
        action='store_true',
        help='Run error analysis on predictions'
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = ResultsAnalyzer(results_dir=args.results_dir)
    analyzer.load_results()

    # Generate plots
    if args.plot:
        print("Generating plots...")
        analyzer.plot_f1_comparison(
            save_path=str(Path(args.results_dir) / "f1_comparison.png")
        )
        analyzer.plot_compression_tradeoff(
            save_path=str(Path(args.results_dir) / "compression_tradeoff.png")
        )

    # Statistical comparison
    if args.compare:
        analyzer.statistical_comparison(args.compare[0], args.compare[1])

    # Generate report
    if args.report:
        analyzer.generate_report(
            output_file=str(Path(args.results_dir) / "analysis_report.txt")
        )
    
    # Error analysis
    if args.error_analysis:
        try:
            from utils.error_analysis import ErrorAnalyzer
            print("\nRunning error analysis...")
            
            error_analyzer = ErrorAnalyzer()
            
            for exp_name, exp_data in analyzer.results.items():
                metrics = exp_data.get('metrics', {})
                predictions = metrics.get('predictions', [])
                ground_truths = metrics.get('ground_truths', [])
                
                if predictions and ground_truths:
                    analysis = error_analyzer.analyze_batch(predictions, ground_truths)
                    report = error_analyzer.generate_error_report(analysis)
                    print(f"\n--- Error Analysis for {exp_name} ---")
                    print(report)
        except ImportError:
            print("Error analysis module not available")


if __name__ == "__main__":
    main()
