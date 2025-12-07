"""
Research Paper Visualizations

Generates publication-quality figures for the Chain of Clarifications paper.
Supports 5 key visualizations for research paper inclusion.

Usage:
    python experiments/visualize.py --results_dir results --output_dir results/figures
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette for consistency
COLORS = {
    'none': '#2ecc71',           # Green - baseline
    'fixed': '#e74c3c',          # Red - naive
    'role_specific': '#3498db',  # Blue - our method
    'dynamic': '#9b59b6',        # Purple
    'semantic': '#f39c12',       # Orange
}

METHOD_LABELS = {
    'none': 'No Compression',
    'fixed': 'Fixed Compression',
    'role_specific': 'Role-Specific (Ours)',
    'dynamic': 'Dynamic',
    'semantic': 'Semantic',
}


class ResultsLoader:
    """Load and parse experiment results from JSON files."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results = {}
        self._load_all_results()
    
    def _load_all_results(self):
        """Load all JSON result files."""
        if not self.results_dir.exists():
            logger.warning(f"Results directory not found: {self.results_dir}")
            return
        
        for json_file in self.results_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    name = json_file.stem
                    self.results[name] = data
                    logger.info(f"Loaded: {name}")
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
    
    def get_by_compression_type(self, compression_type: str) -> List[Dict]:
        """Get all results for a specific compression type."""
        matching = []
        for name, data in self.results.items():
            config = data.get('configuration', {})
            if config.get('compression_type') == compression_type:
                matching.append(data)
        return matching
    
    def get_comparison_data(self) -> Dict:
        """Extract data organized for comparison plots."""
        comparison = defaultdict(lambda: defaultdict(list))
        
        for name, data in self.results.items():
            config = data.get('configuration', {})
            metrics = data.get('metrics', {})
            summary = metrics.get('summary', {})
            
            comp_type = config.get('compression_type', 'unknown')
            comp_ratio = config.get('compression_ratio', 0.5)
            
            # Get F1 and EM from summary or raw scores
            f1_mean = summary.get('f1_mean', 0)
            if f1_mean == 0 and 'f1_scores' in metrics:
                f1_scores = metrics['f1_scores']
                f1_mean = np.mean(f1_scores) if f1_scores else 0
            
            em_mean = summary.get('em_mean', 0)
            if em_mean == 0 and 'em_scores' in metrics:
                em_scores = metrics['em_scores']
                em_mean = np.mean(em_scores) if em_scores else 0
            
            comparison[comp_type]['f1'].append(f1_mean)
            comparison[comp_type]['em'].append(em_mean)
            comparison[comp_type]['ratio'].append(comp_ratio)
            comparison[comp_type]['latency'].append(
                summary.get('latency_mean', 0)
            )
        
        return dict(comparison)


class ResearchVisualizer:
    """Generate research paper visualizations."""
    
    def __init__(self, results_dir: str, output_dir: str):
        self.loader = ResultsLoader(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(self):
        """Generate all visualizations."""
        logger.info("Generating all visualizations...")
        
        self.plot_compression_comparison()
        self.plot_accuracy_vs_compression()
        self.plot_ablation_heatmap()
        self.plot_statistical_significance()
        self.plot_context_flow()
        
        logger.info(f"All figures saved to: {self.output_dir}")
    
    def plot_compression_comparison(self):
        """
        #1: Compression Method Comparison Bar Chart
        Shows F1/EM scores across different compression methods.
        """
        logger.info("Generating: Compression Comparison")
        
        data = self.loader.get_comparison_data()
        
        if not data:
            logger.warning("No data for compression comparison")
            self._create_placeholder("compression_comparison.png", 
                                     "Compression Comparison\n(Run experiments first)")
            return
        
        # Prepare data
        methods = []
        f1_scores = []
        em_scores = []
        
        for method in ['none', 'fixed', 'role_specific', 'dynamic', 'semantic']:
            if method in data:
                methods.append(METHOD_LABELS.get(method, method))
                f1_scores.append(np.mean(data[method]['f1']))
                em_scores.append(np.mean(data[method]['em']))
        
        if not methods:
            self._create_placeholder("compression_comparison.png",
                                     "No comparison data available")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score',
                       color='#3498db', edgecolor='white', linewidth=1)
        bars2 = ax.bar(x + width/2, em_scores, width, label='Exact Match',
                       color='#2ecc71', edgecolor='white', linewidth=1)
        
        # Customize
        ax.set_xlabel('Compression Method')
        ax.set_ylabel('Score')
        ax.set_title('Compression Method Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        self._save_figure(fig, "compression_comparison.png")
    
    def plot_accuracy_vs_compression(self):
        """
        #2: Accuracy vs Compression Tradeoff Curve
        Shows how accuracy changes with compression ratio for each method.
        """
        logger.info("Generating: Accuracy vs Compression")
        
        data = self.loader.get_comparison_data()
        
        if not data:
            self._create_placeholder("accuracy_vs_compression.png",
                                     "Accuracy vs Compression\n(Run comparison experiments)")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method in ['fixed', 'role_specific', 'dynamic', 'semantic']:
            if method not in data:
                continue
            
            ratios = data[method]['ratio']
            f1_scores = data[method]['f1']
            
            if len(ratios) < 2:
                continue
            
            # Sort by ratio
            sorted_pairs = sorted(zip(ratios, f1_scores))
            ratios_sorted, f1_sorted = zip(*sorted_pairs)
            
            color = COLORS.get(method, '#333333')
            label = METHOD_LABELS.get(method, method)
            
            ax.plot(ratios_sorted, f1_sorted, 'o-', 
                   label=label, color=color, linewidth=2, markersize=8)
        
        # Add baseline if available
        if 'none' in data and data['none']['f1']:
            baseline_f1 = np.mean(data['none']['f1'])
            ax.axhline(y=baseline_f1, color=COLORS['none'], linestyle='--',
                      linewidth=2, label='No Compression (Baseline)')
        
        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel('F1 Score')
        ax.set_title('Accuracy vs Compression Tradeoff')
        ax.legend(loc='best')
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        self._save_figure(fig, "accuracy_vs_compression.png")
    
    def plot_ablation_heatmap(self):
        """
        #7: Ablation Study Heatmap
        Shows impact of different configuration choices.
        """
        logger.info("Generating: Ablation Heatmap")
        
        # Look for ablation results
        ablation_data = {}
        for name, data in self.loader.results.items():
            if 'ablation' in name.lower():
                ablation_data[name] = data
        
        if not ablation_data:
            # Create from available data
            data = self.loader.get_comparison_data()
            if not data:
                self._create_placeholder("ablation_heatmap.png",
                                        "Ablation Heatmap\n(Run ablation.py first)")
                return
            
            # Create heatmap from comparison data
            methods = []
            metrics_matrix = []
            
            for method in ['none', 'fixed', 'role_specific', 'dynamic', 'semantic']:
                if method in data:
                    methods.append(METHOD_LABELS.get(method, method))
                    f1 = np.mean(data[method]['f1']) if data[method]['f1'] else 0
                    em = np.mean(data[method]['em']) if data[method]['em'] else 0
                    lat = np.mean(data[method]['latency']) if data[method]['latency'] else 0
                    metrics_matrix.append([f1, em, lat])
        
        if not methods:
            self._create_placeholder("ablation_heatmap.png",
                                    "No ablation data available")
            return
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        metrics_array = np.array(metrics_matrix)
        
        # Normalize columns for better visualization
        normalized = metrics_array.copy()
        for i in range(metrics_array.shape[1]):
            col_max = metrics_array[:, i].max()
            if col_max > 0:
                normalized[:, i] = metrics_array[:, i] / col_max
        
        im = ax.imshow(normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(range(3))
        ax.set_xticklabels(['F1 Score', 'Exact Match', 'Latency (inv)'])
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        
        # Add value annotations
        for i in range(len(methods)):
            for j in range(3):
                value = metrics_array[i, j]
                text = f'{value:.2f}' if j < 2 else f'{value:.1f}s'
                ax.text(j, i, text, ha='center', va='center', 
                       color='white' if normalized[i, j] < 0.5 else 'black',
                       fontsize=10, fontweight='bold')
        
        ax.set_title('Method Comparison Heatmap')
        plt.colorbar(im, ax=ax, label='Normalized Score')
        
        plt.tight_layout()
        self._save_figure(fig, "ablation_heatmap.png")
    
    def plot_statistical_significance(self):
        """
        #8: Statistical Significance Plot
        Shows confidence intervals and significance of differences.
        """
        logger.info("Generating: Statistical Significance")
        
        data = self.loader.get_comparison_data()
        
        if not data:
            self._create_placeholder("statistical_significance.png",
                                    "Statistical Significance\n(Run experiments first)")
            return
        
        # Calculate means and confidence intervals
        methods = []
        means = []
        errors = []
        colors = []
        
        for method in ['none', 'fixed', 'role_specific', 'dynamic', 'semantic']:
            if method in data and data[method]['f1']:
                f1_scores = data[method]['f1']
                methods.append(METHOD_LABELS.get(method, method))
                means.append(np.mean(f1_scores))
                # 95% CI: mean ± 1.96 * (std / sqrt(n))
                std = np.std(f1_scores)
                n = len(f1_scores)
                ci = 1.96 * (std / np.sqrt(n)) if n > 1 else 0
                errors.append(ci)
                colors.append(COLORS.get(method, '#333333'))
        
        if not methods:
            self._create_placeholder("statistical_significance.png",
                                    "No data for significance analysis")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=errors, capsize=5, 
                     color=colors, edgecolor='white', linewidth=1,
                     error_kw={'linewidth': 2, 'capthick': 2})
        
        ax.set_xlabel('Compression Method')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Scores with 95% Confidence Intervals')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.set_ylim(0, 1.0)
        
        # Add significance annotations if we have role_specific and fixed
        if 'role_specific' in data and 'fixed' in data:
            rs_f1 = data['role_specific']['f1']
            fixed_f1 = data['fixed']['f1']
            if rs_f1 and fixed_f1:
                # Simple difference
                diff = np.mean(rs_f1) - np.mean(fixed_f1)
                if diff > 0:
                    ax.annotate(f'+{diff:.2f}', 
                               xy=(0.5, 0.95), xycoords='axes fraction',
                               fontsize=12, ha='center',
                               bbox=dict(boxstyle='round', facecolor='lightgreen'))
        
        plt.tight_layout()
        self._save_figure(fig, "statistical_significance.png")
    
    def plot_context_flow(self):
        """
        #3: Context Size Flow Diagram
        Shows context reduction through agent stages.
        """
        logger.info("Generating: Context Flow")
        
        # Get context sizes from results
        context_data = defaultdict(lambda: defaultdict(list))
        
        for name, data in self.loader.results.items():
            config = data.get('configuration', {})
            metrics = data.get('metrics', {})
            context_sizes = metrics.get('context_sizes', {})
            
            comp_type = config.get('compression_type', 'unknown')
            
            for stage, sizes in context_sizes.items():
                if sizes:
                    context_data[comp_type][stage].extend(sizes)
        
        if not context_data:
            self._create_placeholder("context_flow.png",
                                    "Context Flow\n(Run experiments first)")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        stages = ['retriever', 'retriever_compressed', 
                  'reasoner', 'reasoner_compressed']
        stage_labels = ['Retriever\nOutput', 'After\nCompression',
                       'Reasoner\nOutput', 'After\nCompression']
        
        x = np.arange(len(stages))
        width = 0.15
        offset = 0
        
        for method in ['none', 'fixed', 'role_specific']:
            if method not in context_data:
                continue
            
            sizes = []
            for stage in stages:
                if stage in context_data[method]:
                    sizes.append(np.mean(context_data[method][stage]))
                else:
                    sizes.append(0)
            
            if any(s > 0 for s in sizes):
                color = COLORS.get(method, '#333333')
                label = METHOD_LABELS.get(method, method)
                ax.bar(x + offset, sizes, width, label=label, 
                      color=color, edgecolor='white', linewidth=1)
                offset += width
        
        ax.set_xlabel('Agent Stage')
        ax.set_ylabel('Context Size (tokens)')
        ax.set_title('Context Size Through Agent Pipeline')
        ax.set_xticks(x + width)
        ax.set_xticklabels(stage_labels)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        self._save_figure(fig, "context_flow.png")
    
    def _save_figure(self, fig, filename: str):
        """Save figure as PNG and PDF."""
        png_path = self.output_dir / filename
        pdf_path = self.output_dir / filename.replace('.png', '.pdf')
        
        fig.savefig(png_path, format='png', dpi=300)
        fig.savefig(pdf_path, format='pdf')
        plt.close(fig)
        
        logger.info(f"Saved: {png_path}")
    
    def _create_placeholder(self, filename: str, message: str):
        """Create placeholder figure with message."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center',
               fontsize=14, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightyellow'))
        ax.set_axis_off()
        self._save_figure(fig, filename)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate research paper visualizations"
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory containing experiment results'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/figures',
        help='Directory to save figures'
    )
    
    parser.add_argument(
        '--figure',
        type=str,
        choices=['all', 'comparison', 'tradeoff', 'ablation', 
                 'significance', 'context'],
        default='all',
        help='Which figure to generate'
    )
    
    args = parser.parse_args()
    
    visualizer = ResearchVisualizer(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    
    if args.figure == 'all':
        visualizer.generate_all()
    elif args.figure == 'comparison':
        visualizer.plot_compression_comparison()
    elif args.figure == 'tradeoff':
        visualizer.plot_accuracy_vs_compression()
    elif args.figure == 'ablation':
        visualizer.plot_ablation_heatmap()
    elif args.figure == 'significance':
        visualizer.plot_statistical_significance()
    elif args.figure == 'context':
        visualizer.plot_context_flow()
    
    print(f"\n✅ Figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
