"""
Interactive Visualizations with Plotly

Creates interactive HTML plots for exploratory analysis:
- Compression method comparison
- Context flow Sankey diagram
- Attention/importance heatmaps
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Check for plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not installed. Run: pip install plotly")


def create_interactive_comparison(
    results: Dict[str, Dict],
    output_path: str = "results/interactive_comparison.html",
    title: str = "Compression Method Comparison"
) -> Optional[str]:
    """
    Create interactive bar chart comparing compression methods.
    
    Args:
        results: Dict mapping method names to their metrics
        output_path: Path to save HTML file
        title: Chart title
        
    Returns:
        Path to saved HTML file or None if Plotly not available
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly not available")
        return None
    
    methods = list(results.keys())
    f1_scores = [r.get('f1_mean', 0) for r in results.values()]
    latencies = [r.get('latency_mean', 0) for r in results.values()]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('F1 Score by Method', 'Latency by Method')
    )
    
    # F1 Score bars
    fig.add_trace(
        go.Bar(
            name='F1 Score',
            x=methods,
            y=f1_scores,
            text=[f'{v:.3f}' for v in f1_scores],
            textposition='outside',
            marker_color='#4CAF50',
            hovertemplate='%{x}<br>F1: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Latency bars
    fig.add_trace(
        go.Bar(
            name='Latency (s)',
            x=methods,
            y=latencies,
            text=[f'{v:.1f}s' for v in latencies],
            textposition='outside',
            marker_color='#2196F3',
            hovertemplate='%{x}<br>Latency: %{y:.2f}s<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=title,
        showlegend=False,
        height=500,
        template='plotly_white'
    )
    
    # Save HTML
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Interactive comparison saved to: {output_path}")
    
    return output_path


def create_compression_flow(
    context_sizes: Dict[str, List[int]],
    output_path: str = "results/compression_flow.html",
    title: str = "Context Compression Flow"
) -> Optional[str]:
    """
    Create Sankey diagram showing context size reduction through pipeline.
    
    Args:
        context_sizes: Dict with keys like 'retriever', 'retriever_compressed', etc.
        output_path: Path to save HTML
        title: Chart title
        
    Returns:
        Path to saved file
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    # Calculate average sizes
    avg_sizes = {k: sum(v) / len(v) if v else 0 for k, v in context_sizes.items()}
    
    # Define nodes
    labels = [
        'Document',
        'Retriever Output',
        'After Compression 1',
        'Reasoner Output',
        'After Compression 2',
        'Verifier Output'
    ]
    
    # Get values (use defaults if not present)
    values = [
        avg_sizes.get('document', 500),
        avg_sizes.get('retriever', 250),
        avg_sizes.get('retriever_compressed', 125),
        avg_sizes.get('reasoner', 200),
        avg_sizes.get('reasoner_compressed', 100),
        avg_sizes.get('verifier', 150)
    ]
    
    # Links (source, target, value)
    links = [
        (0, 1, values[1]),
        (1, 2, values[2]),
        (2, 3, values[3]),
        (3, 4, values[4]),
        (4, 5, values[5])
    ]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=['#e3f2fd', '#64b5f6', '#1976d2', '#64b5f6', '#1976d2', '#0d47a1']
        ),
        link=dict(
            source=[l[0] for l in links],
            target=[l[1] for l in links],
            value=[l[2] for l in links],
            color='rgba(100, 181, 246, 0.4)'
        )
    )])
    
    fig.update_layout(
        title=title,
        font_size=12,
        height=400
    )
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Compression flow saved to: {output_path}")
    
    return output_path


def create_attention_heatmap(
    sentences: List[str],
    importance_scores: List[float],
    kept_indices: List[int],
    output_path: str = "results/attention_heatmap.html",
    title: str = "Compression Attention Visualization"
) -> Optional[str]:
    """
    Create heatmap showing what compression keeps/removes.
    
    Args:
        sentences: List of original sentences
        importance_scores: Score for each sentence (0-1)
        kept_indices: Indices of sentences that were kept
        output_path: Path to save HTML
        title: Chart title
        
    Returns:
        Path to saved file
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    # Truncate sentences for display
    display_sents = [s[:50] + '...' if len(s) > 50 else s for s in sentences]
    
    # Create color based on kept/removed
    colors = ['#4CAF50' if i in kept_indices else '#f44336' 
              for i in range(len(sentences))]
    
    fig = go.Figure()
    
    # Importance score bars
    fig.add_trace(go.Bar(
        y=display_sents,
        x=importance_scores,
        orientation='h',
        marker_color=colors,
        text=[f'{s:.2f}' for s in importance_scores],
        textposition='outside',
        hovertemplate='%{y}<br>Score: %{x:.3f}<br>' + 
                      '<b>%{customdata}</b><extra></extra>',
        customdata=['KEPT' if i in kept_indices else 'REMOVED' 
                    for i in range(len(sentences))]
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Importance Score',
        yaxis_title='Sentence',
        height=max(400, len(sentences) * 30),
        template='plotly_white',
        showlegend=False
    )
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Attention heatmap saved to: {output_path}")
    
    return output_path


def create_error_distribution(
    error_counts: Dict[str, int],
    output_path: str = "results/error_distribution.html",
    title: str = "Error Category Distribution"
) -> Optional[str]:
    """
    Create interactive pie/sunburst chart of error categories.
    
    Args:
        error_counts: Dict mapping error category to count
        output_path: Path to save HTML
        title: Chart title
        
    Returns:
        Path to saved file
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    categories = list(error_counts.keys())
    values = list(error_counts.values())
    
    # Color mapping
    color_map = {
        'correct': '#4CAF50',
        'format_error': '#f44336',
        'hallucination': '#ff9800',
        'incomplete': '#ffeb3b',
        'wrong_entity': '#9c27b0',
        'semantic_drift': '#2196f3',
        'too_short': '#607d8b',
        'too_long': '#795548'
    }
    colors = [color_map.get(c, '#9e9e9e') for c in categories]
    
    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=values,
        marker_colors=colors,
        hole=0.3,
        textinfo='label+percent',
        hovertemplate='%{label}<br>Count: %{value}<br>%{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title=title,
        height=500
    )
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Error distribution saved to: {output_path}")
    
    return output_path


def create_metrics_over_examples(
    metrics_history: List[Dict],
    output_path: str = "results/metrics_timeline.html",
    title: str = "Metrics Over Examples"
) -> Optional[str]:
    """
    Create line chart showing metrics progression over examples.
    
    Args:
        metrics_history: List of metric dicts per example
        output_path: Path to save HTML
        title: Chart title
        
    Returns:
        Path to saved file
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    fig = go.Figure()
    
    # Extract metrics
    examples = list(range(1, len(metrics_history) + 1))
    f1_scores = [m.get('f1', 0) for m in metrics_history]
    latencies = [m.get('latency', 0) for m in metrics_history]
    
    # Running average F1
    running_avg = []
    for i, f1 in enumerate(f1_scores):
        avg = sum(f1_scores[:i+1]) / (i + 1)
        running_avg.append(avg)
    
    fig.add_trace(go.Scatter(
        x=examples,
        y=f1_scores,
        mode='markers',
        name='F1 Score',
        marker=dict(color='#4CAF50', size=8, opacity=0.6),
        hovertemplate='Example %{x}<br>F1: %{y:.3f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=examples,
        y=running_avg,
        mode='lines',
        name='Running Average',
        line=dict(color='#1976d2', width=2),
        hovertemplate='Example %{x}<br>Avg F1: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Example Number',
        yaxis_title='F1 Score',
        height=400,
        template='plotly_white'
    )
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info(f"Metrics timeline saved to: {output_path}")
    
    return output_path


def generate_all_interactive_plots(
    results_dir: str = "results",
    output_dir: str = "results/interactive"
) -> Dict[str, str]:
    """
    Generate all interactive plots from saved results.
    
    Args:
        results_dir: Directory containing result JSON files
        output_dir: Directory to save HTML plots
        
    Returns:
        Dict mapping plot name to file path
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly not installed. Run: pip install plotly")
        return {}
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    generated = {}
    
    # Load and aggregate results
    results_path = Path(results_dir)
    all_results = {}
    
    for json_file in results_path.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            name = data.get('experiment_name', json_file.stem)
            metrics = data.get('metrics', {}).get('summary', {})
            all_results[name] = metrics
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
    
    # Generate comparison chart
    if all_results:
        path = create_interactive_comparison(
            all_results,
            f"{output_dir}/comparison.html"
        )
        if path:
            generated['comparison'] = path
    
    logger.info(f"Generated {len(generated)} interactive plots")
    return generated
