# Chain of Clarifications - Google Colab Setup Guide

## Complete Setup & Run Commands

### Step 1: Clone Repository
```python
!git clone https://github.com/harshaygadekar/chain-of-clarifications-v2.git
%cd chain-of-clarifications-v2
```

### Step 2: Install Dependencies
```python
!pip install -q transformers datasets accelerate scipy seaborn plotly
```

### Step 3: Check GPU
```python
!nvidia-smi
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Step 4: Run Experiments

#### Quick Test (5 examples)
```python
!python experiments/baseline.py --dataset qasper --compression_type role_specific --num_examples 5
```

#### Full Experiment (50 examples)
```python
!python experiments/baseline.py \
    --dataset qasper \
    --compression_type role_specific \
    --compression_ratio 0.5 \
    --num_examples 50
```

#### Run All Compression Methods
```python
!python experiments/baseline.py --dataset qasper --compression_type none --num_examples 50
!python experiments/baseline.py --dataset qasper --compression_type fixed --num_examples 50
!python experiments/baseline.py --dataset qasper --compression_type role_specific --num_examples 50
!python experiments/baseline.py --dataset qasper --compression_type dynamic --num_examples 50
!python experiments/baseline.py --dataset qasper --compression_type semantic --num_examples 50
```

### Step 5: Visualize Results

#### Static Plots (Matplotlib)
```python
!python experiments/visualize.py --results_dir results
```

#### Interactive Plots (Plotly)
```python
from experiments.interactive_viz import generate_all_interactive_plots
generate_all_interactive_plots("results", "results/interactive")
```

### Step 6: Error Analysis
```python
import json
from analysis.error_analysis import ErrorAnalyzer

# Load results
with open('results/role_specific_0.5_latest.json') as f:
    data = json.load(f)

# Analyze errors
analyzer = ErrorAnalyzer()
for output in data.get('detailed_outputs', []):
    analyzer.analyze(
        prediction=output['prediction'],
        ground_truth=output['ground_truth'],
        source_context=output.get('agent_outputs', {}).get('retriever', ''),
        question=output['question']
    )

analyzer.print_report()
```

### Step 7: View Detailed Outputs
```python
import json

with open('results/role_specific_0.5_latest.json') as f:
    data = json.load(f)

for ex in data.get('detailed_outputs', [])[:3]:
    print(f"Question: {ex['question'][:80]}...")
    print(f"Ground Truth: {ex['ground_truth'][:100]}...")
    print(f"Prediction: {ex['prediction'][:100]}...")
    print(f"Retriever: {ex['agent_outputs']['retriever'][:150]}...")
    print("-" * 50)
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `microsoft/Phi-3.5-mini-instruct` | HuggingFace model |
| `--dataset` | `qasper` | Dataset name |
| `--compression_type` | `none` | Compression method |
| `--compression_ratio` | `0.5` | Compression ratio (0.0-1.0) |
| `--num_examples` | `100` | Number of examples |

## Available Datasets
- `qasper` - Scientific paper QA (recommended)
- `squad` - Factoid QA
- `hotpotqa` - Multi-hop QA
- `cnn_dailymail` - Summarization
- `eli5` - Long-form QA

## Available Compression Types
- `none` - No compression (baseline)
- `fixed` - Fixed-ratio truncation
- `role_specific` - Role-aware compression (ours)
- `dynamic` - Adaptive ratio selection
- `semantic` - Semantic-aware compression
