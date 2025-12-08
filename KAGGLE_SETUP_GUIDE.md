# Chain of Clarifications - Kaggle Notebook Setup Guide

## Initial Setup

### Step 1: Enable GPU
1. Go to **Settings** (right sidebar) → **Accelerator** → Select **GPU T4 x2** or **P100**
2. Turn on **Internet** (required for downloading models)

### Step 2: Clone Repository
```python
!git clone https://github.com/harshaygadekar/chain-of-clarifications-v2.git
%cd chain-of-clarifications-v2
```

### Step 3: Install Dependencies
```python
!pip install -q transformers datasets accelerate scipy seaborn plotly
```

### Step 4: Verify GPU
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
!nvidia-smi
```

---

## Running Experiments

### Quick Test (5 examples)
```python
!python experiments/baseline.py --dataset qasper --compression_type role_specific --num_examples 5
```

### Full Experiment (50 examples)
```python
!python experiments/baseline.py \
    --dataset qasper \
    --compression_type role_specific \
    --compression_ratio 0.5 \
    --num_examples 50
```

### Run All Compression Methods
```python
# No compression (baseline)
!python experiments/baseline.py --dataset qasper --compression_type none --num_examples 50

# Fixed ratio
!python experiments/baseline.py --dataset qasper --compression_type fixed --num_examples 50

# Role-specific (ours)
!python experiments/baseline.py --dataset qasper --compression_type role_specific --num_examples 50

# Dynamic
!python experiments/baseline.py --dataset qasper --compression_type dynamic --num_examples 50

# Semantic
!python experiments/baseline.py --dataset qasper --compression_type semantic --num_examples 50
```

---

## Visualization & Analysis

### Generate Static Plots
```python
!python experiments/visualize.py --results_dir results
```

### Interactive Plots (Plotly)
```python
from experiments.interactive_viz import generate_all_interactive_plots
generate_all_interactive_plots("results", "results/interactive")

# Display in notebook
from IPython.display import HTML
HTML(open("results/interactive/comparison.html").read())
```

### Error Analysis
```python
import json
from analysis.error_analysis import ErrorAnalyzer

with open('results/role_specific_0.5_latest.json') as f:
    data = json.load(f)

analyzer = ErrorAnalyzer()
for out in data.get('detailed_outputs', []):
    analyzer.analyze(
        out['prediction'], 
        out['ground_truth'],
        out.get('agent_outputs', {}).get('retriever', ''),
        out['question']
    )
analyzer.print_report()
```

---

## Kaggle-Specific Tips

### Save Results to Output
```python
import shutil
shutil.copytree('results', '/kaggle/working/results')
```

### Download Results
Results will appear in the **Output** tab after notebook execution.

### Persistent Storage
```python
# Save to Kaggle dataset (if you have write access)
!cp -r results /kaggle/working/
```

### Memory Management
```python
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
```

---

## Configuration Reference

| Parameter | Default | Options |
|-----------|---------|---------|
| `--model_name` | `microsoft/Phi-3.5-mini-instruct` | Any HuggingFace model |
| `--dataset` | `qasper` | `qasper`, `squad`, `hotpotqa`, `cnn_dailymail`, `eli5` |
| `--compression_type` | `none` | `none`, `fixed`, `role_specific`, `dynamic`, `semantic` |
| `--compression_ratio` | `0.5` | `0.0` - `1.0` |
| `--num_examples` | `100` | Any integer |

---

## Troubleshooting

### Out of Memory
```python
# Use smaller batch or fewer examples
!python experiments/baseline.py --num_examples 20
```

### Internet Not Enabled
Go to Settings → Internet → Turn ON

### Model Download Fails
```python
# Pre-download model
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct")
```
