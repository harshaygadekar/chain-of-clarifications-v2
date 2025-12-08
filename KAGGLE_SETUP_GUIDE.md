# Chain of Clarifications - Kaggle Notebook Setup Guide

## Step 1: Setup
```python
# Clone repo
!git clone https://github.com/harshaygadekar/chain-of-clarifications-v2.git
%cd chain-of-clarifications-v2

# Install dependencies
!pip install -q transformers datasets accelerate scipy seaborn plotly
```

## Step 2: Check GPU
```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

## Step 3: Run Experiments

### Quick Test (5 examples)
```python
!python experiments/baseline.py --dataset narrativeqa --compression_type role_specific --num_examples 5
```

### Full Experiment (50 examples)
```python
!python experiments/baseline.py --dataset narrativeqa --compression_type role_specific --compression_ratio 0.5 --num_examples 50
```

### Run All Compression Methods
```python
!python experiments/baseline.py --dataset narrativeqa --compression_type none --num_examples 50
!python experiments/baseline.py --dataset narrativeqa --compression_type fixed --num_examples 50
!python experiments/baseline.py --dataset narrativeqa --compression_type role_specific --num_examples 50
!python experiments/baseline.py --dataset narrativeqa --compression_type dynamic --num_examples 50
!python experiments/baseline.py --dataset narrativeqa --compression_type semantic --num_examples 50
```

## Step 4: Visualize
```python
!python experiments/visualize.py --results_dir results
```

## Step 5: Download Results
```python
import shutil
shutil.copytree('results', '/kaggle/working/results')
!zip -r /kaggle/working/results.zip /kaggle/working/results
```

## Available Datasets
| Dataset | Description |
|---------|-------------|
| `narrativeqa` | Long-form story QA (default) |
| `squad` | Factoid QA |
| `hotpotqa` | Multi-hop reasoning |
| `cnn_dailymail` | Summarization |
| `eli5` | Long-form explanations |

## Compression Types
- `none` - Baseline (no compression)
- `fixed` - Fixed-ratio truncation
- `role_specific` - Our method (role-aware)
- `dynamic` - Adaptive ratio
- `semantic` - Semantic-aware
