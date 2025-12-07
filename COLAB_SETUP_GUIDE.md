# 🚀 Google Colab T4 GPU Setup Guide

## Quick Setup

### **STEP 1: Open Google Colab**
1. Go to **https://colab.research.google.com** → **New Notebook**
2. `Runtime` → `Change runtime type` → **T4 GPU** → Save

---

### **STEP 2: Clone & Setup**
```python
!git clone https://github.com/harshaygadekar/chain-of-clarifications-v2.git
%cd chain-of-clarifications-v2
!pip install -q transformers>=4.30.0 datasets>=2.14.0 scikit-learn>=1.3.0 seaborn>=0.12.0 scipy>=1.11.0 accelerate>=0.24.0

print("✅ Setup complete!")
```

---

### **STEP 3: Verify GPU**
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## Running Experiments

### Quick Test (5-10 min)
```python
!python experiments/baseline.py --num_examples 5
```
> Uses **Phi-2** + **CNN/DailyMail** by default (long articles → summaries)

### Full Comparison (~1.5 hours)
```python
!python experiments/baseline.py --comparison --num_examples 15
```

### Try Different Datasets
```python
# Long-form QA (explanatory answers)
!python experiments/baseline.py --dataset eli5 --num_examples 10

# Multi-hop QA
!python experiments/baseline.py --dataset hotpotqa --num_examples 10
```

---

## Generate Visualizations
```python
!python experiments/visualize.py
```

---

## Download Results
```python
from google.colab import files
!zip -r results.zip results/
files.download('results.zip')
```

---

## Default Configuration

| Setting | Value |
|---------|-------|
| Model | `microsoft/phi-2` (2.7B params) |
| Dataset | `cnn_dailymail` (summarization) |
| GPU Memory | ~6GB |

---

## Available Datasets

| Dataset | Type | Output Length |
|---------|------|---------------|
| `cnn_dailymail` | Summarization | 2-4 sentences |
| `eli5` | Long-form QA | 2-5 sentences |
| `hotpotqa` | Multi-hop QA | 1-2 sentences |
| `squad` | Extractive QA | 1-3 words |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No GPU | Runtime → Change runtime type → T4 |
| OOM Error | Reduce `--num_examples` to 5 |
| Import Error | Restart runtime |
