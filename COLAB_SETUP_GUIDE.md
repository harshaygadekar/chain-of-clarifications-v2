# 🚀 Google Colab T4 GPU Setup Guide

## Quick Setup (GitHub Clone Method)

### **STEP 1: Open Google Colab**
1. Go to **https://colab.research.google.com** → **New Notebook**
2. `Runtime` → `Change runtime type` → **T4 GPU** → Save

---

### **STEP 2: Clone & Setup**
```python
# Clone repo and install dependencies
!git clone https://github.com/harshaygadekar/chain-of-clarifications-v2.git
%cd chain-of-clarifications-v2
!pip install -q transformers>=4.30.0 datasets>=2.14.0 scikit-learn>=1.3.0 seaborn>=0.12.0 scipy>=1.11.0

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

### Quick Test (5 min)
```python
!python experiments/baseline.py --num_examples 10
```
> Uses **flan-t5-large** + **HotpotQA** by default

### Full Comparison (~1.5 hours)
```python
!python experiments/baseline.py --comparison --num_examples 25
```

### Ablation Studies (~45 min)
```python
!python experiments/ablation.py --all --num_examples 15
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
| Model | `google/flan-t5-large` (780M params) |
| Dataset | HotpotQA (multi-hop QA) |
| GPU Memory | ~3GB |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| No GPU | Runtime → Change runtime type → T4 |
| OOM Error | Reduce `--num_examples` to 15 |
| Import Error | Restart runtime |
