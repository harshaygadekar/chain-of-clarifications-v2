# 🚀 Google Colab T4 GPU Setup Guide
**Deadline: 3:00 PM | Started: 12:30 PM | Time Available: ~2.5 hours**

---

## 📋 COMPLETE STEP-BY-STEP GUIDE (Option A)

### **STEP 1: Create a ZIP of Your Project (2 min)**

Run this in your terminal:
```bash
cd /home/hrsh
zip -r coc_project.zip coc -x "coc/venv/*" -x "coc/venv_python313_backup/*" -x "coc/__pycache__/*" -x "coc/*/__pycache__/*"
```

---

### **STEP 2: Open Google Colab (1 min)**

1. Go to: **https://colab.research.google.com**
2. Click **"New Notebook"**
3. **IMPORTANT**: `Runtime` → `Change runtime type` → **T4 GPU** → Save

---

### **STEP 3: Upload & Extract Your Project (3 min)**

```python
# CELL 1: Upload and extract project
from google.colab import files
import zipfile

# Upload your zip file
print("📤 Upload your coc_project.zip file...")
uploaded = files.upload()

# Extract
with zipfile.ZipFile('coc_project.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

%cd coc
!ls -la
print("✅ Project extracted successfully!")
```

---

### **STEP 4: Verify GPU & Install Dependencies (5 min)**

```python
# CELL 2: Verify GPU
import torch
print("🔧 GPU Check:")
print(f"  CUDA Available: {torch.cuda.is_available()}")
print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

```python
# CELL 3: Install dependencies
!pip install transformers>=4.30.0 datasets>=2.14.0 wandb>=0.15.0 scikit-learn>=1.3.0 seaborn>=0.12.0
print("✅ Dependencies installed!")
```

---

### **STEP 5: Run Baseline Experiment (Quick Test)**

```python
# CELL 4: Run baseline experiment (quick test with 10 examples)
!python experiments/baseline.py --num_examples 10 --compression_type none

print("✅ Baseline complete!")
```

---

### **STEP 6: Run Full Compression Comparison (Main Experiment)**

```python
# CELL 5: Full comparison across compression ratios
!python experiments/baseline.py --comparison --num_examples 50

print("🎉 Full comparison complete!")
```

---

### **STEP 7: Analyze Results**

```python
# CELL 6: Run analysis
!python experiments/analyze_results.py

# View results
!ls -la results/
```

---

### **STEP 8: Download Results**

```python
# CELL 7: Download all results
from google.colab import files
import os

# Zip results
!zip -r results.zip results/

# Download
files.download('results.zip')
print("📥 Your results are downloading!")
```

---

## ⏰ TIME SCHEDULE

| Step | Duration | Complete By |
|------|----------|-------------|
| Create ZIP | 2 min | 12:35 |
| Open Colab + GPU | 2 min | 12:37 |
| Upload & Extract | 3 min | 12:40 |
| Install deps | 5 min | 12:45 |
| Baseline (10 examples) | ~15 min | 1:00 |
| Full comparison (50 examples) | ~45 min | 1:45 |
| Analyze + Download | 5 min | 1:50 |
| **Buffer for submission** | 70 min | **3:00 ✅** |

---

## � TROUBLESHOOTING

| Issue | Fix |
|-------|-----|
| "No GPU" | Runtime → Change runtime type → T4 GPU |
| Import Error | Restart runtime after pip install |
| Out of Memory | Reduce `--num_examples` to 25 |
| File Not Found | Run `%cd coc` first |

---

## 🔧 ADDITIONAL COMMANDS

### Check GPU Status
```python
!nvidia-smi
```

### Check Current Directory
```python
!pwd
!ls -la
```

### Clear GPU Cache (if memory issues)
```python
import torch
torch.cuda.empty_cache()
```

---

**Good luck! 🎯 You've got this!**
