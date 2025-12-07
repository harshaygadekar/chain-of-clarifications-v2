# 📊 Chain of Clarifications - Project Analysis

## 🎯 What is This Project?

This is a **research project** investigating **role-specific context compression** for multi-agent LLM systems. 

### Core Research Question
> "Can we compress the context passed between LLM agents in a way that's **tailored to each agent's role**, achieving better accuracy than naive fixed-ratio compression?"

### Key Innovation
Instead of compressing all content uniformly (e.g., keep 50% of everything), the system **adapts compression based on what the next agent needs**:
- **Retriever → Reasoner**: Keep facts, entities, question-relevant sentences
- **Reasoner → Verifier**: Keep final answers, reasoning steps, evidence

---

## 🔄 System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   RETRIEVER     │────▶│    REASONER     │────▶│    VERIFIER     │
│                 │     │                 │     │                 │
│ Extracts        │     │ Generates       │     │ Validates &     │
│ relevant info   │     │ answer          │     │ finalizes       │
│ from document   │     │ with reasoning  │     │ the answer      │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                      │                      │
        ▼                      ▼                      ▼
   [Clarifier]           [Clarifier]            [Final Answer]
   Compress for          Compress for
   Reasoner              Verifier
```

### Agent Roles

| Agent | Purpose | Input | Output |
|-------|---------|-------|--------|
| **Retriever** | Extract relevant passages | Question + Document | Relevant context |
| **Reasoner** | Generate answer with reasoning | Question + Compressed context | Answer + Reasoning |
| **Verifier** | Validate and finalize | Question + Compressed reasoning | Final answer |

---

## 📏 Metrics - How Success is Measured

### Primary Metrics

#### 1. **F1 Score** (0.0 - 1.0)
Measures word overlap between prediction and ground truth.

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Where:
- Precision = Common words / Words in prediction
- Recall = Common words / Words in ground truth
```

**Interpretation:**
- **0.9+** = Excellent (very close to ground truth)
- **0.7-0.9** = Good 
- **0.5-0.7** = Moderate
- **<0.5** = Poor

#### 2. **Exact Match (EM)** (0 or 1)
Binary: Does the normalized prediction exactly match ground truth?

**Interpretation:**
- Higher EM = Model producing exact correct answers
- Typically lower than F1 (stricter measure)

### Secondary Metrics

| Metric | What it Measures | Good Value |
|--------|------------------|------------|
| **Context Size** | Tokens passed to each agent | Lower = more compression |
| **Latency** | Time per example (seconds) | Lower = faster |
| **Success Rate** | % of examples completed | Higher = more robust |
| **Memory (MB)** | GPU memory used | Lower = efficient |

---

## 🧪 Experiment Configurations

### Compression Types

```bash
--compression_type none           # Baseline: No compression
--compression_type fixed          # Naive: Fixed ratio across all content
--compression_type role_specific  # OURS: Tailored to agent roles
```

### What Gets Compared

| Configuration | Description | Expected F1 |
|--------------|-------------|-------------|
| `no_compression` | Full context passed | ~0.75 (baseline) |
| `fixed_25` | Keep 25% of all content | ~0.55-0.60 |
| `fixed_50` | Keep 50% of all content | ~0.65 |
| `fixed_75` | Keep 75% of all content | ~0.70 |
| `role_specific_50` | **Our method** - 50% adaptive | ~0.70 ✓ |

---

## 🚀 Experiment Workflow

### Step 1: Single Experiment
```bash
python experiments/baseline.py \
    --num_examples 10 \
    --compression_type role_specific \
    --compression_ratio 0.5
```

### Step 2: Full Comparison (runs all configurations)
```bash
python experiments/baseline.py --comparison --num_examples 50
```

This automatically runs:
- No compression baseline
- Fixed compression at 25%, 50%, 75%
- Role-specific compression at 25%, 50%, 75%

### Step 3: Analyze Results
```bash
python experiments/analyze_results.py --plot --report
```

Generates:
- F1 score comparison bar charts
- Compression vs accuracy tradeoff plots
- Statistical significance tests (p-values)
- Analysis report

---

## 📈 How to Know Experiments are Working

### ✅ Success Indicators

1. **Logs show progress**:
```
Example 1/10
Q: What year did X happen?
GT: 1990
Prediction: 1990
✓ Successful
```

2. **F1 scores are reasonable** (> 0.5)

3. **Results saved** to `results/` folder:
```
results/
├── baseline_no_compression.json
├── fixed_compression_50.json
├── role_specific_50.json
└── comparison_YYYYMMDD_HHMMSS.json
```

4. **Memory usage stays stable** (not constantly increasing)

### ❌ Failure Indicators

- F1 scores near 0 → Model not generating answers
- OOM errors → Reduce `--num_examples`
- Empty predictions → Check model loading
- All failures → Check dependencies

---

## 📊 Interpreting Results

### Result JSON Structure
```json
{
  "experiment_name": "role_specific_50",
  "configuration": {
    "compression_type": "role_specific",
    "compression_ratio": 0.5,
    "num_examples": 50
  },
  "metrics": {
    "f1_mean": 0.72,      // ← Main accuracy metric
    "f1_std": 0.15,       // ← Consistency
    "em_mean": 0.58,      // ← Exact match rate
    "success_rate": 0.96  // ← % completed
  },
  "counts": {
    "successful": 48,
    "failed": 2,
    "total": 50
  }
}
```

### Key Comparisons to Make

1. **Role-Specific vs Fixed (same ratio)**
   - Role-specific should have **higher F1** at same compression
   
2. **Compression vs No-Compression**
   - How much accuracy is lost for memory savings?
   
3. **Statistical Significance**
   - p-value < 0.05 = significant difference

---

## ⚡ Quick Reference Commands

### Run Quick Test (10 examples)
```bash
python experiments/baseline.py --num_examples 10
```

### Run Full Comparison (recommended: 50+)
```bash
python experiments/baseline.py --comparison --num_examples 50
```

### Analyze and Generate Plots
```bash
python experiments/analyze_results.py --results_dir results --plot
```

### Check GPU Status
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

---

## 📋 Expected Results Table

| Method | F1 ↑ | EM ↑ | Context Reduction ↑ | Memory ↓ |
|--------|------|------|---------------------|----------|
| No Compression | ~0.75 | ~0.65 | 0% | ~5.5 GB |
| Fixed 50% | ~0.65 | ~0.55 | 50% | ~3.5 GB |
| **Role-Specific 50%** | **~0.70** | **~0.60** | **45%** | **~3.8 GB** |

**Key Insight**: Role-specific retains ~93% of baseline accuracy while fixed retains only ~87%.

---

## 🎯 Summary for Your Deadline

1. **Run the experiment**: `python experiments/baseline.py --comparison --num_examples 50`
2. **Track F1 and EM scores** - main success metrics
3. **Role-specific > Fixed** at same compression ratio = success
4. **Download results**: `results/*.json` files
5. **Generate plots**: `python experiments/analyze_results.py --plot`

**Good luck! 🚀**
