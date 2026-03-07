# Role based context compression

A multi-agent question-answering system that uses a sequential **Retriever → Reasoner → Verifier** pipeline with intelligent context compression. The core idea is that each agent in the chain only receives the information it actually needs — reducing noise, saving tokens, and improving answer quality.

---

## How It Works

```
Document + Question
        │
        ▼
  ┌─────────────┐
  │  Retriever  │  Identifies relevant facts, entities, and evidence
  └──────┬──────┘
         │  [Compress for Reasoner]
         ▼
  ┌─────────────┐
  │   Reasoner  │  Applies step-by-step logic to produce a candidate answer
  └──────┬──────┘
         │  [Compress for Verifier]
         ▼
  ┌─────────────┐
  │   Verifier  │  Validates the reasoning and emits the final answer
  └─────────────┘
```

Each handoff between agents can optionally be compressed. The **role-specific compression** strategy is the primary innovation — it scores and selects content based on what the *next* agent actually needs rather than applying a blind truncation.

---

## Features

- **Three-agent pipeline** with shared model weights (single GPU load)
- **Five compression strategies**: none, fixed-ratio, role-specific, semantic, dynamic
- **Seven dataset loaders**: NarrativeQA, SQuAD, HotpotQA, DROP, CNN/DailyMail, ELI5, QASPER
- **Multi-turn conversation** support with anaphora resolution
- **Ablation framework** to isolate the contribution of each component
- **Multi-GPU parallel runner** for sweeping compression types concurrently
- **Publication-quality visualizations** for research reporting
- **File-based result cache** to avoid redundant inference during development

---

## Project Structure

```
chain-of-clarifications-v2/
├── agents/
│   ├── base_agent.py         # Shared model loading, generation, GPU diagnostics
│   ├── retriever.py          # Extracts relevant facts from source document
│   ├── reasoner.py           # Reasons over extracted info to form a candidate answer
│   ├── verifier.py           # Validates and produces the final answer
│   ├── agent_chain.py        # Orchestrates the full pipeline with compression
│   └── conversation_chain.py # Multi-turn wrapper with history and reference resolution
│
├── compression/
│   ├── naive_compression.py  # Fixed-ratio strategies (first_n, last_n, random, sentence_first)
│   ├── role_specific.py      # Role-aware scoring — the core innovation
│   ├── semantic_compression.py # Redundancy removal + sentence fusion
│   ├── dynamic_compression.py # Auto-selects ratio from question/doc complexity
│   └── attention_scorer.py   # Attention-based token importance scoring
│
├── data/
│   ├── dataset_factory.py    # Unified get_loader() interface
│   ├── load_narrativeqa.py
│   ├── load_squad.py
│   ├── load_hotpotqa.py
│   ├── load_drop.py
│   ├── load_cnn_dailymail.py
│   ├── load_eli5.py
│   └── load_qasper.py
│
├── experiments/
│   ├── baseline.py           # Main experiment runner (CLI entry point)
│   ├── ablation.py           # Component ablation studies
│   ├── run_parallel.py       # Multi-GPU parallel sweep
│   ├── analyze_results.py    # Statistical comparison of saved results
│   ├── visualize.py          # Publication-quality matplotlib/seaborn figures
│   └── interactive_viz.py    # Interactive exploration of results
│
├── utils/
│   ├── metrics.py            # F1, Exact Match, ROUGE, latency, memory tracking
│   ├── cache.py              # Hash-based file cache for agent results
│   ├── memory_tracker.py     # GPU/CPU memory monitoring
│   └── error_analysis.py     # Failure mode categorization
│
├── analysis/
│   └── error_analysis.py     # Cross-run error pattern analysis
│
├── requirements.txt
└── KAGGLE_SETUP_GUIDE.md
```

---

## Installation

```bash
git clone https://github.com/harshaygadekar/chain-of-clarifications-v2.git
cd chain-of-clarifications-v2

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA-capable GPU recommended (falls back to CPU).

---

## Quick Start

```bash
# Run baseline (no compression) on SQuAD, 10 examples
python experiments/baseline.py --dataset squad --compression_type none --num_examples 10

# Run role-specific compression on NarrativeQA
python experiments/baseline.py \
    --dataset narrativeqa \
    --compression_type role_specific \
    --compression_ratio 0.5 \
    --num_examples 50

# Compare all compression methods
python experiments/baseline.py --dataset narrativeqa --compression_type none --num_examples 50
python experiments/baseline.py --dataset narrativeqa --compression_type fixed --num_examples 50
python experiments/baseline.py --dataset narrativeqa --compression_type role_specific --num_examples 50
python experiments/baseline.py --dataset narrativeqa --compression_type dynamic --num_examples 50
python experiments/baseline.py --dataset narrativeqa --compression_type semantic --num_examples 50
```

---

## CLI Reference

### `experiments/baseline.py`

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `squad` | Dataset to use (see table below) |
| `--compression_type` | `none` | Compression strategy |
| `--compression_ratio` | `0.5` | Fraction of context to keep (0–1) |
| `--num_examples` | `100` | Number of examples to evaluate |
| `--model_name` | `microsoft/Phi-3.5-mini-instruct` | HuggingFace model ID |
| `--device` | auto | `cuda` or `cpu` |
| `--output_dir` | `results` | Directory to save JSON results |
| `--dynamic_compression` | `False` | Enable dynamic ratio selection |
| `--skip_verification_if_confident` | `False` | Skip verifier when reasoner is confident |

### `experiments/ablation.py`

Runs a sweep over all ablation configurations (full system vs. no keywords, no entities, no position, no role-awareness, etc.) and writes comparative JSON results.

```bash
python experiments/ablation.py --dataset narrativeqa --num_examples 50
```

### `experiments/run_parallel.py`

Distributes compression-type experiments across multiple GPUs.

```bash
python experiments/run_parallel.py --num_examples 50 --num_gpus 2
```

---

## Datasets

| Key | Dataset | Task type |
|---|---|---|
| `narrativeqa` | NarrativeQA | Long-form story QA |
| `squad` | SQuAD 1.1 | Factoid extractive QA |
| `hotpotqa` | HotpotQA | Multi-hop reasoning |
| `drop` | DROP | Discrete reasoning / arithmetic |
| `cnn_dailymail` | CNN/DailyMail | Abstractive summarization |
| `eli5` | ELI5 | Long-form explanation generation |
| `qasper` | QASPER | Scientific paper QA |

---

## Compression Strategies

| Type | Description |
|---|---|
| `none` | No compression — full context passed between agents |
| `fixed` | Fixed-ratio truncation (keeps first N% of tokens) |
| `role_specific` | **Core method** — scores sentences by relevance to the next agent's role |
| `semantic` | Removes redundancy, fuses similar sentences, selects by relevance |
| `dynamic` | Automatically adjusts ratio based on question complexity and document density |

### Role-Specific Compression (Core Idea)

When compressing the Retriever's output **for the Reasoner**, the system scores sentences by:
- Keyword overlap with the question
- Presence of named entities, numbers, and dates
- Positional importance (first/last sentences)

When compressing the Reasoner's output **for the Verifier**, scoring shifts to prioritize:
- The candidate answer sentence
- Reasoning chain and justifications
- Contradictions or uncertainty markers

---

## Metrics

Results are saved as JSON in `results/` and include:

| Metric | Description |
|---|---|
| `f1` | Token-level F1 between prediction and ground truth |
| `exact_match` | Strict exact match accuracy |
| `rouge_l` | ROUGE-L score (for summarization tasks) |
| `context_size_*` | Token count entering each agent |
| `latency` | Wall-clock time per example |
| `memory_mb` | Peak GPU/CPU memory usage |
| `success_rate` | Fraction of examples with successful inference |

---

## Visualizations

```bash
# Generate all figures from saved results
python experiments/visualize.py --results_dir results --output_dir results/figures

# Interactive exploration
python experiments/interactive_viz.py --results_dir results
```

Figures include F1 vs. compression ratio curves, context size reduction bars, per-dataset breakdowns, latency comparisons, and ablation heatmaps.

---

## Multi-Turn Conversations

```python
from agents.conversation_chain import ConversationChain

chain = ConversationChain(
    model_name="microsoft/Phi-3.5-mini-instruct",
    compression_type="role_specific",
    max_history_turns=5
)

chain.start_conversation(document="...")

result1 = chain.process_turn("Who founded the company?")
result2 = chain.process_turn("When did they start it?")   # resolves "they"
```

The conversation chain maintains a rolling history window and resolves pronouns / co-references across turns before invoking the agent pipeline.

---

## Kaggle / Cloud Setup

See [KAGGLE_SETUP_GUIDE.md](KAGGLE_SETUP_GUIDE.md) for a step-by-step notebook setup, including GPU checks, full experiment sweeps, and downloading results.

---

## Ablation Configurations

The ablation framework (`experiments/ablation.py`) tests these configurations:

| Config | Description |
|---|---|
| `full` | Full system with all scoring components |
| `no_keyword_scoring` | Disables keyword overlap — tests its importance |
| `no_entity_scoring` | Disables entity-based scoring |
| `no_position_scoring` | Disables position-based scoring |
| `no_role_awareness` | Same scoring for all roles (removes role specificity) |
| `only_keywords` | Uses only keyword scoring |
| `only_entities` | Uses only entity scoring |

---

## License

MIT
