# Instruction-Preserving KV-Cache Compression for Long-Context LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A reproducible research study investigating KV-cache compression and instruction retention trade-offs for long-context LLM inference, prepared for IIT Bombay's *"Efficiency of Long Context Inference"* project application.

---

## Research Question

> **How much can the Key-Value (KV) cache be compressed before long-context model quality degrades, and does protecting instruction-critical positions preserve task behavior at higher compression?**  
> While attention-guided cache eviction reduces memory footprint, standard compression policies suffer from uneven degradation, frequently evicting early system instructions and constraints. This project implements a simplified SnapKV-style attention selection method and evaluates a controlled instruction-protected extension to determine whether pinning critical token spans sustains task adherence under severe cache compression.

---

## Key Experimental Results

Evaluated across **480 controlled benchmark trials** on `EleutherAI/pythia-70m` across sequence lengths (512–2048 tokens) with 3 repeats per configuration:

| Policy | Budget Ratio | KV Cache Retained (%) | Instruction Retention | Relative Quality | KV Cache Footprint (MB) | Throughput (tok/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full KV (Baseline A)** | 1.000 | 100.0% | **93.75%** | 1.000 | 13.02 MB | 107.4 tok/s |
| **Recent Window (Baseline B)** | 0.500 | 50.0% | **0.00%** | 0.000 | 6.51 MB | 114.4 tok/s |
| **Recent Window (Baseline B)** | 0.250 | 25.0% | **0.00%** | 0.000 | 3.25 MB | 116.4 tok/s |
| **Recent Window (Baseline B)** | 0.125 | 12.5% | **0.00%** | 0.000 | 1.63 MB | 119.3 tok/s |
| **Attention-Guided (Method C)** | 0.500 | 50.0% | **37.50%** | 0.400 | 6.51 MB | 114.8 tok/s |
| **Attention-Guided (Method C)** | 0.250 | 25.0% | **56.25%** | 0.600 | 3.25 MB | 115.8 tok/s |
| **Attention-Guided (Method C)** | 0.125 | 12.5% | **37.50%** | 0.400 | 1.63 MB | 117.7 tok/s |
| **Protected Attention (Extension D)** | 0.500 | 50.0% | **93.75%** | 1.000 | 6.51 MB | 113.3 tok/s |
| **Protected Attention (Extension D)** | 0.250 | 25.0% | **100.00%** | **1.067** | 3.25 MB | 116.4 tok/s |
| **Protected Attention (Extension D)** | 0.125 | 12.5% | **100.00%** | **1.067** | **1.63 MB** | **117.2 tok/s** |

### Key Findings
1. **Sliding-Window Eviction Fails Catastrophically**: Recent-window policies discard early instructions, dropping retention to **0.0%** across all budgets.
2. **Attention-Guided Selection Mitigates Loss**: SnapKV-style selection restores retention to 37.5%–56.25% by retaining salient historical keys.
3. **Instruction-Protected Extension Preserves 100% Quality**: Explicitly pinning instruction spans guarantees **100% instruction retention at 12.5% cache budget** ($8\times$ compression), yielding an **87.5% memory reduction** and a **9.1% throughput gain**.

---

## Publication-Ready Plots

All plots are generated automatically and saved in `results/plots/`:

- **Key Trade-off**: `results/plots/retention_vs_cache.png` (Instruction Retention vs. Retained Cache %)
- **Memory Scaling**: `results/plots/memory_vs_retention.png` (KV Retention vs. Peak Memory)
- **Throughput**: `results/plots/throughput_vs_retention.png` (KV Retention vs. Decode Speed)
- **Latency**: `results/plots/latency_vs_context.png` (Context Length vs. Decode Latency)

---

## Repository Structure

```
├── kv_cache/
│   ├── __init__.py
│   ├── baseline.py          # Policy A: Full KV & Policy B: Recent Window
│   ├── attention_select.py  # Method C: SnapKV-style attention-guided selection
│   ├── protected_kv.py      # Extension D: Instruction-protected attention policy
│   └── utils.py             # Memory/latency profilers & cache slicing helpers
├── benchmarks/
│   ├── __init__.py
│   ├── dataset_generator.py # Controlled synthetic task generator with distractor injections
│   ├── prompts.jsonl        # Pre-generated synthetic long-context prompts
│   ├── run_benchmark.py     # Main experiment harness
│   └── evaluate.py          # Metric aggregator and matplotlib/seaborn plotter
├── tests/
│   └── test_kv_cache.py     # Unit test suite verifying cache pruning and protected sets
├── results/
│   ├── results.csv          # Raw per-trial benchmark records (480 runs)
│   └── plots/               # High-resolution benchmark figures
├── report/
│   └── mini_report.md       # Comprehensive 2-4 page research mini-report
├── run_all.py               # Single-command end-to-end reproducer
├── requirements.txt         # Pinned lightweight dependencies
└── LICENSE                  # MIT License
```

---

## Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/SudiptaSanki/KV-Cache-Compression.git
cd KV-Cache-Compression
pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
python -m unittest discover tests
```

### 3. Reproduce Full Benchmark (One-Click)
```bash
python run_all.py --limit-prompts 16 --repeats 3 --max-new-tokens 25
```

---

## Honest Framing & Scope
- **What this project demonstrates**: A faithful, reproducible study of attention-based KV-cache compression inspired by SnapKV, with a controlled protected-token extension evaluated under synthetic long-context instruction tasks.
- **What this project does not claim**: This project does not claim a new universal state-of-the-art compression algorithm or a solution to arbitrary million-token context. It demonstrates controlled experimental design, system profiling, and empirical reasoning regarding the efficiency-quality Pareto frontier.

---

## Citations & Prior Work
1. **SnapKV**: Li, Y., et al. (2024). *SnapKV: LLM Knows What You are Looking for Before Generation*. [arXiv:2404.14469](https://arxiv.org/abs/2404.14469).
2. **The Pitfalls of KV Cache Compression**: (ACL 2026). *The Pitfalls of KV Cache Compression: Uneven Degradation and Vulnerability in Instruction Following*. [ACL Anthology](https://aclanthology.org/2026.acl-long.1926/).
3. **StreamingLLM**: Xiao, G., et al. (2023). *Efficient Streaming Language Models with Attention Sinks*. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453).
