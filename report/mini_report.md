# Instruction-Preserving KV-Cache Compression for Long-Context Inference

**Author / Candidate:** Sudipta Sanki  
**Project Application:** IIT Bombay — *Efficiency of Long Context Inference*  
**Date:** September 2026  
**Repository:** [https://github.com/SudiptaSanki/KV-Cache-Compression](https://github.com/SudiptaSanki/KV-Cache-Compression)

---

## 1. Executive Summary & Research Question

### 1.1 Research Question
> **How much can the Key-Value (KV) cache be compressed before long-context model quality degrades, and does protecting instruction-critical positions preserve task behavior at higher compression ratios?**

As large language models (LLMs) scale to long-context inputs (16K–1M tokens), memory consumption during autoregressive generation becomes memory-bandwidth and capacity bound by the Key-Value (KV) cache. In causal transformers, the KV cache grows $O(N)$ with context length $N$, quickly surpassing the model parameter weights in memory footprint and degrading decode throughput.

While recent attention-guided pruning techniques such as **SnapKV** demonstrate that significant portions of the KV cache can be evicted by monitoring attention patterns, recent analyses (*The Pitfalls of KV Cache Compression*, ACL 2026) report that cache eviction degrades task capabilities unevenly: early task instructions and safety constraints are frequently evicted before general context.

This study implements and evaluates a controlled KV cache compression system with four policies:
1. **Full KV Baseline (Policy A)**: Standard uncompressed reference cache ($100\%$ budget).
2. **Recent-Window Baseline (Policy B)**: Sliding-window with attention sinks (*StreamingLLM* style).
3. **Attention-Guided Selection (Method C)**: Simplified *SnapKV*-style top-$K$ historical selection pooled from observation window queries.
4. **Instruction-Protected Attention Selection (Extension D)**: A controlled extension that explicitly pins instruction-critical token spans into the cache budget, allocating residual capacity to attention-scored historical tokens.

---

## 2. Theoretical Background & Hypotheses

### 2.1 The KV Cache Bottleneck
During autoregressive generation at step $t$, the query vector $q_t \in \mathbb{R}^{d}$ must attend to all preceding keys $K_{1:t-1} \in \mathbb{R}^{(t-1) \times d}$ and values $V_{1:t-1} \in \mathbb{R}^{(t-1) \times d}$. For an $L$-layer model with $H$ attention heads per layer and head dimension $d_h$, the total KV cache memory footprint is:
$$\text{Memory}_{\text{KV}} = 2 \times B \times L \times H \times N \times d_h \times \text{bytes\_per\_elem}$$

For long sequences, memory access overhead to retrieve this cache at every decode step bounds memory bandwidth and limits batch concurrency.

### 2.2 Working Hypotheses
1. **Hypothesis 1 (Uniform Eviction Failure)**: Uniform recent-window eviction reduces cache memory linearly but catastrophically evicts early task instructions, reducing instruction retention to near zero in long contexts.
2. **Hypothesis 2 (Attention Saliency Advantage)**: Attention-guided selection (SnapKV) preserves salient context more effectively than sliding windows at equivalent budgets by utilizing observation query attention weights.
3. **Hypothesis 3 (Instruction-Protection Pareto Superiority)**: Explicitly protecting task-critical instruction positions ensures $100\%$ instruction survival even under severe compression ($12.5\%$ cache budget), maintaining Full-KV fidelity while retaining an $8\times$ memory reduction.

---

## 3. System Architecture & Algorithms

The system is organized into modular components adhering to the build specification:

```
long-context-kv-research/
├── kv_cache/
│   ├── baseline.py          # Full KV & Recent Window policies
│   ├── attention_select.py  # SnapKV-style attention-guided selection
│   ├── protected_kv.py      # Extension: Protected instruction policy
│   └── utils.py             # Memory/latency profilers & cache pruners
├── benchmarks/
│   ├── dataset_generator.py # Synthetic long-context prompt generator
│   ├── prompts.jsonl        # Multi-domain long-context task dataset
│   ├── run_benchmark.py     # Main experiment harness (3 repeats)
│   └── evaluate.py          # Scoring, relative quality, and plot generator
├── results/
│   ├── results.csv          # Raw per-trial benchmark records
│   └── plots/               # Publication-grade figures
└── report/
    └── mini_report.md       # Research mini-report
```

### 3.1 Policy Implementations

#### Policy A: Full KV Baseline
Prefills the complete sequence and retains all $N$ token key-values throughout generation.

#### Policy B: Recent-Window Baseline
Retains $S = 4$ initial attention sink tokens (to avoid attention collapse as proved by StreamingLLM) plus the most recent $W = B - S$ tokens, where budget $B = \lfloor N \times r \rfloor$.

#### Method C: SnapKV-Style Attention-Guided Selection
Given the attention tensor $A^{(l)} \in \mathbb{R}^{H \times N \times N}$ extracted during prefill, we define an observation window of the last $W_{\text{obs}} = 32$ prompt tokens. The importance score for historical position $k$ is computed across layers and heads:
$$S(k) = \frac{1}{L \cdot H \cdot W_{\text{obs}}} \sum_{l=1}^{L} \sum_{h=1}^{H} \sum_{q=N-W_{\text{obs}}}^{N-1} A^{(l)}_{h, q, k}$$

Candidate historical positions outside the sink $S$ and observation window $W_{\text{obs}}$ are ranked by $S(k)$, retaining the top-$K$ positions to satisfy budget $B$.

#### Extension D: Instruction-Protected Attention Selection
Identifies token span indices $P = \{p_1, \dots, p_m\}$ corresponding to task-critical instructions. It reserves positions $P \cup S \cup W_{\text{obs}}$, and allocates the remaining budget $B_{\text{rem}} = B - |P \cup S \cup W_{\text{obs}}|$ to candidate historical positions with the highest attention scores:
$$\text{Indices}_{\text{retained}} = P \cup S \cup W_{\text{obs}} \cup \text{TopK}_{k \notin (P \cup S \cup W_{\text{obs}})}(S(k), B_{\text{rem}})$$

#### RoPE Relative Distance Preservation
When pruning the KV cache, the absolute sequence length is preserved in the generation loop by passing explicit `position_ids = [[N + step]]`. This prevents rotational embedding disorientation where queries would otherwise observe shifted relative angles.

---

## 4. Experimental Setup

- **Model**: `EleutherAI/pythia-70m` (eager attention implementation, `torch.float32` precision on CPU/CUDA).
- **Target Context Lengths**: 512, 1024, and 2048 tokens.
- **Cache Budgets**: $100\%$, $50\%$, $25\%$, and $12.5\%$ of the original KV cache sequence length.
- **Tasks**: Multi-domain synthetic instruction tasks across 8 operational domains (Database Choice, Network Ports, Encryption Standards, Logging Levels, Application Caches, Auth Headers, Deployment Regions, Retry Limits).
- **Distractors**: Harmless realistic technical text (Kubernetes ingress, network latency, distributed tracing, WAL archiving, TypeScript asset pipelines) separating early instructions from trailing queries.
- **Trials**: 480 total executions (16 prompts $\times$ 10 policy-budget configurations $\times$ 3 repeats with fixed seeds).
- **Evaluation Metrics**:
  - **Instruction Retention Rate**: Fraction of trials where the critical entity/rule is correctly recalled.
  - **Relative Quality**: $\text{Retention}_{\text{Policy}} / \text{Retention}_{\text{Full\_KV}}$.
  - **KV Cache Size (MB)**: Exact tensor memory footprint across all layers.
  - **Decode Throughput**: Tokens generated per second.

---

## 5. Empirical Results

The complete 480-run benchmark produced the following aggregated findings:

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

---

## 6. Key Insights & Analysis

### 6.1 The Catastrophic Failure of Sliding Windows
As demonstrated by the $0.0\%$ retention rate for Recent Window across all budgets, sliding-window eviction blindly prioritizes recency over semantic importance. In multi-turn dialogues or agentic workflows where constraints are defined early in the context, uniform windowing guarantees task failure.

### 6.2 Attention Saliency vs. Attention Dilution
Attention-guided compression recovers $37.5\% - 56.25\%$ retention without prior knowledge of instruction spans. However, as distractor context scales, query attention becomes diffuse across background tokens, occasionally evicting critical tokens that have lower cumulative attention scores than frequent repetitive phrases.

### 6.3 The Power of Explicit Instruction Protection (Extension D)
By explicitly protecting critical instruction spans, **Extension D achieves 100% retention at 12.5% cache budget** ($8\times$ compression). Because the non-instruction tokens are compressed using attention scores, the model avoids distraction while maintaining complete fidelity to the original specification. Furthermore, the reduced cache size reduces attention memory bandwidth during decode, lifting throughput from **107.4 tok/s to 117.2 tok/s (+9.1% speedup)**.

---

## 7. Failure Modes & Limitations

1. **Instruction Boundary Annotation**: In our benchmark, token spans were identified from template offsets. In arbitrary user prompts, boundary detection requires a secondary boundary identification module or parser.
2. **Fixed Observation Window**: Using a fixed observation window ($W_{\text{obs}} = 32$) assumes that query intent is concentrated in the immediate prompt suffix. For multi-faceted queries distributed across the prompt, adaptive observation windows are required.
3. **Head Uniformity**: Averaging attention scores across heads treats all attention heads equally, whereas certain retrieval heads ("induction heads") carry disproportionate importance for long-range retrieval.

---

## 8. Proposed 6-Month Roadmap for IIT Bombay Project

During the 6-month internship on *Efficiency of Long Context Inference*, this initial work will be expanded along three axes:
1. **Per-Head Adaptive Pruning**: Implement head-specialized KV selection, preserving diverse key positions per head rather than forcing identical indices across the layer.
2. **Dynamic Cross-Layer Eviction**: Allocate varying token budgets across layers (e.g., deeper layers often require fewer historical tokens than early sensory layers).
3. **Paged Attention & Triton Kernel Integration**: Replace PyTorch tensor slicing with custom Triton/CUDA paged memory kernels to eliminate memory fragmentation and achieve true zero-copy KV cache eviction in production engines (vLLM / TensorRT-LLM).

---

## 9. Target Resume Bullet

> *Implemented and evaluated attention-based KV-cache compression for long-context LLM inference, demonstrating an 8x cache reduction (13MB to 1.6MB) with 100% instruction retention under protected-token eviction.*

---

## 10. References

1. **SnapKV**: Li, Y., et al. (2024). *SnapKV: LLM Knows What You are Looking for Before Generation*. arXiv:2404.14469.
2. **The Pitfalls of KV Cache Compression**: (ACL 2026). *The Pitfalls of KV Cache Compression: Uneven Degradation and Vulnerability in Instruction Following*. ACL Anthology.
3. **StreamingLLM**: Xiao, G., et al. (2023). *Efficient Streaming Language Models with Attention Sinks*. arXiv:2309.17453.
