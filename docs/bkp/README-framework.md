<div align="center">


### **Enterprise-Grade Reinforcement Learning for Large-Scale Model Training**
### **High-Performance Rollout • Low Precision Training • Production Stability**



</div>

---


## Latest Updates


## What is the framework?


> *"A journey of a thousand the framework begins with a single rollout."* — the framework focuses on the low-level system optimizations that make large-scale RL stable, efficient, and reproducible.

---


## Key Features

### 🌪️ Advanced MoE & Low-Precision Training

*   **Unified FP8 Pipeline**: The first framework to implement end-to-end FP8 sampling and training. By unifying precision across rollout and training, the framework eliminates the quantization-induced discrepancy that causes RL collapse in large MoE models.
*   **Rollout Routing Replay (R3)**: Records expert routing decisions during SGLang inference and replays them during training to ensure bit-wise expert alignment.
*   **INT4 QAT Support**: Recommendation for 1TB+ models to enable single-machine (e.g., H200) deployment by significantly reducing memory footprint.

### 🛡️ Eliminating Train-Inference Mismatch

*   **Bit-wise Identical Training and Inference Log Probs**: System-level solution achieving deterministic forward/backward passes through kernel-level optimization (FlashAttention-3, DeepGEMM).
*   **Algorithmic Correction (TIS/MIS)**: When mismatch is unavoidable, the framework provides **Truncated Importance Sampling (TIS)** and **Masked Importance Sampling (MIS)** to mitigate off-policy bias and prevent training divergence.

### ⚡ Extreme Performance & Efficiency

*   **Speculative RL Training**: Achieve **25%+ rollout speedup** by using an **Online SFT Draft Model**. Unlike frozen draft models, the framework updates the draft policy during RL to prevent policy drift.
*   **Zero-Copy Weight Sync**: Optimized weight refit via **CUDA IPC zero-copy mapping**, async tensor gathering, and bucketed flattening. Sync time reduced by 50% compared to standard HTTP/RPC transfers.
*   **Partial Rollout & Over-Sampling**: Handles the "Long-Tail Effect" in multi-turn RL by over-sampling requests and recycling half-finished trajectories to maximize GPU utilization.

## Model Support & Training Diversity

### 🏗️ Supported Models
the framework supports a wide range of state-of-the-art architectures, with a special emphasis on **DeepSeek, Qwen, Llama** and mainstream models.

| Family | Supported Models |
| :--- | :--- |
| **DeepSeek** | **R1, V3, V3.2** |
| **Qwen** | **Qwen 2, 2.5, 3** |
| **Llama** | **Llama 3, 3.1, 3.3, 4** |
| **Gemma** | **Gemma 2, 3, 3N** |
| **GLM** | **GLM-4.5, GLM-4.6, GLM-4.7** |
| **MiniMax** | **M2, M2.1** |
| **Others** | **Mistral, Mixtral, Phi, gpt-oss and any model supported by SGLang and Megatron** |

### 🧩 Diverse Training Scenarios
the framework is designed to handle the complexity of modern RL workloads across various dimensions:
*   **Multi-Turn Interaction**: Optimized for complex, multi-round conversations and tool-use scenarios.
*   **VLM & LLM Support**: Unified framework for both Vision-Language and pure Text models.
*   **Reasoning & Coding**: Specific recipes and optimizations for **Reasoning (Math/Logic)** and **Coding Agent** tasks.
*   **Multi-Agent Training**: Support for advanced co-training and collaborative multi-agent reinforcement learning.

---

## Quick Start

### Installation

We recommend using our official Docker image for the best performance and compatibility:

```bash
# Pull the latest image

# Or install from source
pip install -r requirements.txt
pip install -e .
```

### Launch Training

the framework provides a unified entry point for complex RL tasks. Here is an example of FP8 GRPO training for Qwen3:

```bash
python train.py \
    --advantage-estimator grpo \
    --model-name qwen3-30b-a3b \
    --hf-checkpoint /path/to/qwen3-30b-a3b-hf \
    --rollout-batch-size 512 \
    --n-samples-per-prompt 8
```


---

## Roadmap

### ✅ Completed

- [x] **Unified FP8** E2E Training & Rollout
- [x] **INT4 Quantization-Aware Training (QAT)**: Single-machine 1TB models
- [x] **Speculative RL** with Online SFT
- [x] **Support DeepSeek V3.2 Models**
- [x] **VLM Multi-Turn Training**
- [x] **Aligning SGLang with Megatron in Dense Models**
- [x] **Rollout Routing Replay (R3)**

### 🏗️ In Progress & Planned

- [ ] **Zero mismatch for MoE RL**
- [ ] **Aligning SGLang with Megatron in MoE Models**
- [ ] **Diffusion RL** Support
- [ ] **Omni RL** Support
- [ ] **Diffusion LLM RL** Support
- [ ] **Elastic Resource Scheduling**: Dynamic scaling of rollout vs. training workers



---

## Acknowledgements

the framework is built upon the shoulders of giants in the LLM infrastructure ecosystem:


---

## Links

*   **Developer Guide**: Check the `docs/` and `examples/` directories for in-depth technical notes.

<div align="center">


</div>
