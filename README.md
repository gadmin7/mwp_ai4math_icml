# Sequential Fine-Tuning with Progressive LoRA Rank Shrinking for Math Word Problems

This repository contains code and experiments for our ICML 2025 paper that introduces a novel **Progressive LoRA Rank Shrinking (PLRS)** strategy in a **Sequential Fine-Tuning (SeqFT)** setup for mathematical reasoning. Our approach shows consistent improvement (2–7% EM gains) over standard fine-tuning strategies on the MATH benchmark.

## 🧠 Abstract

This work investigates a human-inspired sequential fine-tuning (SeqFT) method to improve the performance of resource-constrained large language models (LLMs) on math word problems. Instead of training on the entire dataset simultaneously, models are exposed to progressively harder tasks level by level, while earlier data is periodically reintroduced to mitigate catastrophic forgetting.

We propose a **Progressive LoRA Rank Shrinking (PLRS)** mechanism, which progressively reduces the LoRA rank across stages to minimize forgetting while preserving parameter efficiency. Evaluations on the MATH dataset with models like LLaMA-3.2 (1B, 3B), Qwen-2.5-Math (1.5B), and Qwen-2 (0.5B) show that our approach consistently outperforms single-stage fine-tuning and naive curriculum learning.

---

## 📂 Repository Structure

```bash
.
├── README.md
├── evaluation/
│   └── exact_match_accuracy.ipynb   # For computing EM from predictions
├── experiments/
│   ├── llama1b/
│   │   ├── plrs_llama1b.ipynb
│   │   └── ...
│   ├── llama3b/
│   │   ├── plrs_llama3b.ipynb
│   │   └── ...
│   └── qwen/
│       ├── plrs_qwen.ipynb
│       └── ...
├── results/
│   └── *.csv                        # Output predictions used in evaluation
└── requirements.txt / environment.yml
