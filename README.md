# Sequential Fine-Tuning with Progressive LoRA Rank Shrinking for Math Word Problems

This repository accompanies our **AI4MATH@ICML 2025** paper on **Progressive LoRA Rank Shrinking (PLRS)** in a **Sequential Fine-Tuning (SeqFT)** framework for mathematical reasoning. Our method consistently improves exact match accuracy by **+2–7%** on the MATH benchmark over standard parameter-efficient fine-tuning techniques.

---

## 🧠 Abstract

This work investigates a human-inspired sequential fine-tuning (SeqFT) method to improve the performance of resource-constrained large language models (LLMs) on math word problems. Instead of training on the entire dataset simultaneously, models are exposed to progressively harder tasks level-by-level, while earlier data is periodically reintroduced to mitigate catastrophic forgetting. 

In addition, a strategy called **Progressive LoRA Rank Shrinking (PLRS)** is proposed, which progressively reduces the LoRA rank at each stage to prevent overwriting earlier learned parameters. Evaluations on the MATH dataset demonstrate that this approach consistently outperforms both naive multi-stage training and standard LoRA fine-tuning.

We show the individual impact of:
- ✅ Repeated data exposure  
- ✅ Difficulty-based task ordering (curriculum)  
- ✅ PLRS for improved retention and generalization  

---



## 📊 Baseline Comparisons

We evaluate six key configurations on the MATH benchmark:

| ID | Method | Description |
|----|--------|-------------|
| **Baseline 1** | Direct Baseline | Fine-tune on entire dataset in one stage, LoRA rank=32 |
| **Baseline 2** | Sequential No-Replay | Fine-tune level-by-level, discard earlier levels, LoRA rank=32 |
| **Baseline 3** | SNR + Rank Shrink | Same as SNR, but progressively reduce LoRA rank (256 → 16) |
| **Baseline 4** | Sequential Full Replay | Reintroduce previous levels at each stage, fixed LoRA rank=32 |
| **Baseline 5** | Replay + Rank Shrink | Replay all prior data and shrink LoRA rank (256 → 16) |
| **Baseline 6** | SFR + PLRS + No Final Shrink | Same as above, but stop shrinking at final stage (256 → 32) |

## 🧪 Running Experiments

The [`experiment_notebooks/`](./experiment_notebooks/) folder contains training notebooks for each model variant:

- `llama1b/`
- `llama3b/`
- `qwen/`

Each notebook will:
1. Fine-tune the model using specified curriculum stages and PLRS.
2. Automatically save predictions to `.csv` files.
3. (Optional) Push the trained model to the Hugging Face Hub.

### 🔐 Hugging Face Authentication

To push models or access gated models like LLaMA, first login:

```python
from huggingface_hub import login
login("your_huggingface_token_here")
```

Set save directory and base model:
```python
save_directory = "yourname/llama3b-math-plrs"
model_name = "meta-llama/Meta-Llama-3-1B-Instruct"
```

---

## 📊 Evaluation

After training, run the following notebook to compute exact match accuracy:
```bash
experiment_notebooks/evaluations/exact_match_accuracy.ipynb
```

This will read the generated `.csv` files containing model predictions and compute accuracy metrics on the MATH test set.

---
