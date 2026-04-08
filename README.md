# Applying LoRA to GPT-2 for Positive Review Generation

Fine-tuning GPT-2 using **Low-Rank Adaptation (LoRA)** on IMDB positive reviews to steer the model toward generating sentiment-positive movie review text.

---

## Overview

Large language models like GPT-2 are expensive to fully fine-tune. This project applies **parameter-efficient fine-tuning (PEFT)** via LoRA, which freezes the base model weights and injects trainable low-rank matrices into the attention layers. The result is a model that generates positive-sentiment movie review text while training only ~0.1% of the original parameters.

**Key idea:** Instead of updating all 117M parameters of GPT-2, LoRA decomposes weight updates into two small matrices (rank `r=8`), reducing trainable parameters dramatically while preserving model quality.

---

## Project Structure

```
├── model.py          # LoRA config and model setup via PEFT
├── data_utils.py     # Dataset loading, filtering, and tokenization
├── train.py          # Training loop using HuggingFace Trainer
├── inference.py      # Load fine-tuned model and generate text
├── requirements.txt  # Dependencies
```

---

## Model Architecture

| Component        | Detail                          |
|------------------|---------------------------------|
| Base Model       | GPT-2 (117M parameters)         |
| Fine-tuning Method | LoRA via PEFT                 |
| LoRA Rank (`r`)  | 8                               |
| LoRA Alpha       | 32                              |
| Target Modules   | `c_attn` (attention projection) |
| Dropout          | 0.05                            |
| Task Type        | Causal Language Modeling        |

---

## Dataset

- **Source:** IMDB dataset (HuggingFace `datasets`)
- **Filter:** Positive reviews only (`label == 1`)
- **Subset:** 500 reviews used for training
- **Split:** 80% train / 20% validation
- **Tokenization:** GPT-2 tokenizer, max length 128, padded to max length
- **Labels:** Input IDs copied as labels (standard causal LM setup)

---

## Training Configuration

| Parameter         | Value         |
|-------------------|---------------|
| Epochs            | 3             |
| Learning Rate     | 2e-4          |
| Eval Strategy     | Every 20 steps|
| Save Strategy     | Per epoch     |
| Logging           | TensorBoard   |

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

The fine-tuned LoRA adapter will be saved to `./final_lora_model`.

### 3. Run inference

```bash
python inference.py
```

Example prompt and output:

```
Prompt: "The movie was "

Result 1: The movie was a remarkable achievement in storytelling...
Result 2: The movie was one of the most compelling films I have seen...
Result 3: The movie was beautifully crafted with exceptional performances...
```

---

## Why LoRA?

Full fine-tuning of GPT-2 on a small dataset risks catastrophic forgetting and is computationally expensive. LoRA addresses this by:

- Keeping base model weights **frozen**
- Adding lightweight trainable matrices to the `c_attn` attention layer
- Achieving similar fine-tuning quality at a fraction of the compute cost
- Saving only the adapter weights (~few MB) rather than the full model

---

## Results

The fine-tuned model consistently generates more positive-toned continuations compared to the base GPT-2 model when given neutral or ambiguous prompts about movies, demonstrating that LoRA successfully steered the model's generative behavior toward the target sentiment.

---

## Dependencies

Key libraries used:

- `transformers` — GPT-2 base model and tokenizer
- `peft` — LoRA implementation
- `datasets` — IMDB dataset loading
- `torch` — Training backend
- `tensorboard` — Training monitoring

See `requirements.txt` for full dependency list.

---

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [HuggingFace PEFT Library](https://github.com/huggingface/peft)
- [GPT-2 Model Card](https://huggingface.co/gpt2)