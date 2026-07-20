---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: gguf
pipeline_tag: text-generation
tags:
  - code
  - python
  - qwen2.5-coder
  - distillation
  - qlora
  - gguf
  - cpu
  - small
language:
  - en
---

# Qwen2.5-Coder-1.5B-Instruct — Python self-distill (GGUF)

A tiny, **Python-focused** QLoRA fine-tune of **Qwen2.5-Coder-1.5B-Instruct**,
distilled from a **Qwen2.5-Coder-14B-Instruct** teacher on locally-generated,
**execution-verified** Python data. Quantized to **Q4_K_M — 941 MB** — so it runs
comfortably on **CPU** on old / low-power hardware (built to run on a 2014 ThinkPad
W541, no GPU needed).

> **TL;DR** — 77.8% Python pass@1 in under 1 GB. That's +3.7 points over the stock
> 1.5B, and it essentially **ties a 7B model at Python** while being ~5× smaller.

## Results (execution-based pass@1, Python)

Same 54-sample held-out Python eval, same Q4_K_M quant. Base = the exact weights this
was fine-tuned from.

| Model | Python pass@1 |
|---|---|
| Qwen2.5-Coder-1.5B-Instruct (base) | 74.1% (40/54) |
| **This model** | **77.8%** (42/54) |

For context, on the same eval the 7B models scored 77.8% (base) / 79.6% (distilled)
at Python — so this 941 MB model is right there with them *for Python*.

*pass@1 = the model's code was executed against held-out tests and had to pass.
Carries ±1–2 samples of sampling noise.*

## How it was made

Teacher (Qwen2.5-Coder-14B) generates Python tasks + solutions + tests → each is
**executed**, only passing samples kept → QLoRA SFT of the 1.5B student (Unsloth,
r=16, 3 epochs) on ~376 Python samples → merged 16-bit → GGUF Q4_K_M. Same pipeline
as the 7B sibling, Python-only.

## Evaluation methodology

Execution-based pass@1 on a dedicated eval set **disjoint from training** (exact +
fuzzy dedup). Deliberately **not** HumanEval/MBPP — the goal was an honest,
contamination-controlled comparison against the base, not a leaderboard number.

## Honest limitations

- **Python only.** It was trained and evaluated on Python; don't expect other
  languages to benefit.
- **Modest, specialized gain.** +3.7 points over an already-decent base, on a
  same-distribution eval — a neutral benchmark would likely show less.
- Small model: fine for functions, scripts, and everyday Python help; not a
  reasoning-heavy or large-context coder.

## Usage (CPU-friendly)

Q4_K_M GGUF, 941 MB. On CPU (e.g. an old laptop):

```bash
# llama.cpp on CPU — no GPU offload
llama-cli   -m qwen-coder-1.5b-py-Q4_K_M.gguf -p "Write a Python function to ..."
llama-server -m qwen-coder-1.5b-py-Q4_K_M.gguf -c 4096          # OpenAI-compatible API
# or load the .gguf in LM Studio
```

## Provenance & license

- **License:** Apache-2.0. Base (Qwen2.5-Coder-1.5B) and teacher (Qwen2.5-Coder-14B)
  are both Apache-2.0 — no restriction on training from model outputs — and the data
  is fully self-generated (no scraped corpus, no ToS-restricted API).
- **Base model:** [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- **Teacher:** Qwen/Qwen2.5-Coder-14B-Instruct
- **Full pipeline:** https://github.com/sudo-lucap/code-distill

*A companion to the 7B Python/C distill — see the repo for the reproducible pipeline.*
