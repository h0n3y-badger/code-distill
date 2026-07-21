---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
library_name: gguf
pipeline_tag: text-generation
tags:
  - code
  - python
  - c
  - qwen2.5-coder
  - distillation
  - qlora
  - gguf
language:
  - en
---

# Qwen2.5-Coder-7B-Instruct — Python/C self-distill (GGUF)

A QLoRA fine-tune of **Qwen2.5-Coder-7B-Instruct**, specialized for **Python and C**
by self-distillation from the larger **Qwen2.5-Coder-14B-Instruct** teacher. Trained
entirely on locally-generated, **execution-verified** synthetic data on a single
16 GB consumer GPU (RTX 5070 Ti).

> **TL;DR** — On a held-out, execution-scored eval it lifts the base model from
> **60.8% → 74.2%** pass@1: **C** jumps (47% → 64%) and **Python** jumps too
> (78% → 87%). It also writes complete, interactive programs from natural asks,
> not just bare functions. Specialized for Python/C — read the honest caveats below.

## Results (execution-based pass@1)

Same eval set, same Q5_K_M quant, same prompts. The base is the exact weights this
model was fine-tuned from, so this isolates what the fine-tune did.

| Model | Python | C | **Total** |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct (base) | 77.8% (42/54) | 47.0% (31/66) | **60.8%** (73/120) |
| **This model** | 87.0% (47/54) | **63.6%** (42/66) | **74.2%** (89/120) |
| Δ | **+9.2** | **+16.6** | **+13.4** |

*pass@1 = the model's code was compiled/run against held-out tests and had to pass.
Numbers carry ±1–2 samples of run-to-run sampling noise (temperature 0.2).*

## How it was made

1. **Teacher generates data.** Qwen2.5-Coder-14B-Instruct (served locally via
   llama.cpp) produces coding tasks + reference solutions + tests via seeded
   self-instruct across per-language domains (Python, C, plus some HTML/Java).
2. **Rejection sampling.** Python solutions are executed and C solutions are
   `gcc`-compiled-and-run against their tests; **only samples that pass are kept.**
   The teacher's C prompt is constrained to emit self-contained code (all `#include`s)
   with explicit function/type contracts.
3. **QLoRA SFT** of the 7B student (Unsloth, r=16, 2 epochs, 4-bit base) on ~1.25k
   deduplicated samples — a mix of execution-verified **functions** and complete,
   run-verified **programs from natural requests** — then merged to 16-bit and
   quantized to GGUF Q5_K_M.

> Run with the chat template applied and low temperature (~0.2) — via `llama-server`
> or `llama-cli -cnv`. Raw `-p` completion mode at default temp will underperform.

## Evaluation methodology (and why it's not HumanEval)

The eval is **execution-based** on a **dedicated 120-sample set that is disjoint
from the training data** (exact + fuzzy dedup). It is deliberately **not** HumanEval
or MBPP: the point was to avoid the "benchmaxxing" trap of training toward a public
benchmark (or worse, contaminating on it). The trade-off is honesty over
comparability — these numbers are meaningful *relative to the base on this set*, not
directly comparable to leaderboard scores.

## Honest limitations

- **Specialized, not universally better.** Gains are in Python and C specifically;
  don't expect improvement in other languages.
- **Eval favors the model's home distribution.** The eval tasks come from the same
  teacher/family and follow the same conventions the model was trained on (e.g. the
  C convention of "no `main()`, self-contained solution"). A neutral benchmark would
  likely show a smaller gap.
- **C hit a ceiling.** C stayed at ~64% across data-scaling rounds — the remaining
  C failures are contract/naming and genuine logic bugs that more of the same data
  doesn't fix at 7B. Python and complete-program ability, by contrast, kept improving.
- English-only prompts; Python/C focus (HTML/Java were minor in training).

## Usage

Q5_K_M GGUF (~5.1 GB), runs anywhere llama.cpp does.

```bash
# llama.cpp
llama-server -m qwen-coder-7b-mine-Q5_K_M.gguf -ngl 99 -c 4096
# or LM Studio: drop the .gguf into ~/.lmstudio/models/<folder>/ and load it
```

There is also a **Python-only 1.5B sibling** (941 MB, runs on CPU) built by the same
pipeline — see the repo.

## Provenance & license

- **License:** Apache-2.0 — inherited cleanly. Both the base (Qwen2.5-Coder-7B) and
  the teacher (Qwen2.5-Coder-14B) are Apache-2.0, which places no restriction on
  using model outputs to train other models. Training data is fully self-generated
  (no scraped corpus, no ToS-restricted API).
- **Base model:** [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- **Teacher:** Qwen/Qwen2.5-Coder-14B-Instruct
- **Full training/eval pipeline:** https://github.com/h0n3y-badger/code-distill

*Built as a learning project in honest, contamination-controlled small-model
distillation. The interesting artifact is the reproducible pipeline — see the repo.*
