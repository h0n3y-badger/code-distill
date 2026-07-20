# code-distill

A small, honest pipeline for **distilling a compact, fast, local coding model** from
a larger local teacher — with **execution-based verification** and
**contamination-controlled evaluation** throughout. Runs end-to-end on a single
16 GB consumer GPU.

The goal was not to top a leaderboard. It was to answer a concrete question —
*"can I make a small model that's genuinely better at the languages I care about,
and prove it honestly?"* — and to avoid the **benchmaxxing trap** (tuning toward, or
contaminating on, a public benchmark) at every step.

## What it does

```
Qwen2.5-Coder-14B  ──generate──▶  synthetic coding tasks + solutions + tests
   (teacher, local)                         │
                                    execution / gcc verify  ──▶ keep only passing
                                            │
Qwen2.5-Coder-7B   ──QLoRA SFT──▶  merge 16-bit ──▶ GGUF Q5_K_M ──▶ your model
   (student)                                                │
                                          execution pass@1 on a held-out,
                                          training-disjoint eval set
```

- **Seeded self-instruct** over per-language domains (no nonsense combos like
  "memory management in HTML").
- **Rejection sampling:** Python is executed, C is `gcc`-compiled-and-run; only
  samples that pass their own tests are kept.
- **Contamination-controlled eval:** a dedicated eval set, deduplicated (exact +
  fuzzy) against the training data. Never HumanEval.

## Results

Distilling **Qwen2.5-Coder-7B-Instruct** (Python + C focus), scored by actually
compiling/running the model's output against held-out tests:

| Model | Python | C | **Total** |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct (base) | 77.8% | 47.0% | **60.8%** |
| distilled student | 79.6% | **63.6%** | **70.8%** |

**+10 points overall, driven by C (+16.6).** The base was weak at C (missing
`#include`s, wrong function/type contracts — modern `gcc` hard-errors on these);
training on self-contained, contract-explicit C data fixed a large chunk of it.

A small sibling — a **Python-only 1.5B for an old ThinkPad (CPU inference)** — is
built by the same pipeline (`train_1b.py` / `build_1b.sh`): **77.8% Python pass@1**
(vs 74.1% for the stock 1.5B) in a **941 MB** Q4_K_M GGUF — essentially tying the 7B
at Python while running on CPU. See `RESULTS_1B.md`.

## Quickstart

```bash
make install            # uv venv (Python 3.12; Arch ships 3.14, which Unsloth lags) + deps
make install-train      # unsloth (separate; heavy)

# 1. serve your teacher (edit serve_teacher.sh for your GGUF), then:
make generate           # teacher -> raw.jsonl (execution-verified)
make dedup              # -> clean.jsonl (+ held-out split)

# 2. unload teacher, then:
make train              # QLoRA the 7B student -> merged 16-bit
make quantize           # -> gguf/…-Q5_K_M.gguf

# 3. eval
./serve_student.sh &    # serve the quantized student on :8091
make eval               # execution pass@1, Python + C
```

Tune the seed grid in `config.py` — that grid is the single biggest lever on what
the student ends up good at.

## Hardware notes (16 GB VRAM)

- **Teacher must be dense, not MoE.** A 35B-A3B MoE teacher crawled (~2 kept/min):
  it doesn't fit 16 GB, so cold experts spill to RAM and the *diverse* generation
  sweep triggers constant PCIe fetches. A **dense 14B** fully fits (~11 GB) and runs
  much faster and consistently.
- Teacher and 7B student **can't co-load** — generate, unload, then train.
- Teacher is driven via llama.cpp's own `llama-server` (a bundled runtime worked
  where LM Studio's loader SIGSEGV'd on the same GGUF); see `serve_teacher.sh`.

## Lessons learned (including the ones that didn't work)

Kept here because the negative results are the honest part:

- **Prompting a 7B to "emit all `#include`s" did nothing** — it won't override a
  learned habit. **Retraining on include-complete data did** (raw include-omissions
  dropped). Data > prompts for behavioral fixes.
- **Doubling the C data moved pass@1 by noise.** Small models have a real ceiling;
  once the mechanical failures (includes) are gone, the rest (naming/contract
  mismatch, genuine logic bugs) don't fall to more of the same data.
- **Always compare against the base.** A refinement round looked flat run-to-run —
  but vs. the *stock* base the core distillation was clearly worth it (+10 pts).
  Without the baseline you can't tell adaptation from sampling noise.
- Keep long generation **separate** from the train/eval tail (a >60 min combined job
  got killed mid-merge).

## Repo layout

| file | role |
|---|---|
| `config.py` | central config + the seed grid |
| `gen.py` | teacher generation + execution/gcc verification |
| `dedup.py` | near-dup removal + held-out split |
| `train.py` / `train_1b.py` | QLoRA SFT (7B / Python-only 1.5B) |
| `quantize.sh` | HF → f16 GGUF → Q5/Q4 quant |
| `eval.py` / `eval_py.py` | execution pass@1 |
| `baseline.sh` | stock-model baseline for honest comparison |
| `diag_c.py`, `test_autoinclude.py`, `c_postprocess.py` | C failure diagnosis + include fixer |

## License

Apache-2.0. Base and teacher (Qwen2.5-Coder 7B/14B) are Apache-2.0 — which places no
restriction on using model outputs to train other models — and the training data is
fully self-generated (no scraped corpus, no ToS-restricted API). Clean to reuse and
redistribute, including commercially.
