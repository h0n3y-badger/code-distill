# 1.5B Python model (for ThinkPad W541) — results

Execution-based pass@1 on a held-out, training-disjoint Python eval (54 tasks),
Q4_K_M quant, same prompts. v2 adds complete-program training data.

| model | Python pass@1 |
|---|---|
| stock Qwen2.5-Coder-1.5B-Instruct | 74.1% (40/54) |
| distilled Python student (v1, functions only) | 77.8% (42/54) |
| **distilled Python student (v2, + complete programs)** | **81.5%** (44/54) |

v2 also writes complete, interactive programs (using `input()`, menus) from natural
requests — not just bare functions. **941 MB**, runs on CPU (~10 tok/s on the W541).
Run it with the chat template + low temperature — see `W541_USAGE.md`.
