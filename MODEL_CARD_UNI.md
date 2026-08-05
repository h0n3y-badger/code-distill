---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
  - qwen2.5
  - coding
  - tool-use
  - function-calling
  - gguf
  - local
language:
  - en
pipeline_tag: text-generation
---

# qwen-uni-7b (v12) — a small, local **universal** model

Distilled/adapted from **Qwen2.5-7B-Instruct** into a single 7B that does
**coding + web design + general chat + tool/MCP function-calling** — built on the
`code-distill` pipeline, run entirely on one 16 GB GPU.
GGUF: `qwen-uni-7b-v12-Q5_K_M.gguf` (fp16 weights also in this repo).

## ⚠️ REQUIRED: run it with a context window of **at least ~9K tokens** (16K recommended)

If you use this model with a tool/MCP server (e.g. **searxng web search**), set the
**context length to 9,216+** — 16,384 is a good default (the model supports 32,768).

**Why this matters:** an MCP server injects its tool schemas into the prompt
(searxng's four tools are ~1,600 tokens), and a **single web-search result is
~3,000 tokens**. Together that is **~5,000 tokens of prompt before the model writes
a word**. In a **4K** context those alone overflow the window, so the model has zero
room to reply — which surfaces in LM Studio as **"This message contains no content"**
and, if you nudge it, an endless re-search loop. This is a runtime **configuration**
issue, not a model defect: at ≥9K (ideally 16K) it answers normally, and multi-turn
searches don't choke. In LM Studio, set **Context Length** in the model's load config
and reload.

## The constant test (frozen suite, identical every version)
`eval_universal.py` scores C execution pass@1, HTML structural validity, tool-call
schema-correctness (canonical `<tool_call>`, parsed via llama.cpp `--jinja`), and chat.

| model | C exec | web valid | **tools** | chat |
|---|---|---|---|---|
| base-v3 (prior coding model) | 5/12 (42%) | 8/8 (100%) | **2/8 (25%)** | 100% |
| **qwen-uni-7b (v12)** | 5/12 (42%) | 8/8 (100%) | **8/8 (100%)** | 100% |

**Headline:** tool/MCP calling went **25% → 100%** (correct tool, correct arguments,
canonical `<tool_call>`), while coding/web/chat held. C sits in the **4–5/12 noise
band every iteration showed** — the documented small-model C ceiling.

## What each iteration fixed (v6 → v12)
- **v6** — the core win: switched from the coder base to the **general `Qwen2.5-7B-Instruct`**
  base because the *Coder* base ships the tool tokens `<tool_call>`/`</tool_call>` with an
  **all-zero (dead) embedding** (Qwen ties embed→output, so a zero row can never be emitted).
  Tools 25% → 100%.
- **v8** — presentation: emits code in proper ```fenced blocks```, multi-line-formatted,
  and switches modes (artifact → plain follow-up gets prose).
- **v9/v10** — the artifact-mode tool-lock: once an artifact was in history the model would
  refuse to call a tool and fabricate; fixed with multi-turn `enrich`/`research` data that
  place a `<tool_call>` *after* an artifact.
- **v12** — after a **large, messy** search result, a plain "just tell me" ask sometimes got
  an HTML/markdown *document* instead of prose; retrained on realistic big-blob results with
  a prose-vs-doc **intent** discriminator. Measured in LM Studio on the real search blob:
  **HTML-dump rate 15% → 2%** at temp 0.8, other axes held.

## How to use it
- **Tool-calling:** serve with a current llama.cpp (`llama-server --jinja`) or a recent
  LM Studio runtime; older bundled runtimes mis-render the tool token. And **set context ≥ 9K**
  (see the warning above).
- Coding / web / chat work in any runtime.
- Apache-2.0 throughout (base, teacher = Qwen2.5-32B-Instruct, self-generated data).

## Honest caveats
- **Context length is not optional with MCP tools** — see the warning above.
- It's a universal 7B, not a pure-C specialist: the old coder base scores ~1 noise-item
  higher on the hard C subset. For coding **+ tools + chat + web** in one model, this is the pick.
- **Small-model ceiling on long multi-turn artifacts:** deep in a tool conversation, asking it
  to *expand and merge* a lot of detail into one large HTML document, it tends to produce a
  compact doc and can repeat rather than fully incorporate everything. Single-shot "make a
  detailed page" requests are fine; iterative "put ALL of that inside the doc" is where the 7B
  shows its size.
- Web score is structural validity, not visual-design taste.
