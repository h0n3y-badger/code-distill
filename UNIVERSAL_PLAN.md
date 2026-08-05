# Universal-model pivot — working plan & progress

Pivot `code-distill` from a pure Python+C coding distillation into a **universal**
small model: **retain coding**, **de-emphasize Python**, **add web design (HTML/CSS/JS),
general chat, and tool/MCP function-calling**. Continue from the current coding
student (`qwen-coder-7b-mine-v3`) + a **replay slice** of existing verified coding
data so we add skills without catastrophic-forgetting.

Started 2026-07-21. Budget: ~72h wall-clock, single RTX 5070 Ti (16GB).

## Decisions (confirmed with user 2026-07-21)
- Teacher = **Qwen2.5-32B-Instruct Q4_K_M** (dense, partial GPU offload). Downloading to `teacher/`.
- Python = **downsample hard, keep a little**. C = execution-verified coding anchor. HTML/web = primary coding focus.
- Base student = **`qwen-coder-7b-mine-v3`** (already merged 16-bit dir), continue-train QLoRA.
- Tool/MCP = Hermes/Qwen `<tool_call>{json}</tool_call>` format, schema-verified.

## Data mix (target for v-uni-1)
- Replay coding (existing verified): C (keep all), Python (downsample ~4x), HTML (keep).
- NEW web design: `gen_web.py` — HTML/CSS/JS tasks, HTML-structure verified.
- NEW general chat: `gen_chat.py` — diverse multi-turn instruction/chat.
- NEW tools/MCP: `gen_tools.py` — function-calling convos, JSON-schema verified.

## New/changed code
- `datalib.py`  +HTML validation (`html_issues`/`is_valid_html`), +tool-call validation (`validate_tool_call`).
- `config_uni.py` universal config (mix weights, teacher, task grids).
- `gen_web.py`, `gen_chat.py`, `gen_tools.py` — teacher generators (verified where possible).
- `mix.py` — build `clean_universal.jsonl` (replay + new, weighted, dedup, held-out split).
- `train.py` — already env-parametrized (BASE via STUDENT_MODEL override, TRAIN_DATA, OUT_DIR).
- `eval_universal.py` — THE CONSTANT TEST: frozen golden suite, 4 axes (C-exec, web-valid, tool-schema, chat-capture), versioned results.
- `tests/` — extend regression suite for new datalib fns + mix weighting.

## The constant test
`eval_universal.py` runs an identical frozen suite every version → comparable rows in `UNI_RESULTS.md`.
Axes: C execution pass@1 (regression guard) · HTML structural validity · tool-call schema correctness · chat transcript capture (for manual quiz). I (Claude) additionally quiz each version by hand.

## Baseline (constant test on the starting v3 coder, 2026-07-21)
| ver | C exec | web valid | tools | chat kw |
|---|---|---|---|---|
| base-v3 | 5/12 (42%) | 8/8 (100%) | 2/8 (25%) | 100% |
Key finding: baseline already PICKS the right tool with right args but emits the
WRONG format (`<json>`/`<function_call>` not `<tool_call>`), so it scores 25%.
Highly learnable — canonical-format tool data should lift this sharply.

## Phases / status
- [x] P0 setup + teacher download (Qwen2.5-32B-Instruct Q4, ngl=42, ~6.7 tok/s single / ~13 concurrent)
- [x] P1 build code + regression tests green (49 tests pass)
- [~] P2 generate data (32B teacher) — RUNNING: run_uni_gen.sh (task bj07qxkj1), watchdog bbm8w6ysg
- [ ] P3 mix + train v-uni-1 + quantize
- [ ] P4 constant test + manual quiz v-uni-1
- [ ] P5 iterate (v-uni-2/3) on weaknesses
- [ ] P6 model card + ping user

## Completion alert (user: LOUD ring on v-done AND every step/milestone, even at 3am)
- KDE Connect Pixel 8a device id: `eb32d797b3f0467da06e042599fed67a`
- Use `--ring` (find-my-phone: loud, bypasses DND) + `--ping-msg "<msg>"`. Helper: `./ring.sh "<msg>"`.
- Ring on: each iteration done/fail, and major milestones (train done, quantize done, eval done, "exceeds base"). Pipeline scripts should ring at each major step, not just the end.

## v-uni-1 result + diagnosis (2026-07-21)
Training succeeded (loss 0.625->0.240, 2 epochs, 260 steps, adapter out_uni_adapter, merged qwen-uni-7b, gguf/qwen-uni-7b-Q5_K_M.gguf).
Constant test on OLD LM Studio runtime (2.25.2): C 4/12, web 7/8, tools 2/8, chat 100%. Looked like a regression.
DIAGNOSIS: NOT a model failure. v-uni-1 emits PERFECT tool JSON (right tool + right args every time). The bug is the OLD bundled llama.cpp (2.25.2) mis-rendering the USER_DEFINED token 151657 `<tool_call>` as `+#+#+#+`, so llama-server's --jinja parser never recognizes the call -> false 2/8.
- GGUF vocab is correct (151657->'<tool_call>', type=4 USER_DEFINED). HF tokenizers all round-trip fine. The "Mistral regex" warning is spurious.
- FIX: serve/eval with CURRENT llama.cpp (system has no nvcc -> CPU build into llama.cpp/build; serve_uni_cur.sh). Re-eval BOTH base-v3 and uni-1 on the correct runtime for a fair apples-to-apples comparison.
- Note: user's real LM Studio app likely ships a newer runtime that renders it fine; the 2.25.2 pin in serve_teacher.sh is the culprit here.
- Small-N caveat: constant test C=12/web=8 items -> ±1-2 is noise. Consider larger C sample for power (keep frozen items as a subset to preserve comparability).

## Iteration goal (user): keep improving until v-uni EXCEEDS base on the constant test; regression-test on code change / score drop; ping on every step + iteration.

## BREAKTHROUGH v6 (2026-07-21 ~23:25): tools SOLVED via general-base pivot
- v1-v5fix (coder base): tools stuck 2/8 — coder base ships <tool_call>(151657) embedding DEAD/zero; even injecting a healthy one into the merged model didn't overcome the coder's hidden-state prior. Also found merge_adapter drops a frozen-but-transplanted embedding (only LoRA'd/saved tensors survive merge).
- v6: trained from GENERAL Qwen2.5-7B-Instruct base (native working tool tokens) + replayed coding data. RESULT: **tools 8/8 (100%, verified real: canonical <tool_call> parsed into tool_calls), chat 100%, web 7/8, C 4/12.** vs base-v3 (C42 web100 tools25 chat100): tools +75pts, chat/web ~equal (noise), C -1 item (within the 4-5/12 band ALL 7 versions show = documented small-model C ceiling).
- v7: confirm-ceiling / strict-dominance attempt (general base, TOOL_UPWEIGHT=2 + CODE_UPWEIGHT=2). If C recovers to >=5 great; if not, confirms ceiling -> v6/v7 is the WIN.
- WIN model artifact: qwen-uni-7b-v6 (gguf/qwen-uni-7b-v6-Q5_K_M.gguf) [or v7 if better]. Universal: coding+web+chat+tools/MCP.

## ROOT CAUSE of the tools failure (found after v1/v2/v3) — 2026-07-21
- Constant-test tools stuck at 2/8 across v1 (r16), v2 (3x upweight/r32/3ep), v3 (embed-trained from zero).
- CAUSE: our Coder base (qwen-coder-7b-mine-v3) AND stock unsloth/qwen2.5-coder-7b-instruct ship the tool tokens
  <tool_call>(151657)/</tool_call>(151658) with **embedding norm = 0.0** (dead). Qwen ties embeddings→output head,
  so a zero row = zero logit = the token can NEVER be emitted. Model spilled to garbage (plex/uss/proprié) in that slot.
- v3 embed-training-from-zero reached norm 0.057 but wrong DIRECTION → still garbage.
- General Qwen2.5-7B-Instruct has them ALIVE (151657 norm 0.0172) and emits tool calls fine.
- FIX (v4): TRANSPLANT the two tool-token embedding rows from general-7b-instruct into the coder base, THEN train
  (embed refine on, from the good init). `train_uni.py TRANSPLANT_EMBED=<general snapshot dir>`. run_v4_pipeline.sh.
- Iteration results table is in UNI_RESULTS.md. base-v3: C42% web100% tools25% chat100%(kw).
- Rings are TIME-GATED after 21:00 in run_v4_pipeline.sh (user watching show till 21:00).
- Eval must use CURRENT llama.cpp (llama.cpp/build/bin, CPU) — LM Studio 2.25.2 garbles/parses tool tokens wrong.
  No nvcc on system → can't build CUDA llama.cpp; CPU eval ~ok (tool outputs short).

## Resume notes (if context resets)
- Generation resumable: generators append to web_raw/chat_raw/tools_raw.jsonl, stop at target. Just re-run `bash run_uni_gen.sh`.
- After gen: `python mix.py` -> clean_universal.jsonl; STOP teacher; `TRAIN_DATA=clean_universal.jsonl OUT_DIR=qwen-uni-7b python train_uni.py` (BASE=qwen-coder-7b-mine-v3 default).
- Quantize: adapt quantize.sh (MERGED=qwen-uni-7b -> gguf/qwen-uni-7b-Q5_K_M.gguf).
- Eval: `bash serve_uni.sh` then `VER=uni-1 STUDENT_ID=qwen-uni-7b STAMP=<date> python eval_universal.py`.
