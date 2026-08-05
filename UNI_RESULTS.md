# Universal constant-test results

| ver | C exec | web valid | tools | chat kw | date |
|---|---|---|---|---|---|
| base-v3 | 5/12 (42%) | 8/8 (100%) | 2/8 (25%) | 100% | 2026-07-21 |
| uni-1 | 4/12 (33%) | 7/8 (88%) | 2/8 (25%) | 100% | 2026-07-21 |
| uni-2 | 5/12 (42%) | 7/8 (88%) | 2/8 (25%) | 75% | 2026-07-21 |
| uni-3 | 4/12 (33%) | 7/8 (88%) | 2/8 (25%) | 75% | 2026-07-21 |
| uni-4 | 5/12 (42%) | 8/8 (100%) | 2/8 (25%) | 100% | 2026-07-21 |
| uni-5 | 4/12 (33%) | 7/8 (88%) | 2/8 (25%) | 100% | 2026-07-21 |
| uni-5fix | 4/12 (33%) | 8/8 (100%) | 2/8 (25%) | 75% | 2026-07-21 |
| uni-6 | 4/12 (33%) | 7/8 (88%) | 8/8 (100%) | 100% | 2026-07-21 |
| uni-7 | 3/12 (25%) | 7/8 (88%) | 8/8 (100%) | 100% | 2026-07-22 |

**LEADING CANDIDATE: v12** (qwen-uni-7b-v12-Q5_K_M.gguf) — pending user field re-test in LM Studio; NOT published.

v12 (2026-07-24): v11 still FAILED in the field — with a LARGE noisy searxng blob it
intermittently emitted an HTML/markdown DOCUMENT (<details>/<div>) instead of prose,
**even in a FRESH chat with no artifact**. Two-part methodology failure on my side:
(1) I validated in the :8091 llama.cpp harness, not the user's LM Studio runtime;
(2) my eval used SMALL CLEAN blobs at temp 0.2-0.4. Measured FAITHFULLY in LM Studio
(:1234, real 11KB field blob): v11 = **15% HTML @temp0.8, 10% @temp0.4** (temp is only
a minor lever — it's a real weakness). Root cause: every research/recall/enrich/
subtask row trained on a small clean result; reality is a big messy blob, so the base
model's HTML-doc prior (<details> is in 0/2940 rows) wins occasionally. Fix (teacher-
FREE, augment_tools.py): (a) INFLATE every training tool-result to a ~18-hit searxng
blob — INCLUDING enrich/subtask which still emit DOCS, so the discriminator stays
USER INTENT not blob size; (b) DERIVE `freshresearch` (plain Q -> search -> PROSE, no
artifact). Result, measured in LM Studio: **15%->2% @temp0.8, 10%->0% @temp0.4**, every
frozen axis HELD (C42/web100/tools100/chat100, artifact_search 3/3). Validation now
runs on :1234 (validate_bigblob.py) — the deployment runtime, not the harness that hid
this twice. Staged + loaded in LM Studio for the user's re-test.

The arc: v8 fixed v6's *presentation* failures (bare/minified code, no mode-switch)
and scored clean on the constant test — but a second LM Studio field-test found a
DEEPER failure the eval still wasn't measuring: **once an assistant artifact is in
the chat history, the model refuses to emit a tool call and FABRICATES instead**
(told to search spacex.com it re-dumped HTML / wrote a markdown doc with made-up
data / embedded fake JS). An isolation probe (tool passed explicitly, as LM Studio
does) confirmed this is WEIGHTS, not client: v8 fires tools in a fresh chat but
never after an artifact.

Fix (v9): two new multi-turn types — `enrich` (artifact → search → revise) and
`research` (artifact → search → prose answer) — putting a <tool_call> AFTER an
artifact, + a new `artifact_search` eval axis + placeholder-image gold. v9 fixed
the mode-lock (verified live on a full-size artifact) BUT regressed coding
(C 50→33%, web 100→88%) because a legacy TOOL_UPWEIGHT=3 bloated the mix
tool-heavy. v10: same data, TOOL_UPWEIGHT→1 (tools is saturated, no longer the
weak skill) so code/web regain their share. Result — the fix HELD and the
regression recovered: **artifact_search 3/3 + live big-artifact probe passes,
web 100%, tools 100%, chat 100%, C 42%** (noise-band — C bounced 33-50% across all
versions; v7 confirmed that ceiling; v8's 50% was the high end of noise, not lost
capability). Also fenced 8/8, modeswitch 3/3, tool-subtask 3/3.

v11 (2026-07-24): the v10 LM Studio field-test had ONE residual — when an artifact
was in history and the user said "no html, just give me the <profile/rundown>,
search for it", v10 INTERMITTENTLY wrapped the plain answer in a ```html/```markdown
doc instead of prose. New eval axis `research_prose` (multi-SAMPLE, temp 0.4, with a
MESSY searxng-style result blob + "profile/rundown" framing — the clean 1-shot
version passed 3/3 and hid the bug) measured it: **v10 = 12/18 (67%), and the SpaceX
"launch profile" case failed 0/6** (always fenced). Fix: `research` (search→PROSE)
and `enrich` (search→revise-doc) were both weighted x3, so after a search the model
saw as much 'emit markup' as 'emit prose'; gave research its OWN x5 upweight (enrich
untouched — it WORKED in the field Florida turn) + topped up ~60 research rows with
terse "no html" phrasing. Result: **research_prose 17/18 (94%), SpaceX case 0/6→5/6**,
with EVERY other axis held (C 42%, web 100% fenced 8/8, tools 100%, chat 100%,
modeswitch/subtask/artifact_search 3/3). Effective mix: research 80→355 rows vs
enrich 156 — prose-after-search now out-masses fence-after-search. Staged in LM
Studio next to v6/v8/v10; NOT published.
| uni-8 | 6/12 (50%) | 8/8 (100%) | 8/8 (100%) | 100% | 2026-07-23 |
| uni-9 | 4/12 (33%) | 7/8 (88%) | 8/8 (100%) | 100% | 2026-07-24 |
| uni-10 | 5/12 (42%) | 8/8 (100%) | 8/8 (100%) | 100% | 2026-07-24 |
| uni-11 | 5/12 (42%) | 8/8 (100%) | 8/8 (100%) | 100% | 2026-07-24 |
| uni-12 | 5/12 (42%) | 8/8 (100%) | 8/8 (100%) | 100% | 2026-07-24 |
