"""Universal-model config. Extends the coding pipeline into a general model that
KEEPS coding, DE-EMPHASIZES Python, and ADDS web design + general chat + tool/MCP
calling. Imported by the gen_*.py generators, mix.py, and eval_universal.py.

The coding pipeline's config.py is left intact (still describes the C/Py teacher
run). This file is the single lever for the universal run.
"""

# --- Teacher: Qwen2.5-32B-Instruct, served by llama.cpp on :8091 --------------
# Dense 32B Q4_K_M (~20GB): does NOT fully fit 16GB, so serve_teacher_32b.sh
# offloads as many layers as fit and spills the rest to RAM. Slower than the
# 14B coder, but a far stronger GENERAL/chat/tool teacher — and coding is
# preserved by REPLAY of the already-verified data, so the teacher's job here
# is chat + web + tools, where 32B-general shines.
TEACHER_BASE_URL = "http://localhost:8091/v1"
TEACHER_API_KEY  = "sk-noauth"
TEACHER_MODEL    = "qwen2.5-32b-instruct"   # llama-server ignores this; any string
GEN_TEMPERATURE  = 0.9

# --- Student: CONTINUE from the current coding model, not stock ---------------
# Starting from the merged v3 coder means we add skills on top of earned coding
# ability instead of re-teaching it. Replay (below) guards against forgetting.
STUDENT_MODEL = "qwen-coder-7b-mine-v3"     # local merged 16-bit dir
MAX_SEQ_LEN   = 4096

# --- Output artifacts ---------------------------------------------------------
WEB_RAW   = "web_raw.jsonl"      # gen_web.py  (HTML-structure verified)
CHAT_RAW  = "chat_raw.jsonl"     # gen_chat.py (quality-gated)
TOOLS_RAW = "tools_raw.jsonl"    # gen_tools.py (JSON-schema verified)
MULTI_RAW = __import__("os").environ.get("MULTI_RAW_FILE", "multi_raw.jsonl")  # gen_multi.py; v12 points at multi_aug.jsonl (big noisy blobs + freshresearch)
IMAGE_GOLD = "image_gold.jsonl"  # gold_images.py (safe placeholder-image behavior)
UNI_CLEAN   = "clean_universal.jsonl"    # mix.py -> training set
UNI_HELDOUT = "heldout_universal.jsonl"  # mix.py -> held-out slice
MERGED_DIR  = "qwen-uni-7b"
GGUF_DIR    = "gguf"
HELDOUT_FRAC = 0.03

# --- Replay: how much of the EXISTING verified coding data to fold back in -----
# Fractions of clean_v3.jsonl kept per language. C is the execution-verified
# anchor (keep all). Python is downsampled hard (user: "instead of Python, HTML").
REPLAY_SRC = "clean_v3.jsonl"
REPLAY_KEEP = {
    "C": 1.0,        # verified coding anchor — keep every sample
    "HTML": 1.0,     # web focus — keep all the old ones too
    "Java": 1.0,     # small; general breadth
    "Python": 0.25,  # downsample: keep a little so we don't lose it
}

# --- New-data targets (per category; generators stop at target) ---------------
# Sized for the measured ~13 tok/s aggregate (3 generators concurrent) to fit
# generation into the budget with room for a targeted iteration. tools gets the
# most (biggest measured gap in the baseline constant test).
TARGET_WEB   = 400   # v8: REGENERATED fresh (multi-line-formatted; old minified set retired)
TARGET_CHAT  = 550
TARGET_TOOLS = 550
# v8: the multi-turn mode-switch axis — the actual gap the v6 field-test exposed
# (once the model produced an artifact it could not re-call a tool, answer a
# plain follow-up in prose, or revise). Time-bounded on the slow 32B teacher.
TARGET_MULTI = 350

# v8 multi-turn scenario types. Each teaches a distinct mode-SWITCH that v6
# lacked; WE assemble the canonical message list and verify it structurally.
#   subtask  : call a tool as a STEP toward a larger deliverable, then deliver
#   recall   : answer, then a follow-up needs ANOTHER tool call
#   modeswitch: produce an artifact, then answer a PLAIN question in prose
#   revise   : produce an artifact, then revise it on request
#   enrich   : an artifact ALREADY EXISTS, then the user says "search for real
#              info and update it" -> the model must emit a <tool_call> (NOT keep
#              editing blind, NOT embed JS). This is the v8 field-test failure
#              (artifact-first, THEN a tool call). site:-scoping taught here too.
#   research : an artifact exists, then "search and just SUMMARIZE (no code)" ->
#              tool_call, then a PROSE answer. Covers the v8 failure where it made
#              a MARKDOWN doc with fabricated data instead of calling the tool.
MULTI_TYPES = ["subtask", "recall", "modeswitch", "revise", "enrich", "research"]

# --- Web-design grid: domains x task x difficulty -----------------------------
WEB_DOMAINS = [
    "semantic page structure", "forms & input validation",
    "tables & data display", "accessibility (a11y)",
    "responsive CSS layout (flexbox/grid)", "CSS styling & theming",
    "vanilla-JS interactivity", "reusable components / cards",
    "navigation & menus", "landing / hero sections",
]
WEB_TASKS = [
    "build a complete, self-contained HTML page for",   # -> full doc (doctype)
    "build a single reusable HTML+CSS component for",    # -> fragment
    "find and fix the markup/CSS bug in",
    "make this markup accessible and semantic:",
    "add responsive CSS (mobile-first) to",
    "add vanilla-JS interactivity to",
]
WEB_DIFFICULTIES = ["beginner", "intermediate", "hard"]

# --- General-chat grid: topic x style ----------------------------------------
CHAT_TOPICS = [
    "everyday science & how-things-work", "history & culture",
    "practical how-to & life advice", "writing help & editing",
    "brainstorming & ideation", "explaining a concept simply",
    "step-by-step reasoning / logic puzzle", "math word problem",
    "summarizing & rephrasing text", "comparing options / pros-cons",
    "cooking & food", "health & fitness (general, non-medical)",
    "travel & geography", "technology & gadgets (non-code)",
    "career & productivity", "personal finance basics",
    "language & grammar", "nature & environment",
]
CHAT_STYLES = [
    "a concise single-turn Q&A",
    "a 2-3 turn follow-up conversation",
    "a request for a structured answer (list/steps/table)",
    "a request that needs a clarifying question first, then an answer",
]

# --- Tool / MCP scenario grid -------------------------------------------------
# Each scenario asks the teacher to pick the right tool + fill arguments; WE
# assemble the canonical Qwen tool-call turn and schema-verify it (gen_tools.py).
TOOL_DIFFICULTIES = ["single obvious tool", "pick among several tools",
                     "multi-step (call, read result, answer)",
                     "user query needs no tool (must answer directly)"]

# Upweight rows by kind in the TRAIN split (repetition helps a weak skill).
# TOOL_UPWEIGHT drove <tool_call> format learning; CODE_UPWEIGHT (v7) boosts the
# replayed coding data to recover C on the general base.
TOOL_UPWEIGHT = int(__import__("os").environ.get("TOOL_UPWEIGHT", "1"))
CODE_UPWEIGHT = int(__import__("os").environ.get("CODE_UPWEIGHT", "1"))
# v9: the hand-authored placeholder-image gold set is tiny (~7 rows); repeat it
# so the "use a live placeholder host / inline SVG, don't web-search for images"
# behavior actually registers (same rationale as gold_c's header examples).
IMG_UPWEIGHT = int(__import__("os").environ.get("IMG_UPWEIGHT", "8"))
# v9: the enrich/research (tool-call-AFTER-an-artifact) shape fights an entrenched
# mode-lock prior, so repeat those rows to give them weight against it.
TOOLLOCK_UPWEIGHT = int(__import__("os").environ.get("TOOLLOCK_UPWEIGHT", "3"))
# v11: research REMOVED from the shared toollock set and given its OWN, larger
# upweight below. Rationale from the v10 field-test: enrich (search -> revise the
# HTML doc) and research (search -> answer in PROSE) were both weighted x3, so in
# the "after a search" regime the model saw as much 'emit fenced HTML' as 'emit
# prose' — and with 400 fenced web docs also pulling toward markup, it defaulted
# to wrapping even a plain answer in ```html when told "no html, just search".
# Tilt the after-search balance decisively toward prose.
TOOLLOCK_TYPES = {"enrich"}
# v11: research (tool_call -> PLAIN-PROSE answer) upweight. Higher than enrich's
# x3 so prose-after-search out-masses fence-after-search. Covers BOTH the
# tool-call-after-artifact signal AND the "answer in prose, don't fence" signal.
# v12: also applies to `freshresearch` (plain question -> search -> prose, NO
# artifact) — the actual field failure was a FRESH chat.
RESEARCH_UPWEIGHT = int(__import__("os").environ.get("RESEARCH_UPWEIGHT", "5"))
RESEARCH_TYPES = {"research", "freshresearch"}

PASSES = 2          # sweeps of a grid (generators also stop at TARGET_*)
EXEC_TIMEOUT = 10
