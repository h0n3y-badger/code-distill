"""Central config for the distillation run. Edit the seed grid to match YOUR stack —
that grid is the single biggest lever on what the student ends up good at."""

# --- Teacher endpoint (LM Studio, OpenAI-compatible) -------------------------
# Teacher served by llama.cpp's own llama-server (see serve_teacher.sh), NOT LM
# Studio — LM Studio's load orchestration SIGSEGV'd on this GGUF while the bare
# runtime loads it fine. Port 8091 (8080 is taken by SearXNG, 1234 by LM Studio).
TEACHER_BASE_URL = "http://localhost:8091/v1"
TEACHER_API_KEY  = "sk-noauth"                        # llama-server ignores the value
# DENSE 14B teacher: fully fits 16GB VRAM (~11GB resident), no expert offload,
# consistent speed on the diverse sweep. (The 35B-A3B MoE crawled: offloaded cold
# experts got fetched from RAM per token, AND it was a reasoning model emitting
# long <think> blocks. Dense 14B sidesteps both.)
TEACHER_MODEL    = "qwen2.5-coder-14b"               # llama-server ignores this; any string works
GEN_TEMPERATURE  = 0.9                                # high = more diverse tasks

# --- Student (Phase 3) -------------------------------------------------------
STUDENT_MODEL    = "unsloth/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LEN      = 4096

# --- Output artifacts --------------------------------------------------------
RAW_FILE     = "raw.jsonl"       # gen.py appends here (ctrl-C safe)
CLEAN_FILE   = "clean.jsonl"     # dedup.py -> training set
HELDOUT_FILE = "heldout.jsonl"   # dedup.py -> small held-out slice (never trained on)
EVAL_FILE    = "eval_set.jsonl"  # gen_eval.py -> dedicated execution-verified eval set
MERGED_DIR   = "qwen-coder-7b-mine"
GGUF_DIR     = "gguf"
HELDOUT_FRAC = 0.03              # fraction reserved for eval

# --- Seed grid: PER-LANGUAGE domains x task x difficulty ---------------------
# Each language crosses ONLY its own relevant domains -> no nonsense combos like
# "memory management in HTML". Domain COUNT per language sets its share of the
# dataset, so this also weights the mix: Python/HTML/C rich, Java a minority.
# (Python is also the only one execution-verified; see TASKS + gen.py.)
LANG_DOMAINS = {
    "Python": [                                   # primary
        "data structures", "algorithms", "CLI tools", "file & stream I/O",
        "text parsing", "web servers / REST", "concurrency & async",
        "regex", "SQLite / storage", "error handling",
    ],
    "C": [                                         # primary
        "data structures", "algorithms", "memory management & pointers",
        "dynamic arrays / linked lists", "file & stream I/O",
        "string manipulation", "bit manipulation", "parsing", "error handling",
    ],
    "HTML": [                                      # primary (HTML + CSS + a little JS)
        "semantic page structure", "forms & input validation",
        "tables & data display", "accessibility (a11y)",
        "CSS layout (flexbox/grid)", "embedded JS interactivity",
        "templating / components",
    ],
    "Java": [                                      # "some" -> fewer domains
        "collections & generics", "OOP design patterns",
        "concurrency & threads", "file & stream I/O", "exception handling",
    ],
}

# task -> whether generated Python can be auto-verified by execution
TASKS = {
    "implement a self-contained function":        True,
    "find and fix the bug in this snippet":       True,
    "refactor this code for readability":         False,
    "write pytest-style unit tests for":          False,
    "complete the fill-in-the-middle gap in":     False,
    "explain what this code does, then improve it": False,
}

DIFFICULTIES = ["beginner", "intermediate", "hard"]

PASSES = 2          # full sweeps of the grid; bump for more data
EXEC_TIMEOUT = 10   # seconds per rejection-sampling run
