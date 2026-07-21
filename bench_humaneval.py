"""Independent generalization check: HumanEval (openai/human-eval, 164 Python
problems) — a benchmark this project DELIBERATELY never trained on or generated,
so it's genuinely unseen. Run the SAME protocol on the distilled student and the
stock base; the point isn't the absolute number (different format/distribution
than our eval) but whether the student's lead SURVIVES on neutral problems. If
the gap vanishes here, our home eval was benchmaxxing; if it holds, the
distillation generalizes.

Serve a model on :8091, then:  MODEL_TAG=v3 .venv/bin/python bench_humaneval.py
Writes per-model pass@1 to bench/humaneval_<tag>.json.
"""
import os, re, json, subprocess, tempfile, sys
from openai import OpenAI
import config as C

TAG = os.environ.get("MODEL_TAG", "model")
N = int(os.environ.get("HE_N", "164"))
TIMEOUT = 15
client = OpenAI(base_url=C.TEACHER_BASE_URL, api_key=C.TEACHER_API_KEY)

PREAMBLE = ("from typing import List, Dict, Tuple, Optional, Any\n"
            "import math, re, collections, itertools, functools, heapq, bisect\n")


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def solve(prompt):
    msg = ("Complete the following Python function. Return ONLY the complete "
           "function implementation in a single ```python code block — include "
           "the signature, no explanation.\n\n" + prompt)
    r = client.chat.completions.create(model=TAG, temperature=0.2,
        messages=[{"role": "user", "content": msg}], max_tokens=1024)
    return r.choices[0].message.content


def run_program(code, test, entry_point, prompt):
    # If the model reproduced the def, use its code as-is; else treat as a body
    # completion appended to the prompt signature.
    if re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code):
        body = code
    else:
        body = prompt + "\n" + code
    program = f"{PREAMBLE}\n{body}\n\n{test}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=TIMEOUT)
        return p.returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except OSError: pass


def main():
    rows = [json.loads(l) for l in open("bench/HumanEval.jsonl")][:N]
    npass = 0
    fails = []
    for i, r in enumerate(rows, 1):
        code = extract_code(solve(r["prompt"]))
        ok = run_program(code, r["test"], r["entry_point"], r["prompt"])
        npass += ok
        if not ok:
            fails.append(r["task_id"])
        print(f"[{i:>3}/{len(rows)}] {r['task_id']:<14} "
              f"{'PASS' if ok else 'FAIL'}  running={npass/i:.1%}", flush=True)
    res = {"tag": TAG, "n": len(rows), "pass": npass,
           "pass_at_1": npass / len(rows), "fails": fails}
    os.makedirs("bench", exist_ok=True)
    with open(f"bench/humaneval_{TAG}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== HumanEval [{TAG}] pass@1 = {npass}/{len(rows)} = "
          f"{npass/len(rows):.1%} ===")


if __name__ == "__main__":
    main()
