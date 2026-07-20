"""Phase 5 — evaluate the trained student on the dedicated eval set (eval_set.jsonl).

Serve your quantized student on the configured endpoint (serve_student.sh or
LM Studio), then run this. Execution-based pass@1 on Python (run) and C (gcc
compile+run) — contamination-proof, on YOUR data. Never HumanEval (Monolith-1 rule).
"""
import os, re, json
from openai import OpenAI
import config as C, gen

STUDENT_ID = os.environ.get("STUDENT_ID", "qwen-coder-7b-mine")
client = OpenAI(base_url=C.TEACHER_BASE_URL, api_key=C.TEACHER_API_KEY)

# For C, mirror the training convention so the student's code links with the test main().
C_HINT = (" Provide ONLY the implementation (functions/types and any needed "
          "#includes); do NOT write a main() function.")

def solve(instruction, lang):
    prompt = instruction + (C_HINT if lang == "C" else "")
    r = client.chat.completions.create(model=STUDENT_ID, temperature=0.2,
        messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content

def extract_code(text):
    """Pull the first fenced code block from the student's reply, else use it raw."""
    m = re.search(r"```[a-zA-Z0-9+#]*\n(.*?)```", text, re.S)
    return m.group(1) if m else text

def main():
    stats = {}                      # lang -> [passed, total]
    samples = []
    with open(C.EVAL_FILE) as f:
        for line in f:
            rec = json.loads(line)
            lang, tests = rec.get("lang"), rec.get("tests")
            if lang not in ("Python", "C") or not tests:
                continue
            instr = next(m["content"] for m in rec["messages"] if m["role"] == "user")
            code = extract_code(solve(instr, lang))
            ok = gen.verify(lang, code, tests)          # Python: run; C: gcc compile+run
            s = stats.setdefault(lang, [0, 0]); s[0] += ok; s[1] += 1
            tot = sum(v[1] for v in stats.values()); pas = sum(v[0] for v in stats.values())
            print(f"[{tot:>3}] {lang:6} {'PASS' if ok else 'FAIL'}  "
                  f"running pass@1={pas/tot:.1%}", flush=True)
            if len(samples) < 4:
                samples.append((lang, instr, ok))

    if not stats:
        print("No scorable samples in eval set. Run gen_eval.py first.")
        return
    print("\n=== pass@1 by language ===")
    for lang, (p, n) in stats.items():
        print(f"  {lang:6} {p}/{n} = {p/n:.1%}")
    tot = sum(v[1] for v in stats.values()); pas = sum(v[0] for v in stats.values())
    print(f"  {'TOTAL':6} {pas}/{tot} = {pas/tot:.1%}")

if __name__ == "__main__":
    main()
