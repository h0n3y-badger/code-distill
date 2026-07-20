"""Decisive test: is the student's C logic actually correct, with missing #includes
the only mechanical gap? Get the student's normal output, then PREPEND a kitchen-sink
header block and recompile. Duplicate includes are harmless (header guards), so this
only rescues code whose sole defect was a missing header. Contract/naming and logic
bugs will still fail. Student on :8091."""
import json
from collections import Counter
from openai import OpenAI
import config as C
from diag_c import extract_code, compile_run, classify_compile, HAS_MAIN

client = OpenAI(base_url=C.TEACHER_BASE_URL, api_key=C.TEACHER_API_KEY)
HINT = (" Provide ONLY the implementation (functions/types and any needed "
        "#includes); do NOT write a main() function.")

PREAMBLE = "\n".join("#include <%s>" % h for h in [
    "stdio.h", "stdlib.h", "stddef.h", "stdbool.h", "string.h",
    "stdint.h", "limits.h", "math.h", "ctype.h", "assert.h"]) + "\n"

def solve(instruction):
    r = client.chat.completions.create(model="student", temperature=0.2,
        messages=[{"role": "user", "content": instruction + HINT}])
    return r.choices[0].message.content

def main():
    samples = []
    for line in open(C.EVAL_FILE):
        rec = json.loads(line)
        if rec.get("lang") == "C" and rec.get("tests"):
            instr = next(m["content"] for m in rec["messages"] if m["role"] == "user")
            samples.append((instr, rec["tests"]))

    buckets = Counter()
    n_pass = 0
    for i, (instr, tests) in enumerate(samples, 1):
        code = PREAMBLE + extract_code(solve(instr))    # <-- auto-inject headers
        wrote_main = bool(HAS_MAIN.search(code))
        status, detail = compile_run(code, tests)
        if status == "PASS":
            n_pass += 1; bucket = "PASS"
        elif status == "COMPILE_FAIL":
            bucket = classify_compile(detail, wrote_main)
        else:
            bucket = status
        buckets[bucket] += 1
        print(f"[{i:>2}/{len(samples)}] {bucket:<38}", flush=True)

    n = len(samples)
    print(f"\n=== C with AUTO-INJECTED headers ({n} samples) ===")
    print(f"PASS {n_pass}/{n} = {n_pass/n:.1%}   (baseline 41/66 = 62.1%)")
    print("\nremaining failures (these are NOT include problems):")
    for b, c in buckets.most_common():
        if b == "PASS":
            continue
        print(f"  {c:>3}  {b}")

if __name__ == "__main__":
    main()
