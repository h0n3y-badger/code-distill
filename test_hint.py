"""Confirmation experiment for Fix #1: re-run the C eval with a stronger,
include-demanding hint. Identical to diag_c.py in every other way (same samples,
same compile/run gate, same buckets) so the pass@1 is directly comparable to the
62.1% baseline. Student must be served on :8091."""
import re, json
from collections import Counter
from openai import OpenAI
import config as C, gen
from diag_c import extract_code, compile_run, classify_compile, HAS_MAIN

client = OpenAI(base_url=C.TEACHER_BASE_URL, api_key=C.TEACHER_API_KEY)

# The ONLY change vs baseline: spell out that the code must be self-contained
# with all its own #includes, because the test file includes only <assert.h>.
STRONG_HINT = (
    " Provide ONLY the implementation (functions/types and any needed declarations); "
    "do NOT write a main() function. Your code must be SELF-CONTAINED: emit ALL "
    "necessary #include directives yourself — e.g. <stdlib.h> (malloc/free), "
    "<stddef.h> (NULL), <stdbool.h> (bool/true/false), <string.h> (str*), "
    "<stdio.h> (I/O). The test harness includes only <assert.h>, so anything else "
    "your code relies on must be included by you.")

def solve(instruction):
    r = client.chat.completions.create(model="student", temperature=0.2,
        messages=[{"role": "user", "content": instruction + STRONG_HINT}])
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
        code = extract_code(solve(instr))
        wrote_main = bool(HAS_MAIN.search(code))
        status, detail = compile_run(code, tests)
        if status == "PASS":
            n_pass += 1; bucket = "PASS"
        elif status == "COMPILE_FAIL":
            bucket = classify_compile(detail, wrote_main)
        else:
            bucket = status
        buckets[bucket] += 1
        print(f"[{i:>2}/{len(samples)}] {bucket:<38} main={'Y' if wrote_main else 'n'}",
              flush=True)

    n = len(samples)
    print(f"\n=== C RE-EVAL with strong include hint ({n} samples) ===")
    print(f"PASS {n_pass}/{n} = {n_pass/n:.1%}   (baseline was 41/66 = 62.1%)")
    print("\nremaining failure buckets:")
    for b, c in buckets.most_common():
        if b == "PASS":
            continue
        print(f"  {c:>3}  {b}")

if __name__ == "__main__":
    main()
