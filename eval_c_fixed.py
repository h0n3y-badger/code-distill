"""Confirm the auto-#include post-processor lift on the v2 student. For each C eval
sample: get the student's solution, run fix_c_includes on it ALONE (no test peeking
— the honest inference case), then score with the same gcc compile+run gate. Compare
to the v2 C baseline (42/66 = 63.6%). Student must be served on :8091."""
import json
from openai import OpenAI
import config as C
from diag_c import extract_code, compile_run
from c_postprocess import fix_c_includes

client = OpenAI(base_url=C.TEACHER_BASE_URL, api_key=C.TEACHER_API_KEY)
HINT = (" Provide ONLY the implementation (functions/types and any needed "
        "#includes); do NOT write a main() function.")

def solve(instr):
    r = client.chat.completions.create(model="student", temperature=0.2,
        messages=[{"role": "user", "content": instr + HINT}])
    return r.choices[0].message.content

def main():
    samples = []
    for line in open(C.EVAL_FILE):
        rec = json.loads(line)
        if rec.get("lang") == "C" and rec.get("tests"):
            instr = next(m["content"] for m in rec["messages"] if m["role"] == "user")
            samples.append((instr, rec["tests"]))

    raw_pass = fixed_pass = 0
    for i, (instr, tests) in enumerate(samples, 1):
        code = extract_code(solve(instr))
        raw_ok = compile_run(code, tests)[0] == "PASS"
        fixed_ok = compile_run(fix_c_includes(code), tests)[0] == "PASS"
        raw_pass += raw_ok; fixed_pass += fixed_ok
        flag = "  <== RESCUED" if (fixed_ok and not raw_ok) else ""
        print(f"[{i:>2}/{len(samples)}] raw={'P' if raw_ok else 'F'} "
              f"fixed={'P' if fixed_ok else 'F'}{flag}", flush=True)

    n = len(samples)
    print(f"\n=== auto-#include post-processor on v2 student ({n} C samples) ===")
    print(f"  raw       {raw_pass}/{n} = {raw_pass/n:.1%}")
    print(f"  +fixer    {fixed_pass}/{n} = {fixed_pass/n:.1%}   (+{(fixed_pass-raw_pass)} rescued)")

if __name__ == "__main__":
    main()
