"""C-failure diagnosis: re-run the student on the C eval samples, but instead of a
bare PASS/FAIL, capture the student's actual code + the exact gcc/runtime error and
bucket the failure modes. Tells us WHY C is weaker than Python so we can fix the
right thing (prompt vs data). Student must be served on :8091 (./serve_student.sh)."""
import os, re, json, subprocess, tempfile, shutil
from collections import Counter
from openai import OpenAI
import config as C, gen

client = OpenAI(base_url=C.TEACHER_BASE_URL, api_key=C.TEACHER_API_KEY)
C_HINT = (" Provide ONLY the implementation (functions/types and any needed "
          "#includes); do NOT write a main() function.")

def solve(instruction):
    r = client.chat.completions.create(model="student", temperature=0.2,
        messages=[{"role": "user", "content": instruction + C_HINT}])
    return r.choices[0].message.content

def extract_code(text):
    m = re.search(r"```[a-zA-Z0-9+#]*\n(.*?)```", text, re.S)
    return m.group(1) if m else text

HAS_MAIN = re.compile(r"\bint\s+main\s*\(|\bvoid\s+main\s*\(|\bmain\s*\(\s*(void|int|\))", re.I)

def compile_run(solution, tests):
    """Return (status, detail). status in PASS/COMPILE_FAIL/RUNTIME_FAIL/TIMEOUT."""
    src = f"{solution}\n\n{tests}\n"
    d = tempfile.mkdtemp()
    cpath, bpath = os.path.join(d, "prog.c"), os.path.join(d, "prog")
    try:
        with open(cpath, "w") as f:
            f.write(src)
        comp = subprocess.run(["gcc", "-std=c11", "-O0", "-w", cpath, "-o", bpath, "-lm"],
                              capture_output=True, timeout=C.EXEC_TIMEOUT, text=True)
        if comp.returncode != 0:
            return "COMPILE_FAIL", comp.stderr
        run = subprocess.run([bpath], capture_output=True, timeout=C.EXEC_TIMEOUT, text=True)
        if run.returncode == 0:
            return "PASS", ""
        return "RUNTIME_FAIL", f"exit={run.returncode} stderr={run.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return "TIMEOUT", ""
    except Exception as e:
        return "COMPILE_FAIL", f"harness:{e}"
    finally:
        shutil.rmtree(d, ignore_errors=True)

def classify_compile(stderr, wrote_main):
    """Bucket a compile failure by its dominant cause."""
    s = stderr.lower()
    if "redefinition of 'main'" in s or "multiple definition of" in s and "main" in s:
        return "dup_main (student wrote main())"
    if wrote_main and ("main" in s):
        return "dup_main (student wrote main())"
    if "implicit declaration" in s:
        return "missing_include/decl"
    if "unknown type name" in s or "incomplete type" in s:
        return "unknown_type (missing struct/typedef)"
    if "conflicting types" in s or "incompatible" in s:
        return "signature_mismatch"
    if "undefined reference" in s:
        return "link_undefined (fn not defined)"
    if "expected" in s or "error: '" in s:
        return "syntax_error"
    return "other_compile"

def main():
    samples = []
    for line in open(C.EVAL_FILE):
        rec = json.loads(line)
        if rec.get("lang") == "C" and rec.get("tests"):
            instr = next(m["content"] for m in rec["messages"] if m["role"] == "user")
            samples.append((instr, rec["tests"]))

    buckets = Counter()
    wrote_main_count = 0
    examples = {}                       # bucket -> one (instr, code, err)
    n_pass = 0
    for i, (instr, tests) in enumerate(samples, 1):
        code = extract_code(solve(instr))
        wrote_main = bool(HAS_MAIN.search(code))
        wrote_main_count += wrote_main
        status, detail = compile_run(code, tests)
        if status == "PASS":
            n_pass += 1
            bucket = "PASS"
        elif status == "COMPILE_FAIL":
            bucket = classify_compile(detail, wrote_main)
        else:
            bucket = status                       # RUNTIME_FAIL / TIMEOUT
        buckets[bucket] += 1
        if bucket not in examples and bucket != "PASS":
            examples[bucket] = (instr[:160], code, detail[:400])
        print(f"[{i:>2}/{len(samples)}] {bucket:<38} main={'Y' if wrote_main else 'n'}",
              flush=True)

    n = len(samples)
    print(f"\n=== C DIAGNOSIS ({n} samples) ===")
    print(f"PASS {n_pass}/{n} = {n_pass/n:.1%}")
    print(f"student wrote its own main() in {wrote_main_count}/{n} = {wrote_main_count/n:.1%} of replies")
    print("\nfailure buckets:")
    for b, c in buckets.most_common():
        if b == "PASS":
            continue
        print(f"  {c:>3}  {b}")

    print("\n=== one example per failure bucket ===")
    for b, (instr, code, err) in examples.items():
        print(f"\n----- [{b}] -----")
        print(f"TASK: {instr}")
        print(f"STUDENT CODE (first 500 chars):\n{code[:500]}")
        print(f"ERROR: {err}")

if __name__ == "__main__":
    main()
