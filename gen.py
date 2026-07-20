"""Phase 1 — generate distillation data from the local teacher.

Seeded self-instruct over the config grid + rejection sampling for Python.
Appends JSONL to RAW_FILE; safe to ctrl-C and rerun (it resumes / adds more).

Each record:
  {"messages": [{"role":"user",...},{"role":"assistant",...}],
   "lang": str, "task": str, "difficulty": str,
   "tests": str | null}   # tests kept as metadata for held-out eval; not trained on
"""
import json, re, subprocess, tempfile, os, sys, shutil
from openai import OpenAI
import config as C

client = OpenAI(base_url=C.TEACHER_BASE_URL, api_key=C.TEACHER_API_KEY)

SYS = ("You generate high-quality, diverse coding training data. "
       "Return ONLY a single JSON object, no prose, no markdown fences.")

def extract_json(text):
    """Teachers love wrapping JSON in ```json fences or adding chatter. Dig it out."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.S)   # first {...} span
        if m:
            text = m.group(0)
    return json.loads(text)

def strip_fence(code):
    """Teacher sometimes wraps a field's code in ```lang ... ``` fences *inside* the
    JSON value. Strip them so the raw code compiles/runs. Applies to all langs."""
    if not isinstance(code, str):
        return code
    m = re.search(r"^\s*```[a-zA-Z0-9+#]*\s*\n(.*?)\n?```\s*$", code.strip(), re.S)
    return m.group(1) if m else code

def ask(user, temp):
    r = client.chat.completions.create(
        model=C.TEACHER_MODEL, temperature=temp,
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content

VERIFIABLE_LANGS = ("Python", "C")   # langs we can execute to reject bad samples

def python_passes(solution, tests):
    """Rejection sampling: run solution+tests; keep only if exit 0."""
    prog = f"{solution}\n\n{tests}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog); path = f.name
    try:
        p = subprocess.run([sys.executable, path],
                           capture_output=True, timeout=C.EXEC_TIMEOUT)
        return p.returncode == 0
    except Exception:
        return False
    finally:
        try: os.unlink(path)
        except OSError: pass

def c_passes(solution, tests):
    """Rejection sampling for C: compile solution+tests with gcc, run; keep only
    if it both BUILDS and exits 0. (solution = defs w/o main; tests = a main()
    with assert(); see build_prompt.)"""
    src = f"{solution}\n\n{tests}\n"
    d = tempfile.mkdtemp()
    cpath, bpath = os.path.join(d, "prog.c"), os.path.join(d, "prog")
    try:
        with open(cpath, "w") as f:
            f.write(src)
        comp = subprocess.run(["gcc", "-std=c11", "-O0", "-w", cpath, "-o", bpath, "-lm"],
                              capture_output=True, timeout=C.EXEC_TIMEOUT)
        if comp.returncode != 0:
            return False                       # didn't compile -> reject
        run = subprocess.run([bpath], capture_output=True, timeout=C.EXEC_TIMEOUT)
        return run.returncode == 0             # asserts passed -> keep
    except Exception:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)

def verify(lang, solution, tests):
    """Dispatch to the right execution gate; True = keep."""
    if lang == "Python":
        return python_passes(solution, tests)
    if lang == "C":
        return c_passes(solution, tests)
    return True                                # non-executable langs: trust the teacher

def build_prompt(lang, dom, task, diff, verifiable):
    want_tests = verifiable and lang in VERIFIABLE_LANGS
    sol_desc = "<complete, correct " + lang + " code>"
    instr_desc = "<the full task, including any code the user must work with>"
    if want_tests and lang == "Python":
        tests_desc = ('"<python assert-based tests that import/call the solution; '
                      'raise/exit nonzero on failure>"')
    elif want_tests and lang == "C":
        # Two hard-won fixes (see diag_c.py): the 7B student (1) omits stdlib
        # #includes -> modern gcc hard-errors, and (2) names types/functions
        # differently than the test expects. So force SELF-CONTAINED solutions
        # AND make the instruction pin the exact contract the test will call.
        sol_desc = ("<complete, SELF-CONTAINED C. Emit EVERY #include the code "
                    "needs yourself: <stdlib.h> (malloc/free), <stddef.h> (NULL), "
                    "<stdbool.h> (bool/true/false), <string.h> (str*), <stdio.h>, "
                    "<math.h> as applicable. Include all function AND struct/type "
                    "definitions. Do NOT write a main(). The test file includes "
                    "ONLY <assert.h>, so anything else your code uses you must "
                    "#include.>")
        instr_desc = ("<the full task. CRITICAL: state the EXACT function "
                      "signature(s) the solution must define (return type, name, "
                      "parameter types) and the EXACT name + fields of any struct/"
                      "typedef, so the tests call them unambiguously. If fixing a "
                      "bug, include the buggy code.>")
        tests_desc = ('"<a C main() that #includes <assert.h>, exercises the '
                      'solution using ONLY assert() (NO printf/puts/output of any '
                      'kind), and returns 0 on success. Use the EXACT function and '
                      'type names from the instruction. Compiled with the solution.>"')
    tests_field = (', "tests": ' + tests_desc + '}') if want_tests else ', "tests": null}'
    schema = ('{"instruction": "' + instr_desc + '", "solution": "' + sol_desc
              + '"' + tests_field)
    return (f"Invent ONE {diff} {lang} coding task in the domain of {dom}. "
            f"The task type is: '{task}'. Make it realistic and non-trivial; "
            f"avoid textbook clichés like fizzbuzz. Put RAW code in the JSON string "
            f"values — do NOT wrap code in markdown ``` fences. "
            f"Respond with JSON: {schema}")

def main():
    n_kept = n_seen = n_rejected = 0
    # per-language domains: each lang crosses only its own domain list
    grid = [(lang, dom, task_item, diff)
            for lang, doms in C.LANG_DOMAINS.items()
            for dom in doms
            for task_item in C.TASKS.items()
            for diff in C.DIFFICULTIES]
    with open(C.RAW_FILE, "a") as out:
        for p in range(C.PASSES):
            for lang, dom, (task, verifiable), diff in grid:
                n_seen += 1
                try:
                    obj = extract_json(ask(build_prompt(lang, dom, task, diff, verifiable),
                                           C.GEN_TEMPERATURE))
                    instr = obj["instruction"]
                    sol = strip_fence(obj["solution"])
                    tests = strip_fence(obj.get("tests"))
                except Exception:
                    continue
                if verifiable and lang in VERIFIABLE_LANGS and tests:
                    if not verify(lang, sol, tests):
                        n_rejected += 1
                        continue
                out.write(json.dumps({
                    "messages": [{"role": "user", "content": instr},
                                 {"role": "assistant", "content": sol}],
                    "lang": lang, "task": task, "difficulty": diff,
                    "tests": tests}) + "\n")
                out.flush()
                n_kept += 1
                if n_kept % 25 == 0:
                    print(f"[pass {p+1}/{C.PASSES}] kept={n_kept} "
                          f"seen={n_seen} rejected={n_rejected}", flush=True)
    print(f"DONE. kept={n_kept} seen={n_seen} rejected={n_rejected} -> {C.RAW_FILE}")

if __name__ == "__main__":
    main()
