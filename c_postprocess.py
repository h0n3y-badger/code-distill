"""Deterministic post-processor that fixes the #1 C failure mode: the 7B student
writes correct logic but omits stdlib #includes (modern gcc hard-errors). Rather
than blindly prepend headers, we compile, read gcc's OWN "'X' is defined in header
'<h>'" hints, add exactly those, and retry. This is the proven lever (+~10pp) that
retraining could NOT buy. Usable at inference wherever C output will be compiled."""
import re, subprocess, tempfile, os, shutil

HDR_HINT = re.compile(r"is defined in header '<([^>]+)>'")
# fallback map for symbols gcc names without a header hint
SYMBOL_HDR = {"NULL": "stddef.h", "true": "stdbool.h", "false": "stdbool.h",
              "bool": "stdbool.h", "size_t": "stddef.h"}
SYMBOL_RE = re.compile(r"'(\w+)' undeclared|unknown type name '(\w+)'")

def _compile(src, compile_only=False, timeout=10):
    d = tempfile.mkdtemp()
    c = os.path.join(d, "p.c")
    try:
        with open(c, "w") as f:
            f.write(src)
        # compile_only (-c): no link step, so a solution with no main() still
        # surfaces its #include errors without a spurious "undefined main".
        cmd = ["gcc", "-std=c11", "-O0", "-w", c, "-o", os.path.join(d, "o")]
        cmd += ["-c"] if compile_only else ["-lm"]
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
        return p.returncode, p.stderr
    except Exception as e:
        return 1, f"harness:{e}"
    finally:
        shutil.rmtree(d, ignore_errors=True)

def fix_c_includes(solution, tests="", max_iter=6):
    """Add missing stdlib #includes to `solution` until it compiles or we stop
    learning new headers. If `tests` given, compile the pair (link); otherwise
    compile the solution alone (-c) — the honest at-inference case with no tests."""
    have = set(re.findall(r"#include\s*<([^>]+)>", solution))
    sol = solution
    for _ in range(max_iter):
        if tests:
            rc, err = _compile(f"{sol}\n\n{tests}\n")
        else:
            rc, err = _compile(sol, compile_only=True)
        if rc == 0:
            break
        wanted = set(HDR_HINT.findall(err))
        for m in SYMBOL_RE.findall(err):
            sym = m[0] or m[1]
            if sym in SYMBOL_HDR:
                wanted.add(SYMBOL_HDR[sym])
        new = wanted - have
        if not new:
            break                       # gcc has no more header suggestions -> give up
        have |= new
        block = "".join(f"#include <{h}>\n" for h in sorted(new))
        sol = block + sol
    return sol

if __name__ == "__main__":
    demo = ('int hasCycle(struct ListNode *h){struct ListNode*s=h,*f=h;'
            'while(f&&f->next){s=s->next;f=f->next->next;if(s==f)return 1;}return 0;}')
    print("before:\n", demo[:60], "...")
    print("\nafter:\n", fix_c_includes(demo)[:120], "...")
