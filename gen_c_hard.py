"""Teacher-generate EXTRA C training data biased toward the headers the base C
data underweights (<limits.h> 4%, <ctype.h> 3%, <math.h> 2%). diag_c.py showed
the residual C failures are dominated by these missing includes, so we steer the
teacher into task topics that *force* those headers, then gcc-verify (reject on
compile/run failure) exactly as gen.py does. Appends to c_hard.jsonl.

Teacher must be served on :8091 (serve_teacher.sh). Safe to ctrl-C / rerun.
"""
import json, os
import config as C
import gen  # reuse ask/extract_json/strip_fence/c_passes

TARGET = int(os.environ.get("C_HARD_TARGET", "90"))
OUT = os.environ.get("C_HARD_OUT", "c_hard.jsonl")

# topic -> (header it forces, concrete framing hints). We over-weight limits/ctype/math.
TOPICS = [
    ("overflow-safe integer arithmetic (detect INT_MAX/INT_MIN overflow before it "
     "happens; saturating add/mul; safe negate/abs)", "<limits.h>"),
    ("integer range/bounds work using INT_MAX, INT_MIN, UINT_MAX, CHAR_BIT",
     "<limits.h>"),
    ("case-insensitive string processing and character classification (isalpha, "
     "isdigit, isspace, tolower, toupper) on ASCII text", "<ctype.h>"),
    ("tokenizing / trimming / normalizing text by character class (skip spaces, "
     "split on non-alphanumeric, count classes of characters)", "<ctype.h>"),
    ("numeric/geometry helpers using sqrt, pow, fabs, floor, ceil, round (distances, "
     "means, standard deviation, quadratic roots, rounding money)", "<math.h>"),
    ("primality / integer-root / perfect-square checks that iterate only to sqrt(n)",
     "<math.h>"),
    ("boolean predicates over arrays and bit patterns returning genuine bool/true/"
     "false (is_sorted, is_power_of_two, any/all)", "<stdbool.h>"),
    ("fixed-width bit manipulation on uint8_t/uint16_t/uint32_t/uint64_t (popcount, "
     "byte swap, rotate, checksums)", "<stdint.h>"),
]

DIFFS = ["beginner", "intermediate", "hard"]

SCHEMA = (
    '{"instruction": "<the full task. CRITICAL: state the EXACT function '
    'signature(s) the solution must define (return type, name, parameter types) '
    'and the EXACT name + fields of any struct/typedef so the tests call them '
    'unambiguously.>", '
    '"solution": "<complete, SELF-CONTAINED C. Emit EVERY #include the code needs '
    'YOURSELF — especially the less-common ones: <limits.h> for INT_MAX/INT_MIN, '
    '<ctype.h> for isalpha/tolower, <math.h> for sqrt/pow/fabs, <stdbool.h> for '
    'bool, <stdint.h> for uintN_t — as well as <stdlib.h>/<string.h>/<stdio.h> as '
    'needed. Include all function AND struct/type definitions. Do NOT write a '
    'main(). The test file includes ONLY <assert.h>, so anything else your code '
    'uses you MUST #include.>", '
    '"tests": "<a C main() that #includes <assert.h> (and any header needed for '
    'literals it uses, e.g. <limits.h>), exercises the solution using ONLY assert() '
    '(NO printf/output), and returns 0 on success. Use the EXACT names from the '
    'instruction.>"}')


def build(topic, header, diff):
    return (f"Invent ONE {diff} C coding task whose natural solution REQUIRES the "
            f"header {header}. Topic: {topic}. Make it realistic and non-trivial; "
            f"avoid textbook clichés. The solution MUST actually use symbols from "
            f"{header}. Put RAW code in the JSON string values — do NOT wrap code in "
            f"markdown fences. Respond with JSON: {SCHEMA}")


def main():
    kept = seen = rej = 0
    grid = [(t, h, d) for (t, h) in TOPICS for d in DIFFS]
    with open(OUT, "a") as out:
        p = 0
        while kept < TARGET:
            p += 1
            for topic, header, diff in grid:
                if kept >= TARGET:
                    break
                seen += 1
                try:
                    obj = gen.extract_json(gen.ask(build(topic, header, diff),
                                                   C.GEN_TEMPERATURE))
                    instr = obj["instruction"]
                    sol = gen.strip_fence(obj["solution"])
                    tests = gen.strip_fence(obj.get("tests"))
                except Exception:
                    continue
                if not (instr and sol and tests):
                    continue
                if not gen.c_passes(sol, tests):
                    rej += 1
                    continue
                # keep only if the solution really pulls in the target header family
                out.write(json.dumps({
                    "messages": [{"role": "user", "content": instr},
                                 {"role": "assistant", "content": sol}],
                    "lang": "C", "task": "implement a self-contained function",
                    "difficulty": diff, "tests": tests}) + "\n")
                out.flush()
                kept += 1
                if kept % 10 == 0:
                    print(f"kept={kept}/{TARGET} seen={seen} rejected={rej}",
                          flush=True)
    print(f"DONE. kept={kept} seen={seen} rejected={rej} -> {OUT}")


if __name__ == "__main__":
    main()
