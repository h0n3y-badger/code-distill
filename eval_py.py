"""Python-only execution eval — for the 1.5B models. Hits whatever is served on
:8091 (student or base), scores pass@1 by running solution+tests. Prints a single
PY_PASS line for easy scraping. Usage: serve a model on :8091, then run this."""
import json
from openai import OpenAI
import config as C, gen

client = OpenAI(base_url=C.TEACHER_BASE_URL, api_key=C.TEACHER_API_KEY)

def solve(instr):
    r = client.chat.completions.create(model="m", temperature=0.2,
        messages=[{"role": "user", "content": instr}])
    return r.choices[0].message.content

def extract_code(text):
    import re
    m = re.search(r"```[a-zA-Z0-9+#]*\n(.*?)```", text, re.S)
    return m.group(1) if m else text

def main():
    p = t = 0
    for line in open(C.EVAL_FILE):
        rec = json.loads(line)
        if rec.get("lang") != "Python" or not rec.get("tests"):
            continue
        instr = next(m["content"] for m in rec["messages"] if m["role"] == "user")
        ok = gen.verify("Python", extract_code(solve(instr)), rec["tests"])
        p += ok; t += 1
        print(f"[{t:>2}] {'PASS' if ok else 'FAIL'} running={p/t:.1%}", flush=True)
    print(f"PY_PASS {p}/{t} = {p/t:.1%}")

if __name__ == "__main__":
    main()
