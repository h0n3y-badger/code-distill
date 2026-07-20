"""C-boost generation: C is the weak language (62% vs Python 83%). Generate a large
batch of NEW, execution-verified C samples using the upgraded build_prompt (self-
contained #includes + explicit signatures), appending to raw.jsonl. Verifiable tasks
only (implement / fix-bug) so every kept sample compiled + ran green. Teacher must be
served on :8091 (serve_teacher.sh). Stops at TARGET kept or MAX_PASSES."""
import json, re
import config as C, gen

TARGET     = 120
MAX_PASSES = 10
VERIF_TASKS = [("implement a self-contained function", True),
               ("find and fix the bug in this snippet", True)]

def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())

# Don't re-add tasks we already have (exact-instruction dedup vs existing raw).
seen = set()
try:
    for line in open(C.RAW_FILE):
        for m in json.loads(line)["messages"]:
            if m["role"] == "user":
                seen.add(norm(m["content"]))
except FileNotFoundError:
    pass
print(f"{len(seen)} existing instructions loaded for dedup", flush=True)

combos = [(dom, t, d)
          for dom in C.LANG_DOMAINS["C"]
          for t in VERIF_TASKS
          for d in C.DIFFICULTIES]

kept = 0
with open(C.RAW_FILE, "a") as out:
    for p in range(MAX_PASSES):
        if kept >= TARGET:
            break
        for dom, (task, verif), diff in combos:
            if kept >= TARGET:
                break
            try:
                obj = gen.extract_json(gen.ask(
                    gen.build_prompt("C", dom, task, diff, verif), C.GEN_TEMPERATURE))
                instr = obj["instruction"]
                sol = gen.strip_fence(obj["solution"])
                tests = gen.strip_fence(obj.get("tests"))
            except Exception:
                continue
            if not tests:
                continue
            k = norm(instr)
            if k in seen:
                continue
            if not gen.verify("C", sol, tests):     # gcc compile + run gate
                continue
            seen.add(k)
            out.write(json.dumps({
                "messages": [{"role": "user", "content": instr},
                             {"role": "assistant", "content": sol}],
                "lang": "C", "task": task, "difficulty": diff, "tests": tests}) + "\n")
            out.flush()
            kept += 1
            if kept % 10 == 0:
                print(f"[pass {p+1}/{MAX_PASSES}] kept {kept}/{TARGET}", flush=True)

print(f"DONE. added {kept} new verified C samples -> {C.RAW_FILE}")
