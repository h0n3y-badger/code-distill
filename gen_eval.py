"""Generate a dedicated, execution-verified EVAL set — Python + C, verifiable
tasks only, each compiled/run and kept only if it passes, and DISJOINT from
clean.jsonl (anti-contamination: testing on training data is the Monolith-1 sin).
Writes eval_set.jsonl. Run while the teacher is still loaded, before training."""
import json, re
import config as C, gen
from rapidfuzz import process, fuzz

TARGET      = 120
MAX_PASSES  = 4
LANGS       = ["Python", "C"]
VERIF_TASKS = [("implement a self-contained function", True),
               ("find and fix the bug in this snippet", True)]

def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())

# Load training instructions so we can guarantee the eval set doesn't overlap.
train = []
for line in open(C.CLEAN_FILE):
    for m in json.loads(line)["messages"]:
        if m["role"] == "user":
            train.append(norm(m["content"]))
train_set = set(train)
print(f"loaded {len(train)} training instructions for contamination check", flush=True)

combos = [(lang, dom, t, d)
          for lang in LANGS
          for dom in C.LANG_DOMAINS[lang]
          for t in VERIF_TASKS
          for d in C.DIFFICULTIES]

kept = 0
seen_eval = set()
with open(C.EVAL_FILE, "w") as out:
    for p in range(MAX_PASSES):
        if kept >= TARGET:
            break
        for lang, dom, (task, verif), diff in combos:
            if kept >= TARGET:
                break
            try:
                obj = gen.extract_json(gen.ask(
                    gen.build_prompt(lang, dom, task, diff, verif), C.GEN_TEMPERATURE))
                instr = obj["instruction"]
                sol = gen.strip_fence(obj["solution"])
                tests = gen.strip_fence(obj.get("tests"))
            except Exception:
                continue
            if not tests:
                continue
            k = norm(instr)
            if k in seen_eval or k in train_set:                 # exact overlap
                continue
            if process.extractOne(k, train, scorer=fuzz.token_set_ratio,
                                   score_cutoff=92):              # near-dup vs training
                continue
            if not gen.verify(lang, sol, tests):                 # execution gate
                continue
            seen_eval.add(k)
            out.write(json.dumps({
                "messages": [{"role": "user", "content": instr},
                             {"role": "assistant", "content": sol}],
                "lang": lang, "task": task, "difficulty": diff, "tests": tests}) + "\n")
            out.flush()
            kept += 1
            if kept % 10 == 0:
                print(f"kept {kept}/{TARGET}", flush=True)

print(f"DONE. eval set: {kept} verified, training-disjoint samples -> {C.EVAL_FILE}")
