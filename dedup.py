"""Phase 2 — dedup raw.jsonl and split into train / held-out.

- Exact dedup on the user instruction (normalized).
- Near-dup removal with rapidfuzz token_set_ratio (drops paraphrases the teacher
  produced across passes).
- Deterministic held-out split (no RNG, so reruns are stable) using a hash bucket.
"""
import json, re, hashlib
from rapidfuzz import fuzz
import config as C

NEAR_DUP_THRESHOLD = 92   # 0-100; higher = only very close dups removed

def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())

def instruction_of(rec):
    for m in rec["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""

def is_heldout(instr):
    # deterministic bucket: last byte of sha1 -> [0,1)
    h = hashlib.sha1(instr.encode()).digest()[-1] / 255.0
    return h < C.HELDOUT_FRAC

def main():
    seen_exact = set()
    kept = []          # (normalized_instr, rec)
    n_in = n_exact = n_near = 0

    with open(C.RAW_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)
            instr = instruction_of(rec)
            key = norm(instr)
            if not key:
                continue
            if key in seen_exact:
                n_exact += 1
                continue
            # near-dup check against what we've kept (cheap: compare to recent tail)
            dup = False
            for prev_key, _ in kept[-400:]:      # window keeps this ~linear
                if fuzz.token_set_ratio(key, prev_key) >= NEAR_DUP_THRESHOLD:
                    dup = True
                    break
            if dup:
                n_near += 1
                continue
            seen_exact.add(key)
            kept.append((key, rec))

    n_train = n_held = 0
    with open(C.CLEAN_FILE, "w") as tr, open(C.HELDOUT_FILE, "w") as ho:
        for key, rec in kept:
            if is_heldout(key):
                ho.write(json.dumps(rec) + "\n"); n_held += 1
            else:
                tr.write(json.dumps(rec) + "\n"); n_train += 1

    print(f"in={n_in} exact_dups={n_exact} near_dups={n_near}")
    print(f"-> {C.CLEAN_FILE}: {n_train}   {C.HELDOUT_FILE}: {n_held}")

if __name__ == "__main__":
    main()
