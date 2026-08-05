"""Build the universal training set: REPLAY (downsampled existing verified coding
data) + NEW web + chat + tools data. Exact-dedups on the first user turn, holds
out a small slice, and writes shuffled JSONL. Preserves per-row `kind` and (for
tool rows) `tools_spec` so train.py can render each row correctly.

Deterministic (seeded) so a rebuild reproduces the same split.
"""
import json, random, collections
import config_uni as C
from datalib import iter_jsonl, is_valid_messages, has_tool_turn, as_fenced_reply, has_fence

RNG = random.Random(3407)

# lang -> markdown fence tag. Artifact/code replies are wrapped so a renderer
# draws a code box (the v6 field-test 'plain text, no code box' complaint) and
# the student learns the fence habit. multi rows are already fenced by the
# generator; chat rows stay prose.
_LANG_FENCE = {"Python": "python", "C": "c", "Java": "java", "HTML": "html",
               "html": "html", "python": "python", "c": "c", "java": "java"}


def _intro_for(lang, code):
    if _LANG_FENCE.get(lang) == "html":
        return ("Here's a complete, self-contained HTML page:"
                if "<!doctype" in code.lower() else "Here's the HTML:")
    return f"Here's the {lang} solution:" if lang else "Here's the code:"


def normalize_row(obj):
    """Fence artifact/code assistant turns into proper 'intro + ```lang' replies.
    Leaves chat/tool/multi rows alone (multi is already fenced; chat is prose)."""
    if obj.get("kind") in ("multi", "chat", "tools") or has_tool_turn(obj["messages"]):
        return obj
    lang = obj.get("lang", "")
    fence = _LANG_FENCE.get(lang) or ("html" if obj.get("kind") == "web" else None)
    if not fence:
        return obj  # unknown / not code — don't touch
    for m in obj["messages"]:
        if m.get("role") == "assistant" and isinstance(m.get("content"), str) \
                and not has_fence(m["content"]):
            m["content"] = as_fenced_reply(_intro_for(lang, m["content"]),
                                           m["content"], fence)
    return obj


def first_user(msgs):
    for m in msgs:
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def row_sig(obj):
    """Dedup signature. For most rows the first user turn is enough. For TOOL
    rows the SAME query paired with a DIFFERENT offered toolset (or a different
    chosen tool) is a distinct training signal — so include the toolset + the
    called tool, else we'd wrongly drop useful tool-restraint / tool-choice
    variants."""
    msgs = obj["messages"]
    q = first_user(msgs)
    # multi rows: same opening ask under a different scenario TYPE is a distinct
    # signal — key on (query, type) so the four mode-switch types don't collide.
    if obj.get("kind") == "multi":
        return (q, obj.get("mtype", ""))
    if obj.get("kind") != "tools":
        return q
    names = tuple(sorted(t.get("function", {}).get("name", "")
                         for t in obj.get("tools_spec", [])))
    called = ""
    for m in msgs:
        if m.get("tool_calls"):
            called = m["tool_calls"][0]["function"]["name"]
            break
    return (q, names, called)


def row_ok(obj):
    """Validity gate that understands tool rows (whose assistant tool-call turn
    has empty content by design and must skip is_valid_messages)."""
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False
    # rows with a structured tool-call turn (tools + multi subtask/recall) have
    # an empty-content assistant turn by design -> skip is_valid_messages; the
    # generator already schema-verified the call. Just require a final answer.
    if obj.get("kind") == "tools" or has_tool_turn(msgs):
        return isinstance(msgs[-1].get("content"), str) and msgs[-1]["content"].strip()
    return is_valid_messages(msgs)


def load_replay():
    """Existing verified coding data, downsampled per REPLAY_KEEP fractions."""
    by_lang = collections.defaultdict(list)
    for o in iter_jsonl(C.REPLAY_SRC):
        if row_ok(o):
            by_lang[o.get("lang", "?")].append(o)
    out = []
    for lang, rows in by_lang.items():
        keep = C.REPLAY_KEEP.get(lang, 0.0)
        if keep >= 1.0:
            sel = rows
        elif keep <= 0.0:
            sel = []
        else:
            k = max(1, round(len(rows) * keep))
            sel = RNG.sample(rows, k)
        for o in sel:
            o["kind"] = o.get("kind", "code")
        out.extend(sel)
        print(f"  replay {lang}: {len(rows)} -> kept {len(sel)} (frac {keep})")
    return out


def load_new(path, label):
    rows = [o for o in iter_jsonl(path) if row_ok(o)] if _exists(path) else []
    print(f"  new {label}: {len(rows)} from {path}")
    return rows


def _exists(path):
    try:
        open(path).close(); return True
    except FileNotFoundError:
        return False


def main():
    print("Loading replay (existing verified coding data):")
    rows = load_replay()
    print("Loading new data:")
    rows += load_new(C.WEB_RAW, "web")
    rows += load_new(C.CHAT_RAW, "chat")
    rows += load_new(C.TOOLS_RAW, "tools")
    rows += load_new(C.MULTI_RAW, "multi")
    rows += load_new(C.IMAGE_GOLD, "image-gold")

    # fence/format-normalize artifact turns (web + replayed code) so every code
    # reply is a proper 'intro + ```lang' turn (the code-box + formatting fix)
    rows = [normalize_row(o) for o in rows]

    # kind-aware exact-dedup (tool rows keyed by query + toolset + called tool)
    seen, deduped = set(), []
    for o in rows:
        key = row_sig(o)
        if key and key in seen:
            continue
        seen.add(key); deduped.append(o)
    print(f"deduped {len(rows)} -> {len(deduped)}")

    RNG.shuffle(deduped)
    n_hold = max(1, round(len(deduped) * C.HELDOUT_FRAC))
    heldout, train = deduped[:n_hold], deduped[n_hold:]

    # Upweight rows by kind in TRAIN only (never heldout) to strengthen a skill.
    upweights = {"tools": getattr(C, "TOOL_UPWEIGHT", 1),
                 "code": getattr(C, "CODE_UPWEIGHT", 1)}
    for kind, up in upweights.items():
        if up > 1:
            extra = [o for o in train if o.get("kind") == kind] * (up - 1)
            train = train + extra
            print(f"{kind} upweight x{up}: +{len(extra)} rows")
    # img rows are kind=web but flagged; upweight the small placeholder-image gold
    img_up = getattr(C, "IMG_UPWEIGHT", 1)
    if img_up > 1:
        extra = [o for o in train if o.get("img")] * (img_up - 1)
        train = train + extra
        print(f"img upweight x{img_up}: +{len(extra)} rows")
    # v9: emphasize the tool-call-after-artifact rows (enrich/research)
    tl_up = getattr(C, "TOOLLOCK_UPWEIGHT", 1)
    tl_types = getattr(C, "TOOLLOCK_TYPES", set())
    if tl_up > 1 and tl_types:
        extra = [o for o in train if o.get("mtype") in tl_types] * (tl_up - 1)
        train = train + extra
        print(f"toollock upweight x{tl_up}: +{len(extra)} rows")
    # v11: research (search -> PROSE) gets its OWN, larger upweight so prose-after-
    # search out-masses fence-after-search (fixes the v10 field ```html-wrapped
    # answer to "no html, just search"). research is NOT in TOOLLOCK_TYPES, so the
    # base count here is clean x1 — this multiply is exact, not compounded.
    r_up = getattr(C, "RESEARCH_UPWEIGHT", 1)
    r_types = getattr(C, "RESEARCH_TYPES", {"research"})
    if r_up > 1:
        extra = [o for o in train if o.get("mtype") in r_types] * (r_up - 1)
        train = train + extra
        print(f"research upweight x{r_up} ({sorted(r_types)}): +{len(extra)} rows")
    RNG.shuffle(train)
    print(f"train now {len(train)}")

    kinds = collections.Counter(o.get("kind", "?") for o in train)
    with open(C.UNI_CLEAN, "w") as f:
        for o in train:
            f.write(json.dumps(o) + "\n")
    with open(C.UNI_HELDOUT, "w") as f:
        for o in heldout:
            f.write(json.dumps(o) + "\n")
    print(f"WROTE {len(train)} -> {C.UNI_CLEAN} | {len(heldout)} -> {C.UNI_HELDOUT}")
    print(f"train mix by kind: {dict(kinds)}")


if __name__ == "__main__":
    main()
