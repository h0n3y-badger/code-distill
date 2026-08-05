"""v12: make tool-result training data FAITHFUL to what searxng/LM Studio actually
feeds the model. The v10/v11 field failures were all triggered by a LARGE, NOISY
multi-result search blob at realistic temperature — but every research/recall/
enrich/subtask row was trained on a SMALL, CLEAN JSON result. So the model never
learned to stay in the right mode when the result is a big messy blob; it tips
into rendering an HTML/markdown DOCUMENT.

This does two teacher-FREE transforms on multi_raw.jsonl -> multi_aug.jsonl:

  1) INFLATE every role:tool message into a ~18-result searxng-style blob that
     EMBEDS the original result (so the assistant answer stays grounded) among
     realistic noise. Applied to ALL tool-bearing types — crucially INCLUDING
     enrich/subtask, whose correct output is still a DOC. That keeps the learned
     discriminator = USER INTENT (answer vs. doc), NOT blob size (the chess rule:
     don't fix prose-after-search by breaking doc-after-search).

  2) DERIVE a `freshresearch` row from each `research` row: drop the artifact
     turns so it's a FRESH chat (plain question -> search -> PROSE). The field
     failure was a fresh chat; our only fresh coverage was 44 recall rows.

Deterministic (seeded). No teacher, no GPU — just a reshape + retrain.
"""
import json, re, random
import config_uni as C

RNG = random.Random(3407)

_NOISE_SUFFIX = ["- Wikipedia", "| Reddit", "- YouTube", "on Facebook", "| News",
                 "- Tracker", "Explained", "| Forum", "- Blog", "Live Updates",
                 "(2026)", "| Official", "- Fandom", "on X", "| Analysis"]
_NOISE_DESC = [
    "Community discussion thread with speculation and links to primary sources.",
    "Live coverage with countdown, weather notes, and logistics; schedule is dynamic.",
    "Encyclopedic overview: history, prior attempts, and technical parameters.",
    "Video stream and replay coverage with commentary from enthusiasts.",
    "Social post with a partial summary and a long comment section.",
    "Tracker page with real-time status, visibility scoring, and maps.",
    "Opinion/analysis piece with background context and quotes.",
    "FAQ-style rundown of dates, times, and what to expect.",
]


def _topic_hint(msgs):
    # prefer the search query in the tool_call; else the first user turn
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            try:
                a = json.loads(tc["function"]["arguments"])
                if a.get("query"):
                    return str(a["query"])
            except Exception:
                pass
    for m in msgs:
        if m.get("role") == "user":
            return (m.get("content") or "")[:60]
    return "the topic"


def _noisy_blob(original, topic, seed):
    """A ~18-entry searxng-style blob: the ORIGINAL result kept as the top,
    highest-relevance hit (answer stays grounded), padded with plausible noise."""
    r = random.Random(seed)
    orig_txt = original if isinstance(original, str) else json.dumps(original)
    orig_txt = orig_txt.strip()
    # if the original is already a searxng [{"type":"text","text":...}] blob, lift its text
    try:
        parsed = json.loads(orig_txt)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "text" in parsed[0]:
            orig_txt = parsed[0]["text"]
    except Exception:
        pass
    entries = [f"Title: {topic} - Primary source\nDescription: {orig_txt}\n"
               f"URL: https://official.example.com/{abs(hash(topic))%9999}\nRelevance Score: 9.0"]
    n = r.randint(14, 22)
    for i in range(n):
        suf = r.choice(_NOISE_SUFFIX); desc = r.choice(_NOISE_DESC)
        score = round(r.uniform(0.03, 2.6), 3)
        entries.append(f"Title: {topic} {suf}\nDescription: {desc}\n"
                       f"URL: https://{['news','en.wikipedia','reddit','youtube','facebook','x'][i%6]}.example.com/{i}\n"
                       f"Relevance Score: {score}")
    r.shuffle(entries[1:])  # keep primary first-ish but let some noise outrank it
    return json.dumps([{"type": "text", "text": "\n\n".join(entries)}])


_DOCSTOP = re.compile(
    r"^\s*(no[,.]?\s+|actually[,.]?\s+|stop (making|creating) (documents?|docs?)[;,.]?\s*|"
    r"instead[,.]?\s*|skip (the )?(doc|document|html|page)[,.:—-]*\s*|"
    r"forget (the )?(doc|html|page)[,.:—-]*\s*)+", re.I)


def _fresh_opening(research_req):
    """Turn a follow-up 'stop making docs, search X' into a standalone opener."""
    s = _DOCSTOP.sub("", research_req).strip()
    if not s or len(s) < 12:
        s = research_req.strip()
    s = s[0].upper() + s[1:] if s else research_req
    return s


def main():
    src = C.MULTI_RAW
    out_path = "multi_aug.jsonl"
    n_in = n_out = n_fresh = n_inflated = 0
    with open(out_path, "w") as out:
        for li, line in enumerate(open(src)):
            o = json.loads(line); n_in += 1
            msgs = o["messages"]; topic = _topic_hint(msgs)
            # 1) inflate every tool message in place
            ti = 0
            for m in msgs:
                if m.get("role") == "tool":
                    m["content"] = _noisy_blob(m.get("content", ""), topic, seed=li * 17 + ti)
                    ti += 1
            if ti:
                n_inflated += 1
            out.write(json.dumps(o) + "\n"); n_out += 1
            # 2) derive a freshresearch row from each research row
            if o.get("mtype") == "research":
                # research shape: [u art_req, a artifact, u research_req, a tool_call, tool, a answer]
                idx_u = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
                if len(idx_u) >= 2:
                    rreq = msgs[idx_u[1]]["content"]
                    tail = msgs[idx_u[1] + 1:]  # tool_call, tool result, prose answer
                    fresh = [{"role": "user", "content": _fresh_opening(rreq)}] + \
                            [dict(m) for m in tail]
                    row = {"messages": fresh, "kind": "multi", "mtype": "freshresearch"}
                    if o.get("tools_spec"):
                        row["tools_spec"] = o["tools_spec"]
                    out.write(json.dumps(row) + "\n"); n_out += 1; n_fresh += 1
    print(f"augment: {n_in} in -> {n_out} out | inflated {n_inflated} tool rows | "
          f"+{n_fresh} freshresearch -> {out_path}")


if __name__ == "__main__":
    main()
