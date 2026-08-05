"""THE CONSTANT TEST — one frozen suite, run identically against every version so
scores are comparable. Four axes:

  C      execution pass@1  (gcc compile+run; regression guard on retained coding)
  web    HTML structural validity + required-element checks
  tools  emits a schema-valid tool_call for the right tool / correctly declines
  chat   transcript capture (+ light keyword hits) for manual quiz

Serve the student on :8091 with --jinja (serve_uni.sh) FIRST. Writes a row to
UNI_RESULTS.md and full transcripts to uni_eval_<VER>.json for hand review.

  VER=uni-1 STUDENT_ID=qwen-uni-7b python eval_universal.py
"""
import os, re, json, subprocess, tempfile, shutil, datetime
from openai import OpenAI
from datalib import (is_valid_html, html_issues, validate_tool_call,
                     extract_tool_calls, has_fence, is_prose_reply)

STUDENT_URL = os.environ.get("STUDENT_URL", "http://localhost:8091/v1")
STUDENT_ID  = os.environ.get("STUDENT_ID", "student")
VER         = os.environ.get("VER", "unlabeled")
C_EVAL_N    = int(os.environ.get("C_EVAL_N", "12"))
EXEC_TIMEOUT = 10

client = OpenAI(base_url=STUDENT_URL, api_key="sk-noauth")
GOLD = json.load(open("uni_eval.json"))


def chat(messages, tools=None, temp=0.2, max_tokens=2048):
    kw = dict(model=STUDENT_ID, messages=messages, temperature=temp,
              max_tokens=max_tokens)
    if tools is not None:
        kw["tools"] = tools
    try:
        return client.chat.completions.create(**kw)
    except Exception as e:
        if tools is not None:            # server may reject the tools param
            kw.pop("tools")
            return client.chat.completions.create(**kw)
        raise


def extract_code(text):
    m = re.search(r"```[a-zA-Z0-9+#.\-]*\n(.*?)```", text or "", re.S)
    return m.group(1) if m else (text or "")


# --- C execution pass@1 -------------------------------------------------------
C_HINT = (" Provide ONLY the implementation (functions/types and any needed "
          "#includes); do NOT write a main() function.")


def c_passes(solution, tests):
    src = f"{solution}\n\n{tests}\n"
    d = tempfile.mkdtemp()
    cpath, bpath = os.path.join(d, "p.c"), os.path.join(d, "p")
    try:
        open(cpath, "w").write(src)
        comp = subprocess.run(["gcc", "-std=c11", "-O0", "-w", cpath, "-o", bpath, "-lm"],
                              capture_output=True, timeout=EXEC_TIMEOUT)
        if comp.returncode != 0:
            return False
        run = subprocess.run([bpath], capture_output=True, timeout=EXEC_TIMEOUT)
        return run.returncode == 0
    except Exception:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def load_c_items():
    items = []
    for line in open("eval_set.jsonl"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("lang") == "C" and r.get("tests"):
            instr = next(m["content"] for m in r["messages"] if m["role"] == "user")
            items.append({"prompt": instr, "tests": r["tests"]})
            if len(items) >= C_EVAL_N:
                break
    return items


def score_c(tr):
    items = load_c_items()
    passed = 0
    for it in items:
        reply = chat([{"role": "user", "content": it["prompt"] + C_HINT}]).choices[0].message.content
        code = extract_code(reply)
        ok = c_passes(code, it["tests"])
        passed += ok
        tr.append({"axis": "c", "prompt": it["prompt"][:120], "pass": bool(ok),
                   "reply": reply[:1500]})
    return passed, len(items)


def score_web(tr):
    """Returns (valid_passed, total, fenced_count). `fenced` is the v8 axis: did
    the reply put its code in a ``` block (a 'code box') rather than bare text?"""
    passed = fenced = 0
    for it in GOLD["web"]:
        reply = chat([{"role": "user", "content": it["prompt"]}]).choices[0].message.content
        code = extract_code(reply)
        valid = is_valid_html(code, allow_fragment=not it.get("full"))
        was_fenced = has_fence(reply)
        fenced += was_fenced
        hits = [pat for pat in it["must_contain"]
                if re.search(pat, code, re.I)]
        ok = valid and len(hits) == len(it["must_contain"])
        passed += ok
        tr.append({"axis": "web", "prompt": it["prompt"][:120], "pass": bool(ok),
                   "valid_html": valid, "fenced": was_fenced,
                   "missing": [p for p in it["must_contain"]
                   if not re.search(p, code, re.I)], "issues": html_issues(code)[:5],
                   "reply": reply[:1500]})
    return passed, len(GOLD["web"]), fenced


def score_modeswitch(tr):
    """v8: after producing an artifact, does a PLAIN follow-up get a prose answer
    (not another code dump / tool call)? This is the exact v6 mode-lock failure.
    Each gold item supplies conversation history ending in an assistant artifact,
    then a plain user question."""
    passed = 0
    items = GOLD.get("modeswitch", [])
    for it in items:
        msgs = list(it["history"]) + [{"role": "user", "content": it["question"]}]
        reply = chat(msgs).choices[0].message.content or ""
        prose = is_prose_reply(reply)
        need = it.get("must_include", [])
        kw_ok = all(k.lower() in reply.lower() for k in need)
        ok = prose and kw_ok
        passed += ok
        tr.append({"axis": "modeswitch", "question": it["question"], "pass": bool(ok),
                   "is_prose": prose, "kw_ok": kw_ok, "reply": reply[:800]})
    return passed, len(items)


def score_subtask(tr):
    """v8: a 'search/look-up X and THEN build/summarize' request must FIRE the
    tool (not hallucinate the data), even though the ask also wants an artifact."""
    passed = 0
    items = GOLD.get("subtask", [])
    for it in items:
        resp = chat([{"role": "user", "content": it["prompt"]}], tools=it["tools"])
        msg = resp.choices[0].message
        calls = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                calls.append({"name": tc.function.name})
        else:
            parsed, _ = extract_tool_calls(msg.content or "")
            calls = [{"name": c["name"]} for c in parsed]
        want = it["expect"]["tool"]
        ok = any(c["name"] == want for c in calls)
        passed += ok
        tr.append({"axis": "subtask", "prompt": it["prompt"][:120], "pass": bool(ok),
                   "expected": want, "got": [c["name"] for c in calls],
                   "content": (msg.content or "")[:300]})
    return passed, len(items)


def score_artifact_search(tr):
    """v9: the v8 field-test failure. An HTML artifact ALREADY exists in history;
    the user then asks to SEARCH for real info and update it. The model must emit
    a tool_call (not keep blindly editing, not embed JS calling the tool)."""
    passed = 0
    items = GOLD.get("artifact_search", [])
    for it in items:
        msgs = list(it["history"]) + [{"role": "user", "content": it["enrich"]}]
        resp = chat(msgs, tools=it["tools"])
        m = resp.choices[0].message
        calls = []
        if getattr(m, "tool_calls", None):
            calls = [{"name": tc.function.name} for tc in m.tool_calls]
        else:
            parsed, _ = extract_tool_calls(m.content or "")
            calls = [{"name": c["name"]} for c in parsed]
        want = it["expect"]["tool"]
        ok = any(c["name"] == want for c in calls)
        passed += ok
        tr.append({"axis": "artifact_search", "enrich": it["enrich"][:120],
                   "pass": bool(ok), "expected": want, "got": [c["name"] for c in calls],
                   "content": (m.content or "")[:300]})
    return passed, len(items)


_RAW_HTML_RE = re.compile(r"<(?:!doctype|html|body|div|p|h[1-6]|ul|table|span|section|head)\b", re.I)


def _is_plain_prose(text):
    """Stricter than is_prose_reply for this axis: also reject UN-fenced raw HTML
    (a bare <p>/<!doctype dump), not just fenced code + <tool_call>."""
    return is_prose_reply(text) and not _RAW_HTML_RE.search(text or "")


RP_SAMPLES = int(os.environ.get("RP_SAMPLES", "8"))
# v12: temp matters — the field failure only surfaces near LM Studio's GUI default
# (~0.8), NOT at the 0.2-0.4 my earlier evals used. Default high on purpose.
RP_TEMP    = float(os.environ.get("RP_TEMP", "0.8"))


def score_research_prose(tr):
    """v12 (FAITHFUL): the field failure. After a web_search returns a LARGE, NOISY
    searxng-style blob, the user wanted a plain answer ('give me the profile') — the
    model must reply in PLAIN PROSE, but it intermittently renders an HTML/markdown
    DOCUMENT (<details>/<div>/```html) instead. Two levers drive it and BOTH must be
    in the harness or it hides: (1) high temp (~0.8, LM Studio default), (2) a big
    messy multi-result blob. Small/clean blobs at low temp = 0% fail = false green
    (the v10/v11 mistake). Sample RP_SAMPLES x at RP_TEMP; report the prose RATE.
    Items carry variant 'fresh' (no artifact) or 'artifact' (artifact in history)."""
    passed = total = 0
    items = GOLD.get("research_prose", [])
    for it in items:
        oks = []
        for _ in range(RP_SAMPLES):
            reply = chat(list(it["history"]), tools=it.get("tools"),
                         temp=RP_TEMP).choices[0].message.content or ""
            ok = _is_plain_prose(reply)
            oks.append(ok); passed += ok; total += 1
            tr.append({"axis": "research_prose", "variant": it.get("variant", "?"),
                       "pass": bool(ok), "is_prose": is_prose_reply(reply),
                       "reply": reply[:400]})
        print(f"  research_prose[{it.get('variant','?')}]: {sum(oks)}/{len(oks)} prose")
    return passed, total


def score_tools(tr):
    passed = 0
    for it in GOLD["tools"]:
        resp = chat([{"role": "user", "content": it["prompt"]}], tools=it["tools"])
        msg = resp.choices[0].message
        calls = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                calls.append({"name": tc.function.name, "arguments": args})
        else:
            calls, _ = extract_tool_calls(msg.content or "")
        exp = it["expect"]
        if exp.get("none"):
            ok = len(calls) == 0
        else:
            want = exp["tool"]
            ok = False
            for c in calls:
                if c["name"] == want:
                    synth = "<tool_call>" + json.dumps(c) + "</tool_call>"
                    if validate_tool_call(synth, it["tools"])["ok"]:
                        ok = True
        passed += ok
        tr.append({"axis": "tools", "prompt": it["prompt"][:120], "pass": bool(ok),
                   "expected": exp, "got": calls, "content": (msg.content or "")[:400]})
    return passed, len(GOLD["tools"])


def score_chat(tr):
    hits = total_kw = 0
    for it in GOLD["chat"]:
        reply = chat([{"role": "user", "content": it["prompt"]}]).choices[0].message.content or ""
        need = it.get("must_include", [])
        got = [k for k in need if k.lower() in reply.lower()]
        hits += len(got); total_kw += len(need)
        tr.append({"axis": "chat", "prompt": it["prompt"], "pass": len(got) == len(need),
                   "kw_hits": f"{len(got)}/{len(need)}", "reply": reply[:1500]})
    kw = (hits / total_kw) if total_kw else 1.0
    return kw, len(GOLD["chat"])


def main():
    tr = []
    print(f"== constant test: VER={VER} model={STUDENT_ID} @ {STUDENT_URL} ==")
    cp, cn = score_c(tr);            print(f"C     pass@1 : {cp}/{cn} = {cp/cn:.1%}")
    wp, wn, wf = score_web(tr);      print(f"web   valid  : {wp}/{wn} = {wp/wn:.1%}  (fenced {wf}/{wn})")
    tp, tn = score_tools(tr);        print(f"tools correct: {tp}/{tn} = {tp/tn:.1%}")
    ck, cc = score_chat(tr);         print(f"chat  kw-hit : {ck:.1%} over {cc} prompts (see transcript)")
    # v8 axes — the field-test failures made measurable
    mp, mn = score_modeswitch(tr);   print(f"modeswitch   : {mp}/{mn} = {mp/mn:.1%}" if mn else "modeswitch   : (no gold)")
    sp, sn = score_subtask(tr);      print(f"tool-subtask : {sp}/{sn} = {sp/sn:.1%}" if sn else "tool-subtask : (no gold)")
    ap, an = score_artifact_search(tr); print(f"artifact-srch: {ap}/{an} = {ap/an:.1%}" if an else "artifact-srch: (no gold)")
    rp, rn = score_research_prose(tr); print(f"research-prose: {rp}/{rn} = {rp/rn:.1%}" if rn else "research-prose: (no gold)")

    dump = f"uni_eval_{VER}.json"
    json.dump({"ver": VER, "model": STUDENT_ID, "transcripts": tr}, open(dump, "w"), indent=1)
    stamp = os.environ.get("STAMP", "")  # pass a date in; Date.now unavailable in some envs
    # --- frozen original table (unchanged columns, comparable across all versions)
    row = (f"| {VER} | {cp}/{cn} ({cp/cn:.0%}) | {wp}/{wn} ({wp/wn:.0%}) | "
           f"{tp}/{tn} ({tp/tn:.0%}) | {ck:.0%} | {stamp} |")
    header = ("| ver | C exec | web valid | tools | chat kw | date |\n"
              "|---|---|---|---|---|---|\n")
    path = "UNI_RESULTS.md"
    if not os.path.exists(path):
        open(path, "w").write("# Universal constant-test results\n\n" + header)
    with open(path, "a") as f:
        f.write(row + "\n")
    # --- v8 axes table (separate so the frozen table stays aligned) -----------
    def pct(a, b):
        return f"{a}/{b} ({a/b:.0%})" if b else "n/a"
    v8row = f"| {VER} | {pct(wf, wn)} | {pct(mp, mn)} | {pct(sp, sn)} | {stamp} |"
    v8path = "UNI_RESULTS_V8.md"
    if not os.path.exists(v8path):
        open(v8path, "w").write(
            "# v8 axes — code-box (fenced), mode-switch, tool-as-subtask\n\n"
            "| ver | web fenced | modeswitch | tool-subtask | date |\n"
            "|---|---|---|---|---|\n")
    with open(v8path, "a") as f:
        f.write(v8row + "\n")
    print(f"\nappended -> {path} + {v8path}\ntranscripts -> {dump}  (read this for the manual quiz)")


if __name__ == "__main__":
    main()
