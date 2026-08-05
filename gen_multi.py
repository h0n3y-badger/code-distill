"""Generate MULTI-TURN mode-switch training data from the 32B teacher (v8).

This axis targets the exact failure the v6 field-test exposed: once the model
produced an artifact it got stuck in "emit code" mode — it would not re-call a
tool, answer a plain follow-up in prose, or revise on request; it just dumped
HTML again (and hallucinated image URLs instead of searching). Single-turn data
never taught the SWITCH between modes inside one conversation.

For each type the teacher supplies the pieces (tools, queries, a tool result, a
deliverable, a follow-up) and WE assemble the canonical message list and VERIFY
it with the same structural gates the eval uses:

  subtask    call a tool as a STEP toward a deliverable, then deliver (fenced)
  recall     answer, then a follow-up that needs ANOTHER tool call
  modeswitch produce an artifact, then answer a PLAIN question in PROSE
  revise     produce an artifact, then revise it on request

Artifacts are HTML so is_valid_html gives a real structural check; deliverable
turns are wrapped in a ```html fence via as_fenced_reply. Appends to MULTI_RAW;
ctrl-C safe; stops at TARGET_MULTI.
"""
import json, os
import config_uni as C
from genlib import ask, extract_json, strip_fence, count_lines
from datalib import (validate_tool_call, is_valid_html, is_prose_reply,
                     as_fenced_reply)
from gen_tools import norm_tools

SYS = ("You design realistic MULTI-TURN assistant training conversations that "
       "involve tools and/or building small web artifacts. Return ONLY one JSON "
       "object, no prose, no markdown fences. Put RAW HTML in the HTML string "
       "values (no ``` fences), cleanly indented, and keep each artifact SMALL "
       "(a self-contained fragment or mini-page, ~10-25 lines).")

_DOMAINS = ("weather, web search, email/calendar, file system, databases, maps, "
            "stock prices, unit conversion, home automation, HTTP APIs")

# --- per-type teacher prompt + expected JSON shape ----------------------------
_SHAPES = {
    "subtask": (
        "The user asks for a DELIVERABLE that first requires looking something "
        "up with a tool (e.g. 'search X and build me an HTML summary of it'). "
        "The assistant must call the tool, then USE its result to build the "
        "artifact. Provide 2-3 realistic tools; exactly one applies.",
        '{"tools":[{"name":"snake_case","description":"...","parameters":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}}],'
        '"query":"<the deliverable request that needs a lookup first>",'
        '"tool_name":"<tool to call>","arguments":{"a":"value"},'
        '"tool_result":"<realistic JSON string the tool returns>",'
        '"deliverable_intro":"<one short sentence introducing the artifact>",'
        '"deliverable_html":"<small, indented HTML that USES the tool result>"}'),
    "recall": (
        "A two-request conversation. Turn 1: the user asks something that needs "
        "a tool; the assistant calls it and answers in prose. Turn 2: a natural "
        "FOLLOW-UP that needs ANOTHER tool call. Provide 2-3 tools.",
        '{"tools":[...same shape as above...],'
        '"q1":"<first request>","tool1":"<tool>","args1":{},"result1":"<json str>","a1":"<prose answer using result1>",'
        '"q2":"<follow-up needing another call>","tool2":"<tool>","args2":{},"result2":"<json str>","a2":"<prose answer using result2>"}'),
    "modeswitch": (
        "The user first asks the assistant to BUILD a small HTML artifact, then "
        "asks a completely ordinary follow-up QUESTION (NOT about code — e.g. a "
        "factual or advice question). The assistant must answer that follow-up "
        "in plain PROSE, not by emitting more code.",
        '{"artifact_request":"<build a small HTML thing>",'
        '"artifact_intro":"<one short sentence>","artifact_html":"<small indented HTML>",'
        '"followup_question":"<an ordinary non-code question>",'
        '"followup_answer":"<a helpful PROSE answer, no code, no tags>"}'),
    "revise": (
        "The user asks the assistant to BUILD a small HTML artifact, then asks "
        "for a concrete CHANGE to it. The assistant returns the UPDATED artifact "
        "reflecting the change.",
        '{"artifact_request":"<build a small HTML thing>",'
        '"artifact_intro":"<one short sentence>","artifact_html":"<small indented HTML>",'
        '"change_request":"<a specific change to make>",'
        '"revise_intro":"<one short sentence>","revised_html":"<the UPDATED indented HTML>"}'),
    "enrich": (
        "The user first asks the assistant to BUILD a small HTML page about some "
        "real-world topic; the assistant builds it from general knowledge. THEN "
        "the user asks the assistant to SEARCH for real, up-to-date information "
        "(sometimes restricted to one official site) and UPDATE the page with it. "
        "The assistant MUST call the search tool (do NOT keep editing blind, do "
        "NOT embed JavaScript), then use the returned results to revise the page. "
        "Provide a realistic web_search/search tool. If the user restricts the "
        "search to a specific site, put a `site:<domain>` filter in the query.",
        '{"tools":[{"name":"web_search","description":"Search the web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}],'
        '"artifact_request":"<build a small HTML page about a real topic>",'
        '"artifact_intro":"<one short sentence>","artifact_html":"<small indented HTML built from general knowledge>",'
        '"enrich_request":"<search for real/updated info (maybe only site X) and update the page>",'
        '"tool_name":"web_search","arguments":{"query":"<search query, with site:domain if restricted>"},'
        '"tool_result":"<realistic JSON string of search results>",'
        '"revised_intro":"<one short sentence noting it used the search results>",'
        '"revised_html":"<the UPDATED indented HTML incorporating the found info>"}'),
    "research": (
        "The user first asks the assistant to BUILD a small HTML page about a "
        "real topic; the assistant builds it. THEN the user tells it to STOP "
        "producing markup and instead SEARCH for real information and just TELL "
        "them the answer in plain prose. VARY the phrasing of this second request "
        "and make it OFTEN TERSE and casual, the way a real user types — e.g. "
        "'No HTML, just the current height. Search for it.', 'No doc, just tell "
        "me — search.', 'Real quick, just search and give me X, no code.', 'Skip "
        "the page, just summarize what you find.'. The assistant MUST call the "
        "search tool, then give a concise PLAIN-PROSE summary from the results — "
        "NOT another HTML/markdown document, NOT a `<p>`, NOT fenced code, NOT "
        "fabricated data. Just sentences.",
        '{"tools":[{"name":"web_search","description":"Search the web","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}],'
        '"artifact_request":"<build a small HTML page about a real topic>",'
        '"artifact_intro":"<one short sentence>","artifact_html":"<small indented HTML>",'
        '"research_request":"<stop making docs; search for real info and just summarize in prose, no HTML>",'
        '"tool_name":"web_search","arguments":{"query":"<search query>"},'
        '"tool_result":"<realistic JSON string of search results>",'
        '"answer":"<a concise PLAIN-PROSE summary using the results; no code, no tags, no markdown headings>"}'),
}


def build_prompt(typ):
    hint, shape = _SHAPES[typ]
    return (f"Invent ONE realistic multi-turn scenario. {hint} "
            f"Draw any tools from realistic domains ({_DOMAINS}); give each a "
            f"proper JSON-Schema for its parameters, with concrete correctly-typed "
            f"argument VALUES. Respond with JSON of exactly this shape: {shape}")


def _tool_call_msg(name, args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function",
                            "function": {"name": name, "arguments": args}}]}


def _tool_result_msg(result):
    return {"role": "tool",
            "content": result if isinstance(result, str) else json.dumps(result)}


def _valid_call(name, args, tools_spec):
    if not name or not isinstance(args, dict):
        return False
    synth = "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"
    return validate_tool_call(synth, tools_spec)["ok"]


def _html_ok(s):
    return isinstance(s, str) and s.strip() and is_valid_html(s, allow_fragment=True)


def assemble(typ, o):
    """Turn a teacher object into (messages, tools_spec) or None if unusable."""
    tools = norm_tools(o.get("tools")) if typ in ("subtask", "recall", "enrich", "research") else None
    tools_spec = ([{"type": "function", "function": t} for t in tools]
                  if tools else None)

    if typ == "subtask":
        if not tools_spec:
            return None
        q = o.get("query"); html = strip_fence(o.get("deliverable_html", ""))
        if not (isinstance(q, str) and q.strip()) or not _html_ok(html):
            return None
        if not _valid_call(o.get("tool_name"), o.get("arguments", {}), tools_spec):
            return None
        reply = as_fenced_reply(o.get("deliverable_intro", ""), html, "html")
        msgs = [{"role": "user", "content": q},
                _tool_call_msg(o["tool_name"], o["arguments"]),
                _tool_result_msg(o.get("tool_result", "")),
                {"role": "assistant", "content": reply}]
        return msgs, tools_spec

    if typ == "recall":
        if not tools_spec:
            return None
        q1, a1, q2, a2 = (o.get("q1"), o.get("a1"), o.get("q2"), o.get("a2"))
        if not all(isinstance(x, str) and x.strip() for x in (q1, a1, q2, a2)):
            return None
        if not _valid_call(o.get("tool1"), o.get("args1", {}), tools_spec):
            return None
        if not _valid_call(o.get("tool2"), o.get("args2", {}), tools_spec):
            return None
        if not (is_prose_reply(a1) and is_prose_reply(a2)):
            return None
        msgs = [{"role": "user", "content": q1},
                _tool_call_msg(o["tool1"], o["args1"]),
                _tool_result_msg(o.get("result1", "")),
                {"role": "assistant", "content": a1},
                {"role": "user", "content": q2},
                _tool_call_msg(o["tool2"], o["args2"]),
                _tool_result_msg(o.get("result2", "")),
                {"role": "assistant", "content": a2}]
        return msgs, tools_spec

    if typ == "modeswitch":
        req = o.get("artifact_request"); html = strip_fence(o.get("artifact_html", ""))
        fq = o.get("followup_question"); fa = o.get("followup_answer")
        if not (isinstance(req, str) and req.strip()) or not _html_ok(html):
            return None
        if not (isinstance(fq, str) and fq.strip()):
            return None
        if not is_prose_reply(fa):   # the whole point: prose, not code, not a call
            return None
        msgs = [{"role": "user", "content": req},
                {"role": "assistant",
                 "content": as_fenced_reply(o.get("artifact_intro", ""), html, "html")},
                {"role": "user", "content": fq},
                {"role": "assistant", "content": fa}]
        return msgs, None

    if typ == "revise":
        req = o.get("artifact_request"); v1 = strip_fence(o.get("artifact_html", ""))
        chg = o.get("change_request"); v2 = strip_fence(o.get("revised_html", ""))
        if not (isinstance(req, str) and req.strip()) or not _html_ok(v1):
            return None
        if not (isinstance(chg, str) and chg.strip()) or not _html_ok(v2):
            return None
        if v1.strip() == v2.strip():   # a revision must actually change something
            return None
        msgs = [{"role": "user", "content": req},
                {"role": "assistant",
                 "content": as_fenced_reply(o.get("artifact_intro", ""), v1, "html")},
                {"role": "user", "content": chg},
                {"role": "assistant",
                 "content": as_fenced_reply(o.get("revise_intro", ""), v2, "html")}]
        return msgs, None

    if typ == "enrich":
        # the v8 field-test failure: artifact ALREADY exists, user asks to search
        # and update -> a <tool_call> must appear BETWEEN the two artifacts
        if not tools_spec:
            return None
        req = o.get("artifact_request"); v1 = strip_fence(o.get("artifact_html", ""))
        enr = o.get("enrich_request"); v2 = strip_fence(o.get("revised_html", ""))
        if not (isinstance(req, str) and req.strip()) or not _html_ok(v1):
            return None
        if not (isinstance(enr, str) and enr.strip()) or not _html_ok(v2):
            return None
        if v1.strip() == v2.strip():          # the update must actually change it
            return None
        if not _valid_call(o.get("tool_name"), o.get("arguments", {}), tools_spec):
            return None
        msgs = [{"role": "user", "content": req},
                {"role": "assistant",
                 "content": as_fenced_reply(o.get("artifact_intro", ""), v1, "html")},
                {"role": "user", "content": enr},
                _tool_call_msg(o["tool_name"], o["arguments"]),
                _tool_result_msg(o.get("tool_result", "")),
                {"role": "assistant",
                 "content": as_fenced_reply(o.get("revised_intro", ""), v2, "html")}]
        return msgs, tools_spec

    if typ == "research":
        # artifact exists, then "stop making docs, search + summarize in prose"
        if not tools_spec:
            return None
        req = o.get("artifact_request"); v1 = strip_fence(o.get("artifact_html", ""))
        rq = o.get("research_request"); ans = o.get("answer")
        if not (isinstance(req, str) and req.strip()) or not _html_ok(v1):
            return None
        if not (isinstance(rq, str) and rq.strip()):
            return None
        if not is_prose_reply(ans):          # final answer must be PROSE, not code
            return None
        if not _valid_call(o.get("tool_name"), o.get("arguments", {}), tools_spec):
            return None
        msgs = [{"role": "user", "content": req},
                {"role": "assistant",
                 "content": as_fenced_reply(o.get("artifact_intro", ""), v1, "html")},
                {"role": "user", "content": rq},
                _tool_call_msg(o["tool_name"], o["arguments"]),
                _tool_result_msg(o.get("tool_result", "")),
                {"role": "assistant", "content": ans}]
        return msgs, tools_spec
    return None


def main():
    # MULTI_ONLY=enrich[,recall...] restricts to a subset of types; MULTI_TARGET
    # overrides the total-line target. Lets v9 append ONLY the new `enrich` shape
    # to the existing multi_raw without regenerating the other types.
    only = os.environ.get("MULTI_ONLY")
    types = [t.strip() for t in only.split(",")] if only else C.MULTI_TYPES
    target = int(os.environ.get("MULTI_TARGET", str(C.TARGET_MULTI)))
    kept = count_lines(C.MULTI_RAW)
    seen = rejected = 0
    print(f"multi: resuming at {kept}/{target} kept; types={types}")
    with open(C.MULTI_RAW, "a") as out:
        while kept < target:
            for typ in types:
                if kept >= target:
                    break
                seen += 1
                try:
                    o = extract_json(ask(build_prompt(typ), SYS))
                    built = assemble(typ, o)
                except Exception:
                    continue
                if not built:
                    rejected += 1
                    continue
                msgs, tools_spec = built
                row = {"messages": msgs, "kind": "multi", "mtype": typ}
                if tools_spec:
                    row["tools_spec"] = tools_spec
                out.write(json.dumps(row) + "\n")
                out.flush(); kept += 1
                if kept % 25 == 0:
                    print(f"multi: kept={kept} seen={seen} rejected={rejected}",
                          flush=True)
    print(f"multi DONE. kept={kept} seen={seen} rejected={rejected} -> {C.MULTI_RAW}")


if __name__ == "__main__":
    main()
