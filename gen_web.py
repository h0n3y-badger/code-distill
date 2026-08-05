"""Generate web-design (HTML/CSS/JS) training data from the 32B teacher.

Verification is structural, not execution: every kept sample's markup must pass
datalib.is_valid_html (balanced tags, no stray closes; full-page tasks must also
carry a <!doctype> + <html> root). That rejects the mechanical failure mode a
small model actually produces — unclosed / misnested tags — the same spirit as
the C-header gate. Appends to WEB_RAW; ctrl-C safe; stops at TARGET_WEB.
"""
import json, itertools
import config_uni as C
from genlib import ask, extract_json, strip_fence, count_lines
from datalib import is_valid_html, html_issues, is_multiline_formatted

SYS = ("You are a senior front-end engineer creating high-quality web-design "
       "training data. Return ONLY one JSON object, no prose, no markdown fences. "
       "Put RAW HTML/CSS/JS in the JSON string values (no ``` fences inside). "
       "The markup MUST be cleanly formatted: one element per line, consistently "
       "indented (2 spaces per level) — never minified onto a single line.")

FULL_PAGE_TASK = "build a complete, self-contained HTML page for"


def build_prompt(dom, task, diff):
    full = task == FULL_PAGE_TASK
    if full:
        sol_desc = ("<a COMPLETE, self-contained HTML5 document: <!doctype html>, "
                    "<html>, <head> (with <title> and a <style> block for CSS), "
                    "and <body>. Inline all CSS/JS — no external files. Every tag "
                    "correctly closed. Semantic, accessible markup. Any <img> needs "
                    "alt=; any <form> needs <label>s.>")
    else:
        sol_desc = ("<the HTML (plus inline <style>/<script> as needed) that solves "
                    "the task. A fragment is fine — no <!doctype> needed — but every "
                    "tag must be correctly closed and nested. Accessible: <img> has "
                    "alt=, <form> has <label>s.>")
    schema = ('{"instruction": "<the full task, including any starter markup the '
              'user must work with>", "solution": "' + sol_desc + '"}')
    return (f"Invent ONE {diff} web-design task in the domain of {dom}. "
            f"The task is to: '{task}'. Make it realistic and specific (a named, "
            f"concrete thing to build/fix), not a textbook cliché. "
            f"Respond with JSON: {schema}")


def main():
    grid = [(d, t, diff) for d in C.WEB_DOMAINS
            for t in C.WEB_TASKS for diff in C.WEB_DIFFICULTIES]
    kept = count_lines(C.WEB_RAW)
    seen = rejected = 0
    print(f"web: resuming at {kept}/{C.TARGET_WEB} kept")
    with open(C.WEB_RAW, "a") as out:
        for _pass in range(C.PASSES):
            for dom, task, diff in grid:
                if kept >= C.TARGET_WEB:
                    print(f"web: reached target {C.TARGET_WEB}"); return
                seen += 1
                try:
                    obj = extract_json(ask(build_prompt(dom, task, diff), SYS))
                    instr = obj["instruction"]
                    sol = strip_fence(obj["solution"])
                except Exception:
                    continue
                if not (isinstance(sol, str) and sol.strip()):
                    continue
                full = task == FULL_PAGE_TASK
                if not is_valid_html(sol, allow_fragment=not full):
                    rejected += 1
                    continue
                # v8: reject minified one-liners so the student learns to emit
                # readable, indented markup (the field-test 'FORMAT THE CODE' bug)
                if not is_multiline_formatted(sol):
                    rejected += 1
                    continue
                out.write(json.dumps({
                    "messages": [{"role": "user", "content": instr},
                                 {"role": "assistant", "content": sol}],
                    "kind": "web", "lang": "HTML", "domain": dom,
                    "task": task, "difficulty": diff,
                    "issues": html_issues(sol)}) + "\n")
                out.flush(); kept += 1
                if kept % 25 == 0:
                    print(f"web: kept={kept} seen={seen} rejected={rejected}",
                          flush=True)
    print(f"web DONE. kept={kept} seen={seen} rejected={rejected} -> {C.WEB_RAW}")


if __name__ == "__main__":
    main()
