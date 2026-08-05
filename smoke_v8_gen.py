"""Quick pre-flight for the v8 generators: hit the live 32B teacher a few times
and confirm (a) web rows come back multi-line-formatted + valid, and (b) each
multi-turn type assembles + verifies. Prints a compact pass/fail so we don't
commit 8h of generation to a broken prompt. No files written."""
import json
import genlib, config_uni as C
import gen_web, gen_multi
from datalib import is_valid_html, is_multiline_formatted, has_tool_turn, has_fence

def try_web():
    dom, task, diff = C.WEB_DOMAINS[0], gen_web.FULL_PAGE_TASK, "intermediate"
    obj = genlib.extract_json(genlib.ask(gen_web.build_prompt(dom, task, diff), gen_web.SYS))
    sol = genlib.strip_fence(obj["solution"])
    return {"valid": is_valid_html(sol, allow_fragment=False),
            "multiline": is_multiline_formatted(sol), "len": len(sol)}

def try_multi(typ):
    o = genlib.extract_json(genlib.ask(gen_multi.build_prompt(typ), gen_multi.SYS))
    built = gen_multi.assemble(typ, o)
    if not built:
        return {"assembled": False}
    msgs, spec = built
    return {"assembled": True, "n_msgs": len(msgs),
            "tool_turn": has_tool_turn(msgs),
            "final_fenced": has_fence(msgs[-1]["content"]) if typ in ("subtask","revise") else "-"}

print("== web ==")
for i in range(2):
    try:
        print(" ", try_web())
    except Exception as e:
        print("  ERR", type(e).__name__, str(e)[:80])

for typ in C.MULTI_TYPES:
    print(f"== multi:{typ} ==")
    ok = 0
    for i in range(3):
        try:
            r = try_multi(typ)
            print("  ", r)
            ok += bool(r.get("assembled"))
        except Exception as e:
            print("  ERR", type(e).__name__, str(e)[:80])
    print(f"  -> {ok}/3 assembled")
