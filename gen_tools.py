"""Generate tool / MCP function-calling training data from the 32B teacher.

For each scenario the teacher invents a small, realistic toolset (name +
description + JSON-Schema params — exactly what an MCP server surfaces to a
model) and a user query, then says which tool to call with which arguments (or
that NO tool is needed). WE assemble the canonical Qwen tool-call turn from that
and VERIFY it: the emitted call must name a declared tool and satisfy its schema
(datalib.validate_tool_call — the same gate the eval uses). "No tool needed"
scenarios are kept too, to teach the model restraint (don't over-call tools).

Rows carry a `tools_spec` so train.py renders them with the tokenizer's tool
template (system <tools> block + <tool_call>/<tool_response>), matching inference.
Appends to TOOLS_RAW; ctrl-C safe; stops at TARGET_TOOLS.
"""
import json
import config_uni as C
from genlib import ask, extract_json, count_lines
from datalib import validate_tool_call

SYS = ("You design realistic tool-use (function-calling / MCP) training "
       "scenarios. Return ONLY one JSON object, no prose, no markdown fences.")

SCHEMA = """{
 "tools": [ {"name": "snake_case_name", "description": "what it does",
   "parameters": {"type":"object","properties":{"argname":{"type":"string|integer|number|boolean|array|object","description":"..."}},"required":["argname"]}} ],
 "query": "<the user's request>",
 "needs_tool": true,
 "tool_name": "<name of the ONE tool to call, if needs_tool>",
 "arguments": { "argname": "value matching the schema type" },
 "tool_result": "<a realistic JSON string the tool would return>",
 "answer": "<the assistant's final natural-language answer to the user, using the tool result>"
}"""


def build_prompt(diff):
    no_tool = "no tool" in diff
    hint = {
        "single obvious tool": "Provide 1-2 tools; exactly one clearly applies.",
        "pick among several tools": "Provide 3-4 plausible tools; the query must "
            "make exactly one the correct choice (the others are near-misses).",
        "multi-step (call, read result, answer)": "Provide 2-3 tools; the query "
            "requires calling one, then USING its returned result to answer.",
        "user query needs no tool (must answer directly)": "Provide 2-3 tools, "
            "but craft a query that none of them can help with, so the assistant "
            "should just answer directly. Set needs_tool=false and OMIT "
            "tool_name/arguments/tool_result.",
    }[diff]
    domains = ("weather, web search, email/calendar, file system, databases, "
               "GitHub/git, HTTP APIs, math/units, home automation, maps")
    return (f"Invent ONE tool-use scenario. {hint} "
            f"Draw tools from realistic domains ({domains}); give each a proper "
            f"JSON-Schema for its parameters. Make argument VALUES concrete and "
            f"correctly typed. {'Set needs_tool=false. ' if no_tool else ''}"
            f"Respond with JSON of exactly this shape: {SCHEMA}")


def norm_tools(tools):
    """Coerce teacher tools into [{name, description, parameters}] with a valid
    object schema; return None if unusable."""
    out = []
    if not isinstance(tools, list) or not tools:
        return None
    for t in tools:
        if not isinstance(t, dict) or not t.get("name"):
            return None
        params = t.get("parameters")
        if not isinstance(params, dict) or params.get("type") != "object":
            params = {"type": "object", "properties": {}}
        out.append({"name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": params})
    return out


def main():
    kept = count_lines(C.TOOLS_RAW)
    seen = rejected = 0
    print(f"tools: resuming at {kept}/{C.TARGET_TOOLS} kept")
    with open(C.TOOLS_RAW, "a") as out:
        while kept < C.TARGET_TOOLS:
            for diff in C.TOOL_DIFFICULTIES:
                if kept >= C.TARGET_TOOLS:
                    break
                seen += 1
                try:
                    o = extract_json(ask(build_prompt(diff), SYS))
                    tools = norm_tools(o.get("tools"))
                    query = o["query"]; answer = o["answer"]
                except Exception:
                    continue
                if not tools or not isinstance(query, str) or not query.strip():
                    continue
                if not isinstance(answer, str) or not answer.strip():
                    continue
                tools_spec = [{"type": "function", "function": t} for t in tools]
                needs = bool(o.get("needs_tool", True))

                if needs:
                    name = o.get("tool_name"); args = o.get("arguments", {})
                    if not name or not isinstance(args, dict):
                        rejected += 1; continue
                    # verify the assembled call against the declared schema
                    synth = ('<tool_call>' +
                             json.dumps({"name": name, "arguments": args}) +
                             '</tool_call>')
                    if not validate_tool_call(synth, tools_spec)["ok"]:
                        rejected += 1; continue
                    result = o.get("tool_result", "")
                    if not isinstance(result, str):
                        result = json.dumps(result)
                    messages = [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function",
                                         "function": {"name": name, "arguments": args}}]},
                        {"role": "tool", "content": result},
                        {"role": "assistant", "content": answer},
                    ]
                else:
                    # restraint case: must NOT emit a tool_call
                    if "<tool_call>" in answer:
                        rejected += 1; continue
                    messages = [
                        {"role": "user", "content": query},
                        {"role": "assistant", "content": answer},
                    ]

                out.write(json.dumps({
                    "messages": messages, "tools_spec": tools_spec,
                    "kind": "tools", "needs_tool": needs,
                    "difficulty": diff}) + "\n")
                out.flush(); kept += 1
                if kept % 25 == 0:
                    print(f"tools: kept={kept} seen={seen} rejected={rejected}",
                          flush=True)
    print(f"tools DONE. kept={kept} seen={seen} rejected={rejected} -> {C.TOOLS_RAW}")


if __name__ == "__main__":
    main()
