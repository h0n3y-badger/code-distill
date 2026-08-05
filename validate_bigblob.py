"""FAITHFUL validation of the big-blob prose behavior, run against LM Studio's OWN
runtime (:1234) — the environment the user actually uses, NOT the :8091 llama.cpp
harness that hid this bug twice. Replays the exact v11 field failure: a FRESH chat,
"give me the flight 13 launch profile", the model's searxng tool call, and the REAL
11KB 28-result blob from the field transcript. Samples N times at temp 0.8 (LM
Studio GUI default) and reports the HTML-doc failure RATE.

  MODEL_ID=...gguf N=60 python validate_bigblob.py     # hits http://localhost:1234
Baseline: v11 = ~15% @temp0.8 / ~10% @temp0.4. Goal: v12 << that.
"""
import json, re, os
from openai import OpenAI

URL = os.environ.get("LMS_URL", "http://localhost:1234/v1")
MODEL = os.environ["MODEL_ID"]
N = int(os.environ.get("N", "60"))
TEMP = float(os.environ.get("TEMP", "0.8"))
c = OpenAI(base_url=URL, api_key="x")

RAW = re.compile(r"<(?:details|summary|p|div|ul|li|h[1-6]|table|section|strong|em|"
                 r"!doctype|html|body|blockquote)\b", re.I)
def is_bad(t):  # HTML/markdown DOCUMENT instead of prose
    return ("```" in (t or "")) or bool(RAW.search(t or ""))

md = open("/home/honeybadger/Documents/v11-test-1.md").read() \
        .replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
BLOB = re.search(r"### Tool\nTool call result:\n\n```\n(.*?)\n```", md, re.S).group(1)
TOOLS = [{"type": "function", "function": {
    "name": "searxng_web_search", "description": "Search the web using SearXNG",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}, "time_range": {"type": "string"},
        "language": {"type": "string"}}, "required": ["query"]}}}]
HIST = [
    {"role": "user", "content": "Are you able to search? Real quick give me the flight 13 launch profile."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
        "function": {"name": "searxng_web_search", "arguments": json.dumps({"query": "flight 13 launch profile"})}}]},
    {"role": "tool", "tool_call_id": "c1", "content": BLOB},
]


def main():
    bad = 0
    for i in range(N):
        r = c.chat.completions.create(model=MODEL, messages=HIST, tools=TOOLS,
                                      temperature=TEMP).choices[0].message.content or ""
        bad += is_bad(r)
    print(f"RATE {bad}/{N} = {bad/N:.0%} HTML-doc  (model={MODEL.split('/')[-1]} temp={TEMP})")


if __name__ == "__main__":
    main()
