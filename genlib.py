"""Shared teacher-call helpers for the universal generators (gen_web / gen_chat /
gen_tools). Thin wrapper over the OpenAI-compatible llama-server endpoint plus
the JSON-extraction hardening the coding pipeline already learned it needs."""
import json, re
from openai import OpenAI
import config_uni as C

client = OpenAI(base_url=C.TEACHER_BASE_URL, api_key=C.TEACHER_API_KEY)


def ask(user, system, temp=None, max_tokens=1536):
    r = client.chat.completions.create(
        model=C.TEACHER_MODEL,
        temperature=C.GEN_TEMPERATURE if temp is None else temp,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}])
    return r.choices[0].message.content


def extract_json(text):
    """Dig a single JSON object out of a possibly-fenced/chatty reply."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
    return json.loads(text)


def strip_fence(code):
    """Strip a ```lang ... ``` fence wrapping a JSON string value, if present."""
    if not isinstance(code, str):
        return code
    m = re.search(r"^\s*```[a-zA-Z0-9+#.\-]*\s*\n(.*?)\n?```\s*$", code.strip(), re.S)
    return m.group(1) if m else code


def count_lines(path):
    try:
        with open(path) as f:
            return sum(1 for ln in f if ln.strip())
    except FileNotFoundError:
        return 0
