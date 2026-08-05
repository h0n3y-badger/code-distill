"""Pure, dependency-free data helpers shared by the pipeline and the test suite.

Kept import-light on purpose: train.py pulls unsloth/torch (heavy, GPU), so the
logic worth regression-testing (row validity, C header-completeness) lives here
where tests can import it in milliseconds.
"""
import json
import re


def is_valid_messages(msgs):
    """A usable turn pair for SFT: a list of >=2 messages, each a dict whose
    `content` is a non-empty string. (~5% of raw generations had a null/empty
    assistant reply — the chat template can't concatenate None, and an empty
    target teaches nothing.) This is the exact gate train.py applies."""
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False
    return all(isinstance(m, dict) and isinstance(m.get("content"), str)
               and m["content"].strip() for m in msgs)


def iter_jsonl(path):
    """Yield parsed objects from a JSONL file, skipping blank lines."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_valid_texts(path, render):
    """Load `path`, keep rows whose messages pass is_valid_messages, and render
    each to a flat string via `render(messages)`. Returns (texts, n_skipped).
    `render` is injected so this stays tokenizer-free (train.py passes the chat
    template)."""
    texts, skipped = [], 0
    for obj in iter_jsonl(path):
        msgs = obj.get("messages")
        if not is_valid_messages(msgs):
            skipped += 1
            continue
        texts.append(render(msgs))
    return texts, skipped


# --- C header-completeness lint ------------------------------------------------
# symbol (word-boundary) -> the header that must be #include'd to use it.
# Deliberately focused on the headers diag_c.py flagged as the residual C
# failure mode (limits/ctype/math/stdbool/stdint), plus the common ones.
_SYMBOL_HEADERS = [
    (r"\bINT_MAX\b|\bINT_MIN\b|\bUINT_MAX\b|\bLONG_MAX\b|\bLONG_MIN\b|\bCHAR_BIT\b",
     "limits.h"),
    (r"\bisalpha\b|\bisdigit\b|\bisalnum\b|\bisspace\b|\bisupper\b|\bislower\b|"
     r"\btoupper\b|\btolower\b|\bispunct\b", "ctype.h"),
    (r"\bsqrt\b|\bpow\b|\bfabs\b|\bfloor\b|\bceil\b|\bround\b|\blog\b|\bsin\b|"
     r"\bcos\b|\btan\b|\bexp\b", "math.h"),
    (r"\bbool\b|\btrue\b|\bfalse\b", "stdbool.h"),
    (r"\buint8_t\b|\buint16_t\b|\buint32_t\b|\buint64_t\b|\bint8_t\b|\bint16_t\b|"
     r"\bint32_t\b|\bint64_t\b", "stdint.h"),
    (r"\bmalloc\b|\bcalloc\b|\brealloc\b|\bfree\b|\bstrtol\b|\bstrtod\b|\bqsort\b",
     "stdlib.h"),
    (r"\bstrlen\b|\bstrcpy\b|\bstrncpy\b|\bstrcmp\b|\bstrncmp\b|\bstrcat\b|"
     r"\bmemcpy\b|\bmemset\b|\bmemmove\b|\bstrchr\b|\bstrstr\b", "string.h"),
]


def _strip_c_comments(src):
    if not isinstance(src, str):
        return ""
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    src = re.sub(r"//[^\n]*", " ", src)
    return src


def included_headers(src):
    """Set of headers the source #includes (angle-bracket form)."""
    return set(re.findall(r"#include\s*<([^>]+)>", src))


def missing_headers(src):
    """Return the set of headers this C source USES symbols from but does not
    #include. Empty set == self-contained w.r.t. the tracked symbols.
    (A `bool`-typed identifier defined via typedef wouldn't trip this, but the
    generated data never does that — the check matches how the model writes.)"""
    if not isinstance(src, str):
        return set()
    code = _strip_c_comments(src)
    have = included_headers(src)
    missing = set()
    for pattern, header in _SYMBOL_HEADERS:
        if header in have:
            continue
        if re.search(pattern, code):
            missing.add(header)
    return missing


# --- HTML / web validity ------------------------------------------------------
# Web output can't be "executed" like Python/C, but a large fraction of a small
# model's web failures ARE mechanical and machine-checkable: unclosed tags,
# mismatched nesting, missing <!doctype>, a <form> with no labels, etc. We lint
# for those the same spirit as the C header check — cheap, deterministic, and a
# real signal the model's markup is structurally sound. Uses only the stdlib.
from html.parser import HTMLParser

# Elements that never have a closing tag (HTML5 void elements).
_VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
}
# Elements whose close tag is optional / auto-closed by the parser — don't
# demand a matching close for these (matches how browsers/html.parser behave).
_OPTIONAL_CLOSE = {
    "li", "dt", "dd", "p", "option", "thead", "tbody", "tfoot", "tr", "td",
    "th", "colgroup", "html", "head", "body",
}


class _BalanceParser(HTMLParser):
    """Track tag nesting to detect unbalanced/misnested markup."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.errors = []
        self.tags = set()

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        if tag not in _VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags.add(tag)  # self-closing <foo/> — nothing to push

    def handle_endtag(self, tag):
        if tag in _VOID_TAGS:
            return
        if tag in self.stack:
            # pop down to the matching open; anything above it was left unclosed
            while self.stack and self.stack[-1] != tag:
                top = self.stack.pop()
                if top not in _OPTIONAL_CLOSE:
                    self.errors.append(f"unclosed <{top}> before </{tag}>")
            if self.stack:
                self.stack.pop()
        else:
            self.errors.append(f"stray </{tag}> (no matching open)")


def html_issues(src):
    """Return a list of structural problems in an HTML string. Empty == clean.
    Deterministic, stdlib-only. Not a full validator — targets the mechanical
    failure modes a small model actually produces."""
    if not isinstance(src, str) or not src.strip():
        return ["empty"]
    p = _BalanceParser()
    try:
        p.feed(src)
        p.close()
    except Exception as e:  # malformed enough to crash the lenient parser
        return [f"parse error: {e}"]
    issues = list(p.errors)
    leftover = [t for t in p.stack if t not in _OPTIONAL_CLOSE]
    if leftover:
        issues.append("unclosed at EOF: " + ", ".join(f"<{t}>" for t in leftover))
    # a11y / quality nudges that are cheap and objective
    if "<form" in src.lower() and "<label" not in src.lower():
        issues.append("form without any <label>")
    if "<img" in src.lower() and not re.search(r"<img[^>]*\balt\s*=", src, re.I):
        issues.append("<img> without alt=")
    return issues


def is_valid_html(src, allow_fragment=True):
    """True if the HTML is structurally sound. With allow_fragment=False, also
    require a <!doctype> and an <html> root (a full document)."""
    issues = html_issues(src)
    hard = [i for i in issues
            if i.startswith(("unclosed", "stray", "parse error", "empty"))]
    if hard:
        return False
    if not allow_fragment:
        low = src.lower()
        if "<!doctype" not in low or "<html" not in low:
            return False
    return True


# --- Tool / MCP function-call validity ----------------------------------------
# We train tool use in the Qwen/Hermes convention: the assistant emits one or
# more   <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>   blocks.
# That's exactly what MCP tools surface as to the model (name + JSON-Schema
# params), so verifying this format IS verifying MCP-readiness. Below is a
# light JSON-Schema arg check — enough to reject the real failure modes
# (bad JSON, unknown tool, missing required arg, wrong scalar type).
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

_JSON_TYPE_OK = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def extract_tool_calls(text):
    """Parse every <tool_call>{...}</tool_call> block. Returns (calls, errors)
    where calls is a list of dicts and errors lists malformed blocks."""
    calls, errors = [], []
    if not isinstance(text, str):
        return calls, ["not a string"]
    for raw in _TOOL_CALL_RE.findall(text):
        try:
            obj = json.loads(raw)
        except Exception as e:
            errors.append(f"invalid JSON in tool_call: {e}")
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            errors.append("tool_call missing 'name'")
            continue
        obj.setdefault("arguments", {})
        if not isinstance(obj["arguments"], dict):
            errors.append(f"'arguments' of {obj.get('name')} is not an object")
            continue
        calls.append(obj)
    return calls, errors


def _tool_index(tools):
    """Map tool name -> its JSON-Schema `parameters`, from an OpenAI/Qwen-style
    tools list ([{"type":"function","function":{"name","parameters"}}] or the
    flat [{"name","parameters"}] form)."""
    idx = {}
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        name = fn.get("name")
        if name:
            idx[name] = fn.get("parameters", {}) or {}
    return idx


# --- Presentation: code fences, formatting, tool-turn awareness (v8) ----------
# The v6 model emitted code as bare text (no ``` fence) and sometimes as a single
# minified line. Renderers (LM Studio) only draw a "code box" for a fenced block,
# and a formatted, multi-line artifact reads far better. These helpers let the
# generators/mix normalize artifact turns into proper assistant messages, and let
# the eval MEASURE the fence/format behavior instead of silently stripping it.
_FENCE_RE = re.compile(r"```[a-zA-Z0-9+#.\-]*\n(.*?)```", re.S)


def has_fence(text):
    """True if the text contains a markdown fenced code block."""
    return isinstance(text, str) and bool(_FENCE_RE.search(text))


def extract_fenced(text):
    """Return the content of the first fenced block, or None if unfenced."""
    if not isinstance(text, str):
        return None
    m = _FENCE_RE.search(text)
    return m.group(1) if m else None


def is_multiline_formatted(code, min_tags=4):
    """Reject minified one-liners. Real, readable HTML/CSS/JS is broken across
    lines and indented. Only enforced once there's enough structure to warrant
    it (>= min_tags tags), so a legitimately tiny fragment isn't punished."""
    if not isinstance(code, str) or not code.strip():
        return False
    n_tags = code.count("<")
    if n_tags < min_tags:
        return True  # too small to demand multi-line
    if code.count("\n") < 3:
        return False  # essentially one line
    # at least one indented line (leading space/tab) => actually formatted
    return any(ln[:1] in (" ", "\t") for ln in code.splitlines())


def as_fenced_reply(intro, code, lang="html"):
    """Build a proper assistant turn: a short prose intro, then the code in a
    fenced block. Idempotent — if `code` is already a full fenced reply, return
    it unchanged (avoids double-fencing already-normalized rows)."""
    if has_fence(code) or has_fence(intro):
        return code if has_fence(code) else intro
    intro = (intro or "").strip()
    lead = (intro + "\n\n") if intro else ""
    return f"{lead}```{lang}\n{code.strip()}\n```"


def has_tool_turn(msgs):
    """True if any assistant message carries a structured tool call (the
    canonical OpenAI/Qwen `tool_calls` field). Such rows must skip the plain
    is_valid_messages gate (the tool-call turn has empty content by design)."""
    if not isinstance(msgs, list):
        return False
    return any(isinstance(m, dict) and m.get("role") == "assistant"
               and m.get("tool_calls") for m in msgs)


def is_prose_reply(text, min_len=25):
    """A plain conversational answer: substantive text with NO code fence and NO
    tool_call. This is exactly what a good model should return when, after
    producing an artifact, it's asked an ordinary question (the v6 'mode-lock'
    failure was dumping code here instead)."""
    if not isinstance(text, str) or len(text.strip()) < min_len:
        return False
    return not has_fence(text) and "<tool_call>" not in text


def validate_tool_call(text, tools=None, require_call=True):
    """Validate the tool_call(s) in `text`. If `tools` is given, also check each
    call names a declared tool and its arguments satisfy that tool's JSON-Schema
    (required keys present, scalar types match). Returns
    {ok, calls, errors}."""
    calls, errors = extract_tool_calls(text)
    if require_call and not calls:
        errors.append("no <tool_call> found")
    if tools is not None:
        idx = _tool_index(tools)
        for c in calls:
            name = c["name"]
            if name not in idx:
                errors.append(f"unknown tool '{name}'")
                continue
            schema = idx[name]
            props = schema.get("properties", {}) or {}
            for req in schema.get("required", []) or []:
                if req not in c["arguments"]:
                    errors.append(f"{name}: missing required arg '{req}'")
            for k, v in c["arguments"].items():
                spec = props.get(k)
                if not spec:
                    continue  # extra args tolerated (open schemas are common)
                t = spec.get("type")
                if t in _JSON_TYPE_OK and not _JSON_TYPE_OK[t](v):
                    errors.append(f"{name}.{k}: expected {t}")
    return {"ok": not errors, "calls": calls, "errors": errors}
