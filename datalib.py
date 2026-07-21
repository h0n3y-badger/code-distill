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
