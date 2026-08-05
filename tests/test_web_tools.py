"""Regression tests for the universal-model additions: HTML validity lint,
tool/MCP call validation, and mix.py's kind-aware row gate. Pure/fast — no model,
no network. Run: python -m unittest discover tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datalib
import mix


class TestHtmlValidity(unittest.TestCase):
    def test_balanced_doc(self):
        src = ("<!doctype html><html><head><title>t</title></head>"
               "<body><h1>Hi</h1></body></html>")
        self.assertEqual(datalib.html_issues(src), [])
        self.assertTrue(datalib.is_valid_html(src, allow_fragment=False))

    def test_stray_close(self):
        self.assertIn("stray </span> (no matching open)",
                      datalib.html_issues("<div><p>hi</span></div>"))
        self.assertFalse(datalib.is_valid_html("<div><p>hi</span></div>"))

    def test_unclosed_at_eof(self):
        issues = datalib.html_issues("<section><h2>t</h2>")
        self.assertTrue(any("unclosed at EOF" in i for i in issues))
        self.assertFalse(datalib.is_valid_html("<section><h2>t</h2>"))

    def test_void_tags_ok(self):
        # void elements need no close tag
        self.assertEqual(datalib.html_issues("<p>a<br>b<img src=x alt=y></p>"), [])

    def test_optional_close_ok(self):
        # <li>/<p> auto-close — a <ul> of unclosed <li> is not a hard error
        self.assertTrue(datalib.is_valid_html("<ul><li>a<li>b</ul>"))

    def test_fragment_not_a_doc(self):
        self.assertTrue(datalib.is_valid_html("<section><p>x</p></section>"))
        self.assertFalse(datalib.is_valid_html("<section><p>x</p></section>",
                                               allow_fragment=False))

    def test_img_without_alt_flagged(self):
        self.assertIn("<img> without alt=", datalib.html_issues("<p><img src=x></p>"))

    def test_form_without_label_flagged(self):
        self.assertIn("form without any <label>",
                      datalib.html_issues("<form><input name=x></form>"))

    def test_empty(self):
        self.assertFalse(datalib.is_valid_html(""))
        self.assertFalse(datalib.is_valid_html(None))


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "weather",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"},
                                  "days": {"type": "integer"}},
                   "required": ["city"]}}}]


class TestToolCall(unittest.TestCase):
    def test_valid(self):
        t = '<tool_call>{"name":"get_weather","arguments":{"city":"Paris","days":3}}</tool_call>'
        self.assertTrue(datalib.validate_tool_call(t, TOOLS)["ok"])

    def test_missing_required(self):
        t = '<tool_call>{"name":"get_weather","arguments":{"days":3}}</tool_call>'
        r = datalib.validate_tool_call(t, TOOLS)
        self.assertFalse(r["ok"])
        self.assertTrue(any("missing required" in e for e in r["errors"]))

    def test_unknown_tool(self):
        t = '<tool_call>{"name":"nope","arguments":{}}</tool_call>'
        self.assertFalse(datalib.validate_tool_call(t, TOOLS)["ok"])

    def test_wrong_type(self):
        t = '<tool_call>{"name":"get_weather","arguments":{"city":"Paris","days":"lots"}}</tool_call>'
        r = datalib.validate_tool_call(t, TOOLS)
        self.assertFalse(r["ok"])

    def test_bad_json(self):
        t = '<tool_call>{"name":"get_weather", bogus}</tool_call>'
        self.assertFalse(datalib.validate_tool_call(t, TOOLS)["ok"])

    def test_no_call_when_required(self):
        self.assertFalse(datalib.validate_tool_call("just text", TOOLS)["ok"])

    def test_no_call_allowed(self):
        # restraint case: no tool expected -> ok when require_call=False
        r = datalib.validate_tool_call("A haiku about autumn.", TOOLS,
                                       require_call=False)
        self.assertTrue(r["ok"])

    def test_multiple_calls_extracted(self):
        t = ('<tool_call>{"name":"a","arguments":{}}</tool_call>'
             '<tool_call>{"name":"b","arguments":{}}</tool_call>')
        calls, errs = datalib.extract_tool_calls(t)
        self.assertEqual([c["name"] for c in calls], ["a", "b"])


class TestMixRowGate(unittest.TestCase):
    def test_chat_row_ok(self):
        self.assertTrue(mix.row_ok({"kind": "chat", "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there, how can I help?"}]}))

    def test_tool_row_ok_despite_empty_call_turn(self):
        # the assistant tool-call turn has empty content BY DESIGN; the row must
        # still pass because it ends in a real text answer
        self.assertTrue(mix.row_ok({"kind": "tools", "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"type": "function",
                             "function": {"name": "get_weather", "arguments": {"city": "X"}}}]},
            {"role": "tool", "content": "{}"},
            {"role": "assistant", "content": "It's sunny."}]}))

    def test_tool_row_needs_final_answer(self):
        self.assertFalse(mix.row_ok({"kind": "tools", "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": ""}]}))

    def test_bad_chat_row_rejected(self):
        self.assertFalse(mix.row_ok({"kind": "chat", "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None}]}))


if __name__ == "__main__":
    unittest.main()
