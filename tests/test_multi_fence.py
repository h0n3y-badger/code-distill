"""Regression tests for the v8 additions: code-fence / formatting helpers,
tool-turn awareness, mix.py fence-normalization, and gen_multi.assemble's
multi-turn mode-switch construction + verification. Pure/fast — no model, no
network. Run: python -m unittest discover tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datalib
import mix
import gen_multi


class TestFenceHelpers(unittest.TestCase):
    def test_has_and_extract_fence(self):
        txt = "Here it is:\n```html\n<p>hi</p>\n```"
        self.assertTrue(datalib.has_fence(txt))
        self.assertEqual(datalib.extract_fenced(txt).strip(), "<p>hi</p>")

    def test_no_fence(self):
        self.assertFalse(datalib.has_fence("<p>hi</p>"))
        self.assertIsNone(datalib.extract_fenced("<p>hi</p>"))

    def test_as_fenced_reply_wraps(self):
        r = datalib.as_fenced_reply("Here's the HTML:", "<div>\n  <p>x</p>\n</div>", "html")
        self.assertTrue(datalib.has_fence(r))
        self.assertTrue(r.startswith("Here's the HTML:"))
        self.assertIn("```html", r)

    def test_as_fenced_reply_idempotent(self):
        already = "Intro\n```html\n<p>x</p>\n```"
        self.assertEqual(datalib.as_fenced_reply("", already, "html"), already)


class TestFormatGate(unittest.TestCase):
    def test_minified_rejected(self):
        minified = ('<!doctype html><html><head><title>t</title></head>'
                    '<body><h1>Hi</h1><p>x</p></body></html>')
        self.assertFalse(datalib.is_multiline_formatted(minified))

    def test_formatted_accepted(self):
        pretty = ("<!doctype html>\n<html>\n  <head>\n    <title>t</title>\n"
                  "  </head>\n  <body>\n    <h1>Hi</h1>\n  </body>\n</html>")
        self.assertTrue(datalib.is_multiline_formatted(pretty))

    def test_tiny_fragment_exempt(self):
        self.assertTrue(datalib.is_multiline_formatted("<br>"))


class TestToolTurnAndProse(unittest.TestCase):
    def test_has_tool_turn(self):
        msgs = [{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"type": "function",
                                 "function": {"name": "f", "arguments": {}}}]}]
        self.assertTrue(datalib.has_tool_turn(msgs))
        self.assertFalse(datalib.has_tool_turn(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "hello there friend"}]))

    def test_is_prose_reply(self):
        self.assertTrue(datalib.is_prose_reply("The capital of Japan is Tokyo."))
        self.assertFalse(datalib.is_prose_reply("```html\n<p>x</p>\n```"))
        self.assertFalse(datalib.is_prose_reply(
            '<tool_call>{"name":"f","arguments":{}}</tool_call>'))
        self.assertFalse(datalib.is_prose_reply("ok"))  # too short


class TestMixNormalize(unittest.TestCase):
    def test_web_row_gets_fenced(self):
        row = {"kind": "web", "lang": "HTML", "messages": [
            {"role": "user", "content": "build a page"},
            {"role": "assistant", "content": "<!doctype html>\n<html></html>"}]}
        out = mix.normalize_row(row)
        self.assertTrue(datalib.has_fence(out["messages"][1]["content"]))
        self.assertIn("```html", out["messages"][1]["content"])

    def test_replay_code_gets_fenced(self):
        row = {"lang": "C", "messages": [
            {"role": "user", "content": "write a function"},
            {"role": "assistant", "content": "int f(void){\n  return 0;\n}"}]}
        out = mix.normalize_row(row)
        self.assertIn("```c", out["messages"][1]["content"])

    def test_chat_row_untouched(self):
        row = {"kind": "chat", "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello, how can I help today?"}]}
        out = mix.normalize_row(row)
        self.assertNotIn("```", out["messages"][1]["content"])

    def test_multi_row_untouched(self):
        # already fenced by the generator; must not be double-fenced
        content = "Here:\n```html\n<div>\n  <p>x</p>\n</div>\n```"
        row = {"kind": "multi", "mtype": "modeswitch", "messages": [
            {"role": "user", "content": "build"},
            {"role": "assistant", "content": content},
            {"role": "user", "content": "what's 2+2?"},
            {"role": "assistant", "content": "It equals four."}]}
        out = mix.normalize_row(row)
        self.assertEqual(out["messages"][1]["content"], content)
        self.assertEqual(out["messages"][1]["content"].count("```html"), 1)


# --- gen_multi.assemble: construct + verify each mode-switch type -------------
_TOOLS = [{"name": "web_search", "description": "search",
           "parameters": {"type": "object",
                          "properties": {"query": {"type": "string"}},
                          "required": ["query"]}}]
_HTML = "<div>\n  <p>hello</p>\n</div>"


class TestGenMultiAssemble(unittest.TestCase):
    def test_modeswitch_ok(self):
        o = {"artifact_request": "build a card", "artifact_intro": "Here",
             "artifact_html": _HTML,
             "followup_question": "what's the capital of Japan?",
             "followup_answer": "The capital of Japan is Tokyo."}
        msgs, spec = gen_multi.assemble("modeswitch", o)
        self.assertIsNone(spec)
        self.assertEqual(len(msgs), 4)
        self.assertTrue(datalib.has_fence(msgs[1]["content"]))     # artifact fenced
        self.assertTrue(datalib.is_prose_reply(msgs[3]["content"]))  # prose answer

    def test_modeswitch_rejects_code_followup(self):
        o = {"artifact_request": "build a card", "artifact_intro": "Here",
             "artifact_html": _HTML, "followup_question": "and a button?",
             "followup_answer": "```html\n<button>x</button>\n```"}
        self.assertIsNone(gen_multi.assemble("modeswitch", o))

    def test_subtask_ok_and_tool_call(self):
        o = {"tools": _TOOLS, "query": "search canada pop and build html",
             "tool_name": "web_search", "arguments": {"query": "canada population"},
             "tool_result": "{\"pop\": 40000000}", "deliverable_intro": "Here",
             "deliverable_html": _HTML}
        msgs, spec = gen_multi.assemble("subtask", o)
        self.assertIsNotNone(spec)
        self.assertTrue(datalib.has_tool_turn(msgs))
        self.assertEqual(msgs[1]["tool_calls"][0]["function"]["name"], "web_search")
        self.assertTrue(datalib.has_fence(msgs[-1]["content"]))

    def test_subtask_rejects_bad_tool(self):
        o = {"tools": _TOOLS, "query": "q", "tool_name": "not_a_tool",
             "arguments": {"query": "x"}, "tool_result": "{}",
             "deliverable_intro": "Here", "deliverable_html": _HTML}
        self.assertIsNone(gen_multi.assemble("subtask", o))

    def test_recall_two_calls(self):
        o = {"tools": _TOOLS,
             "q1": "search a", "tool1": "web_search", "args1": {"query": "a"},
             "result1": "{}", "a1": "Here is what I found about a, quite useful.",
             "q2": "now search b", "tool2": "web_search", "args2": {"query": "b"},
             "result2": "{}", "a2": "And here is what I found about b for you."}
        msgs, spec = gen_multi.assemble("recall", o)
        self.assertEqual(len(msgs), 8)
        tool_turns = [m for m in msgs if m.get("tool_calls")]
        self.assertEqual(len(tool_turns), 2)

    def test_enrich_tool_between_artifacts(self):
        # the v8 field-test failure shape: artifact -> search -> revised artifact
        o = {"tools": _TOOLS, "artifact_request": "build a page about X",
             "artifact_intro": "Here", "artifact_html": _HTML,
             "enrich_request": "search spacex.com and update it with real data",
             "tool_name": "web_search", "arguments": {"query": "site:spacex.com X"},
             "tool_result": "{\"facts\":[\"a\",\"b\"]}", "revised_intro": "Updated",
             "revised_html": "<div>\n  <p>updated with a and b</p>\n</div>"}
        msgs, spec = gen_multi.assemble("enrich", o)
        self.assertIsNotNone(spec)
        self.assertEqual(len(msgs), 6)
        # the tool call must sit BETWEEN the two assistant artifacts
        self.assertTrue(datalib.has_fence(msgs[1]["content"]))   # artifact v1
        self.assertTrue(msgs[3].get("tool_calls"))               # tool call
        self.assertEqual(msgs[4]["role"], "tool")                # result
        self.assertTrue(datalib.has_fence(msgs[5]["content"]))   # revised artifact

    def test_enrich_rejects_missing_call(self):
        o = {"tools": _TOOLS, "artifact_request": "build", "artifact_html": _HTML,
             "enrich_request": "search and update", "tool_name": "not_a_tool",
             "arguments": {"query": "x"}, "tool_result": "{}",
             "revised_html": "<div>\n  <p>changed</p>\n</div>"}
        self.assertIsNone(gen_multi.assemble("enrich", o))

    def test_research_tool_then_prose(self):
        # artifact -> "no code, just search and summarize" -> tool call -> PROSE
        o = {"tools": _TOOLS, "artifact_request": "build a page about X",
             "artifact_intro": "Here", "artifact_html": _HTML,
             "research_request": "stop making docs, search and summarize in prose",
             "tool_name": "web_search", "arguments": {"query": "X facts"},
             "tool_result": "{\"facts\":[\"a\"]}",
             "answer": "Based on the search, X is a real thing with property a and more."}
        msgs, spec = gen_multi.assemble("research", o)
        self.assertIsNotNone(spec)
        self.assertEqual(len(msgs), 6)
        self.assertTrue(msgs[3].get("tool_calls"))               # tool call
        self.assertTrue(datalib.is_prose_reply(msgs[5]["content"]))  # prose, not code

    def test_research_rejects_code_answer(self):
        o = {"tools": _TOOLS, "artifact_request": "build", "artifact_html": _HTML,
             "research_request": "search and summarize", "tool_name": "web_search",
             "arguments": {"query": "x"}, "tool_result": "{}",
             "answer": "```markdown\n# X\n- a\n```"}   # must reject code as the answer
        self.assertIsNone(gen_multi.assemble("research", o))

    def test_revise_must_change(self):
        o = {"artifact_request": "build", "artifact_intro": "Here",
             "artifact_html": _HTML, "change_request": "make it a span",
             "revise_intro": "Updated",
             "revised_html": "<span>\n  <p>hello</p>\n</span>"}
        msgs, spec = gen_multi.assemble("revise", o)
        self.assertEqual(len(msgs), 4)
        # identical v1/v2 must be rejected
        o2 = dict(o, revised_html=_HTML)
        self.assertIsNone(gen_multi.assemble("revise", o2))


class ResearchProseGate(unittest.TestCase):
    """v11: the eval axis that measures the v10 field bug (```html-wrapped answer
    to 'no html, just search'). _is_plain_prose must reject fenced AND raw HTML."""

    def setUp(self):
        import eval_universal
        self.g = eval_universal._is_plain_prose

    def test_accepts_plain_prose(self):
        self.assertTrue(self.g("The Eiffel Tower is about 330 metres tall including its antennas."))

    def test_rejects_fenced_html(self):
        self.assertFalse(self.g("```html\n<p>The tower is 330 m tall.</p>\n```"))

    def test_rejects_raw_html(self):
        self.assertFalse(self.g("<p>The tower is 330 m tall.</p>"))
        self.assertFalse(self.g("<!doctype html><html><body>x</body></html>"))

    def test_rejects_tool_call(self):
        self.assertFalse(self.g("<tool_call>{\"name\":\"web_search\"}</tool_call>"))


class ResearchUpweight(unittest.TestCase):
    """v11: research (search->prose) must get a strictly larger upweight than the
    shared toollock set, and research is no longer inside TOOLLOCK_TYPES."""

    def test_research_not_in_toollock(self):
        import config_uni as C
        self.assertNotIn("research", C.TOOLLOCK_TYPES)
        self.assertIn("enrich", C.TOOLLOCK_TYPES)

    def test_research_upweight_dominates(self):
        import config_uni as C
        self.assertGreater(C.RESEARCH_UPWEIGHT, C.TOOLLOCK_UPWEIGHT)


if __name__ == "__main__":
    unittest.main()
