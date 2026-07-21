"""Regression tests over the actual training/eval corpus files. These guard the
invariants that broke things historically:
  * every trainable row is a valid message pair (the null-content crash);
  * the eval set stays disjoint and well-formed;
  * C data stays include-complete (the fix that lifted C pass@1) — enforced hard
    on the gold set and via a low ceiling on the teacher-generated corpus.
Files that don't exist yet are skipped, so this runs on a fresh checkout too."""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import datalib

VALID_LANGS = {"Python", "C", "HTML", "Java"}


def _path(name):
    return os.path.join(ROOT, name)


def _exists(name):
    return os.path.isfile(_path(name))


class TestGoldC(unittest.TestCase):
    """The hand-authored C gold set must be perfectly self-contained."""

    @unittest.skipUnless(_exists("gold_c.jsonl"), "gold_c.jsonl not built")
    def test_gold_c_all_self_contained(self):
        offenders = []
        for obj in datalib.iter_jsonl(_path("gold_c.jsonl")):
            self.assertTrue(datalib.is_valid_messages(obj["messages"]))
            sol = obj["messages"][-1]["content"]
            miss = datalib.missing_headers(sol)
            if miss:
                offenders.append((miss, sol[:60]))
        self.assertEqual(offenders, [], f"gold C missing headers: {offenders}")


class TestCleanCorpus(unittest.TestCase):
    CORPUS = next((f for f in ("clean_v3.jsonl", "clean_v2.jsonl", "clean.jsonl")
                   if _exists(f)), None)

    @unittest.skipUnless(CORPUS, "no clean corpus present")
    def test_langs_known(self):
        for obj in datalib.iter_jsonl(_path(self.CORPUS)):
            self.assertIn(obj.get("lang"), VALID_LANGS)

    @unittest.skipUnless(CORPUS, "no clean corpus present")
    def test_valid_rows_render_cleanly(self):
        # every row that passes the gate must have a str user + str assistant
        for obj in datalib.iter_jsonl(_path(self.CORPUS)):
            if datalib.is_valid_messages(obj.get("messages")):
                for m in obj["messages"]:
                    self.assertIsInstance(m["content"], str)
                    self.assertTrue(m["content"].strip())

    @unittest.skipUnless(CORPUS, "no clean corpus present")
    def test_c_include_completeness_ceiling(self):
        # rejection sampling guarantees compilation, so header-incompleteness the
        # lint can see should be rare. A spike means a systematic regression in
        # how C data is generated. Ceiling is generous (heuristic false positives
        # exist on bug-fix fragments); baseline observed was ~2%.
        total = flagged = 0
        for obj in datalib.iter_jsonl(_path(self.CORPUS)):
            if obj.get("lang") != "C":
                continue
            sol = obj["messages"][-1]["content"]
            if not isinstance(sol, str):
                continue
            total += 1
            if datalib.missing_headers(sol):
                flagged += 1
        self.assertGreater(total, 0)
        rate = flagged / total
        self.assertLess(rate, 0.05,
                        f"{flagged}/{total} C rows flagged missing-header ({rate:.1%})")


class TestEvalSet(unittest.TestCase):
    @unittest.skipUnless(_exists("eval_set.jsonl"), "eval_set.jsonl missing")
    def test_eval_scorable(self):
        langs = {"Python": 0, "C": 0}
        for obj in datalib.iter_jsonl(_path("eval_set.jsonl")):
            lang = obj.get("lang")
            if lang in langs and obj.get("tests"):
                # must have a user instruction to solve
                roles = [m.get("role") for m in obj["messages"]]
                self.assertIn("user", roles)
                langs[lang] += 1
        self.assertGreater(langs["Python"], 0)
        self.assertGreater(langs["C"], 0)

    @unittest.skipUnless(_exists("eval_set.jsonl") and _exists("clean.jsonl"),
                         "need eval_set.jsonl + clean.jsonl")
    def test_eval_disjoint_from_train(self):
        # contamination guard: no eval user-prompt appears verbatim in training
        train_prompts = set()
        for obj in datalib.iter_jsonl(_path("clean.jsonl")):
            for m in obj.get("messages", []):
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    train_prompts.add(m["content"].strip())
        for obj in datalib.iter_jsonl(_path("eval_set.jsonl")):
            for m in obj.get("messages", []):
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    self.assertNotIn(m["content"].strip(), train_prompts)


if __name__ == "__main__":
    unittest.main()
