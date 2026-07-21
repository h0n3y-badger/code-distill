"""Unit regression tests for datalib: row validity + C header-completeness lint.
These are pure/fast (no model, no network). Run: python -m unittest discover tests
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datalib


class TestIsValidMessages(unittest.TestCase):
    def test_good_pair(self):
        self.assertTrue(datalib.is_valid_messages(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "def f(): pass"}]))

    def test_rejects_none_content(self):
        # the exact bug that broke `make train`: null assistant content
        self.assertFalse(datalib.is_valid_messages(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": None}]))

    def test_rejects_empty_content(self):
        self.assertFalse(datalib.is_valid_messages(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "   "}]))

    def test_rejects_dict_content(self):
        # clean_v2 has 11 C rows with dict content; these must be filtered out
        self.assertFalse(datalib.is_valid_messages(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": {"nested": "obj"}}]))

    def test_rejects_too_few(self):
        self.assertFalse(datalib.is_valid_messages(
            [{"role": "user", "content": "hi"}]))

    def test_rejects_non_list(self):
        self.assertFalse(datalib.is_valid_messages("not a list"))
        self.assertFalse(datalib.is_valid_messages(None))


class TestMissingHeaders(unittest.TestCase):
    def test_limits_needed(self):
        src = "int f(void){ return INT_MAX; }"
        self.assertIn("limits.h", datalib.missing_headers(src))

    def test_limits_satisfied(self):
        src = "#include <limits.h>\nint f(void){ return INT_MAX; }"
        self.assertNotIn("limits.h", datalib.missing_headers(src))

    def test_ctype_needed(self):
        src = "int f(char c){ return isalpha(c); }"
        self.assertIn("ctype.h", datalib.missing_headers(src))

    def test_math_needed(self):
        src = "double f(double x){ return sqrt(x); }"
        self.assertIn("math.h", datalib.missing_headers(src))

    def test_stdbool_needed(self):
        src = "bool f(void){ return true; }"
        self.assertIn("stdbool.h", datalib.missing_headers(src))

    def test_stdint_needed(self):
        src = "uint32_t f(uint32_t x){ return x; }"
        self.assertIn("stdint.h", datalib.missing_headers(src))

    def test_fully_self_contained(self):
        src = ("#include <limits.h>\n#include <ctype.h>\n#include <math.h>\n"
               "int f(char c){ return isalpha(c) ? (int)sqrt((double)INT_MAX) : 0; }")
        self.assertEqual(datalib.missing_headers(src), set())

    def test_symbol_in_comment_ignored(self):
        # a header symbol mentioned only in a comment must not trip the lint
        src = "int f(void){ return 0; } // uses INT_MAX conceptually"
        self.assertEqual(datalib.missing_headers(src), set())

    def test_non_string_safe(self):
        self.assertEqual(datalib.missing_headers({"not": "a string"}), set())


if __name__ == "__main__":
    unittest.main()
