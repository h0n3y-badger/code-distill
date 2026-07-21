"""Regression tests for the execution-verification gate (gen.verify) — the heart
of rejection sampling. If this silently breaks, bad samples leak into training.
Requires gcc on PATH for the C cases (skipped if absent)."""
import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gen

HAVE_GCC = shutil.which("gcc") is not None


class TestPythonVerify(unittest.TestCase):
    def test_passing_python(self):
        sol = "def add(a, b):\n    return a + b"
        tests = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
        self.assertTrue(gen.verify("Python", sol, tests))

    def test_failing_python_assertion(self):
        sol = "def add(a, b):\n    return a - b"       # wrong
        tests = "assert add(2, 3) == 5"
        self.assertFalse(gen.verify("Python", sol, tests))

    def test_python_syntax_error(self):
        self.assertFalse(gen.verify("Python", "def broken(:", "assert True"))


@unittest.skipUnless(HAVE_GCC, "gcc not available")
class TestCVerify(unittest.TestCase):
    def test_passing_c(self):
        sol = "int square(int x) { return x * x; }"
        tests = ("#include <assert.h>\n"
                 "int main(void){ assert(square(4) == 16); return 0; }")
        self.assertTrue(gen.verify("C", sol, tests))

    def test_c_compile_failure(self):
        # uses INT_MAX without <limits.h> -> compile error -> must be rejected
        sol = "int f(void){ return INT_MAX; }"
        tests = "#include <assert.h>\nint main(void){ assert(f() > 0); return 0; }"
        self.assertFalse(gen.verify("C", sol, tests))

    def test_c_runtime_failure(self):
        sol = "int square(int x) { return x + x; }"    # wrong: returns 2x
        tests = ("#include <assert.h>\n"
                 "int main(void){ assert(square(4) == 16); return 0; }")
        self.assertFalse(gen.verify("C", sol, tests))


class TestNonExecutableLangs(unittest.TestCase):
    def test_html_trusted(self):
        # non-verifiable langs are trusted (True) by design
        self.assertTrue(gen.verify("HTML", "<div></div>", "n/a"))


if __name__ == "__main__":
    unittest.main()
