"""Practical smoke test — does the model write COMPLETE, RUNNABLE programs from
NATURAL instructions? (This is what real use looks like, unlike the function+assert
eval.) For each prompt: get the model's reply, extract code, run it with canned
stdin, and check it executes without a traceback. Also dumps every reply to
smoke_out/<tag>/ so the actual quality can be read, not just pass/fail.
Serve a model on :8091, then: python smoke_test.py <tag>"""
import os, re, sys, json, subprocess, tempfile
from openai import OpenAI
import config as C

client = OpenAI(base_url=C.TEACHER_BASE_URL, api_key=C.TEACHER_API_KEY)
TAG = sys.argv[1] if len(sys.argv) > 1 else "model"
OUTDIR = os.path.join("smoke_out", TAG)
os.makedirs(OUTDIR, exist_ok=True)

# Realistic, beginner→intermediate "write me a program" asks, phrased like a person.
PROMPTS = [
    ("calc", "Write a simple command-line calculator in Python. It should ask the user for two numbers and an operator (+, -, *, /) and print the result."),
    ("fizzbuzz", "Write FizzBuzz in Python for the numbers 1 to 100."),
    ("avg_file", "Write a Python script that reads a file called numbers.txt with one integer per line and prints the sum and the average."),
    ("palindrome", "Write a Python function is_palindrome(s) that returns True if s is a palindrome, and show a couple of example calls with print."),
    ("greet", "Write a short Python program that asks the user for their name and then greets them."),
    ("todo", "Write a small command-line to-do list program in Python where you can add a task and list all tasks."),
    ("wordcount", "Write a Python script that counts how many times each word appears in a string and prints the counts."),
    ("guess", "Write a number guessing game in Python: the program picks a random number from 1 to 100 and the user guesses until correct."),
    ("fib", "Write a Python function that returns the nth Fibonacci number, and print the first 10 Fibonacci numbers."),
    ("temp", "Write a Python program that converts a temperature from Celsius to Fahrenheit."),
    ("primes", "Write a Python program that prints all prime numbers below 50."),
    ("reverse_words", "Write a Python function that takes a sentence and returns it with the word order reversed."),
]

STDIN = "5\n3\n+\n42\nAlice\nadd\nbuy milk\nlist\n7\n50\n25\nquit\nexit\n\n"

def extract_code(text):
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()

def runs_ok(code):
    """True if the code is valid Python and runs without raising (timeout = ok)."""
    try:
        compile(code, "<s>", "exec")
    except SyntaxError:
        return False, "syntax-error"
    d = tempfile.mkdtemp()
    p = os.path.join(d, "prog.py")
    open(p, "w").write(code)
    # pre-seed numbers.txt in case the program reads it
    open(os.path.join(d, "numbers.txt"), "w").write("3\n7\n10\n5\n")
    try:
        r = subprocess.run([sys.executable, p], input=STDIN, capture_output=True,
                           text=True, timeout=6, cwd=d)
        if r.returncode == 0:
            return True, "ok"
        if "Traceback" in r.stderr:
            return False, r.stderr.strip().splitlines()[-1][:80]
        return True, f"exit{r.returncode}"       # non-zero but no traceback (e.g. sys.exit)
    except subprocess.TimeoutExpired:
        return True, "timeout(ran)"               # interactive loop kept waiting = it ran
    except Exception as e:
        return False, f"harness:{e}"

def main():
    ok = 0
    for name, prompt in PROMPTS:
        try:
            resp = client.chat.completions.create(model="m", temperature=0.2,
                messages=[{"role": "user", "content": prompt}]).choices[0].message.content
        except Exception as e:
            resp = f"<<request failed: {e}>>"
        code = extract_code(resp)
        good, why = runs_ok(code)
        ok += good
        open(os.path.join(OUTDIR, f"{name}.txt"), "w").write(
            f"# PROMPT\n{prompt}\n\n# REPLY\n{resp}\n\n# VERDICT runs={good} ({why})\n")
        print(f"  {name:14} runs={'Y' if good else 'N'}  ({why})", flush=True)
    print(f"\nSMOKE {TAG}: {ok}/{len(PROMPTS)} programs ran = {ok/len(PROMPTS):.0%}")

if __name__ == "__main__":
    main()
