"""Generate COMPLETE, RUNNABLE Python programs from NATURAL instructions — the
dimension our function+assert data never taught (why the models felt weak on
real 'write me calc.py' asks). Teacher invents a casual user request + writes the
full program; we RUN it (canned stdin, timeout) and keep only non-crashing ones.
Assistant target is stored as a fenced code block so the student learns to reply
like a chat assistant. Teacher must be served on :8091. Appends to programs.jsonl."""
import json, re, os, sys, subprocess, tempfile
import config as C, gen

TARGET     = 170
MAX_PASSES = 8
OUT        = "programs.jsonl"

DOMAINS = [
    "a command-line calculator", "a number guessing game", "a to-do list you interact with",
    "counting words or lines in text", "a temperature converter", "a bank account class with a menu",
    "reading a CSV and summarizing a column", "a Caesar-cipher encoder/decoder", "a countdown timer",
    "rock paper scissors against the computer", "a contact book kept in a dictionary",
    "finding prime numbers", "converting to/from Roman numerals", "a menu-driven text tool",
    "a password strength checker", "reading and writing a small JSON file", "a factorial/Fibonacci tool",
    "printing a tic-tac-toe board", "a length/weight unit converter", "a short multiple-choice quiz",
    "a shopping cart total with tax", "reversing the lines of a file", "a BMI calculator",
    "a simple text-based menu ordering system", "a dice-rolling simulator", "a grade average calculator",
    "a simple stack or queue you can push/pop from a prompt", "a Morse code translator",
    "a simple stopwatch using time", "a program that finds the largest number in a list the user enters",
    "a mad-libs style story filler", "a simple calculator that keeps a running total",
    "a program that checks if a year is a leap year", "counting vowels and consonants in a word",
]

TEMPLATES = [
    "Invent a realistic, casual request a beginner/intermediate programmer might make about {d}, then fully solve it.",
    "Come up with a natural 'can you write me...' request themed around {d}, and answer it completely.",
    "Think of a small practical Python task about {d} phrased the way a real person would ask, then solve it.",
]

STDIN = "5\n3\n+\n42\nAlice\nadd\nbuy milk\nlist\n7\n50\n25\ny\nyes\nquit\nq\nexit\n2\n1\n\n\n"

def build(d, tmpl):
    return (tmpl.format(d=d) +
            " The instruction must read like a person casually asking (1-2 sentences), NOT like a"
            " spec. The solution must be a COMPLETE, runnable Python program: all imports, uses"
            " input() for any interaction, idiomatic and clean, NO placeholders/TODOs/ellipses,"
            " runnable as-is. Put RAW code in the JSON value (no markdown fences)."
            ' Respond ONLY with JSON: {"instruction": "<natural request>", "solution": "<complete program>"}')

def runs_ok(code):
    try:
        compile(code, "<s>", "exec")
    except SyntaxError:
        return False
    if re.search(r"#\s*(your code|todo|implement|fill in)\b|\.\.\.\s*$", code, re.I | re.M):
        return False                                  # reject placeholders
    d = tempfile.mkdtemp()
    p = os.path.join(d, "p.py")
    open(p, "w").write(code)
    open(os.path.join(d, "numbers.txt"), "w").write("3\n7\n10\n5\n")
    open(os.path.join(d, "data.csv"), "w").write("name,score\na,3\nb,7\n")
    try:
        r = subprocess.run([sys.executable, p], input=STDIN, capture_output=True,
                           text=True, timeout=6, cwd=d)
        if "Traceback" in r.stderr:
            return False
        return True
    except subprocess.TimeoutExpired:
        return True
    except Exception:
        return False
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)

def norm(s): return re.sub(r"\s+", " ", s.strip().lower())

seen = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try: seen.add(norm(json.loads(line)["messages"][0]["content"]))
        except Exception: pass

combos = [(d, t) for d in DOMAINS for t in TEMPLATES]
kept = 0
with open(OUT, "a") as out:
    for p in range(MAX_PASSES):
        if kept >= TARGET: break
        for d, tmpl in combos:
            if kept >= TARGET: break
            try:
                obj = gen.extract_json(gen.ask(build(d, tmpl), C.GEN_TEMPERATURE))
                instr = obj["instruction"].strip()
                sol = gen.strip_fence(obj["solution"])
            except Exception:
                continue
            k = norm(instr)
            if not instr or not sol or k in seen: continue
            if not runs_ok(sol): continue
            seen.add(k)
            out.write(json.dumps({
                "messages": [{"role": "user", "content": instr},
                             {"role": "assistant", "content": "```python\n" + sol.strip() + "\n```"}],
                "lang": "Python", "task": "complete program", "difficulty": "", "tests": None}) + "\n")
            out.flush(); kept += 1
            if kept % 10 == 0: print(f"[pass {p+1}] kept {kept}/{TARGET}", flush=True)
print(f"DONE. {kept} complete-program samples -> {OUT}")
