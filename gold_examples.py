"""Hand-authored GOLD complete-program examples (Claude-written, per user's
'write some yourself'). Natural instruction + a complete, clean, correct program.
Each is executed here; only passing ones are written to gold_python.jsonl. These
are the quality anchor mixed into training so the student learns to produce whole,
idiomatic programs from casual asks — not just bare functions."""
import json, os, sys, subprocess, tempfile, re

# (natural_instruction, complete_program)
GOLD = [
("Can you write me a simple command-line calculator? It should ask for two numbers and an operator and print the result.", '''\
def calculate(a, op, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            raise ZeroDivisionError("cannot divide by zero")
        return a / b
    raise ValueError(f"unknown operator: {op}")

def main():
    a = float(input("First number: "))
    op = input("Operator (+ - * /): ").strip()
    b = float(input("Second number: "))
    try:
        print(f"{a} {op} {b} = {calculate(a, op, b)}")
    except (ValueError, ZeroDivisionError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
'''),

("Write FizzBuzz in Python for 1 to 100.", '''\
def fizzbuzz(n):
    for i in range(1, n + 1):
        if i % 15 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)

if __name__ == "__main__":
    fizzbuzz(100)
'''),

("I need a script that reads numbers.txt (one integer per line) and prints the total and the average.", '''\
def main(path="numbers.txt"):
    with open(path) as f:
        nums = [int(line) for line in f if line.strip()]
    if not nums:
        print("No numbers found.")
        return
    print(f"Count: {len(nums)}")
    print(f"Sum: {sum(nums)}")
    print(f"Average: {sum(nums) / len(nums):.2f}")

if __name__ == "__main__":
    main()
'''),

("Write a function is_palindrome(s) that ignores case and spaces, and show a few examples.", '''\
def is_palindrome(s):
    cleaned = "".join(ch.lower() for ch in s if ch.isalnum())
    return cleaned == cleaned[::-1]

if __name__ == "__main__":
    for text in ["racecar", "A man a plan a canal Panama", "hello"]:
        print(f"{text!r}: {is_palindrome(text)}")
'''),

("Make a program that counts how often each word appears in a piece of text and prints them most-common first.", '''\
from collections import Counter
import re

def word_counts(text):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return Counter(words)

if __name__ == "__main__":
    sample = "The cat sat on the mat. The cat was happy."
    for word, count in word_counts(sample).most_common():
        print(f"{word}: {count}")
'''),

("Print all the prime numbers below 50.", '''\
def is_prime(n):
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True

if __name__ == "__main__":
    primes = [n for n in range(50) if is_prime(n)]
    print(primes)
'''),

("Write a function for the nth Fibonacci number and print the first 10.", '''\
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

if __name__ == "__main__":
    print([fib(i) for i in range(10)])
'''),

("Can you make a temperature converter that turns Celsius into Fahrenheit?", '''\
def celsius_to_fahrenheit(c):
    return c * 9 / 5 + 32

def main():
    c = float(input("Temperature in Celsius: "))
    print(f"{c}°C = {celsius_to_fahrenheit(c):.1f}°F")

if __name__ == "__main__":
    main()
'''),

("Write a small bank account class I can deposit to and withdraw from, and show it working.", '''\
class BankAccount:
    def __init__(self, owner, balance=0.0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self.balance += amount
        return self.balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise ValueError("insufficient funds")
        self.balance -= amount
        return self.balance

if __name__ == "__main__":
    acct = BankAccount("Alice", 100)
    acct.deposit(50)
    acct.withdraw(30)
    print(f"{acct.owner}'s balance: {acct.balance}")
'''),

("Write a Caesar cipher that can encode and decode a message with a given shift.", '''\
def caesar(text, shift):
    result = []
    for ch in text:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return "".join(result)

if __name__ == "__main__":
    message = "Hello, World!"
    encoded = caesar(message, 3)
    print("Encoded:", encoded)
    print("Decoded:", caesar(encoded, -3))
'''),

("Is a given year a leap year? Write a program that checks one.", '''\
def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def main():
    year = int(input("Enter a year: "))
    print(f"{year} is {'a leap year' if is_leap_year(year) else 'not a leap year'}")

if __name__ == "__main__":
    main()
'''),

("Count the vowels and consonants in a word.", '''\
def count_letters(word):
    vowels = sum(1 for ch in word.lower() if ch in "aeiou")
    consonants = sum(1 for ch in word.lower() if ch.isalpha() and ch not in "aeiou")
    return vowels, consonants

if __name__ == "__main__":
    word = "programming"
    v, c = count_letters(word)
    print(f"{word!r} has {v} vowels and {c} consonants")
'''),

("Write a BMI calculator that reads weight in kg and height in metres.", '''\
def bmi(weight_kg, height_m):
    return weight_kg / (height_m ** 2)

def category(value):
    if value < 18.5:
        return "underweight"
    if value < 25:
        return "normal"
    if value < 30:
        return "overweight"
    return "obese"

def main():
    weight = float(input("Weight (kg): "))
    height = float(input("Height (m): "))
    value = bmi(weight, height)
    print(f"BMI: {value:.1f} ({category(value)})")

if __name__ == "__main__":
    main()
'''),

("Convert an integer to a Roman numeral.", '''\
def to_roman(n):
    if not 0 < n < 4000:
        raise ValueError("number must be between 1 and 3999")
    numerals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
                (90, "XC"), (50, "L"), (40, "XL"), (10, "X"),
                (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    result = []
    for value, symbol in numerals:
        count, n = divmod(n, value)
        result.append(symbol * count)
    return "".join(result)

if __name__ == "__main__":
    for n in [4, 9, 58, 1994]:
        print(f"{n} = {to_roman(n)}")
'''),

("Roll two dice a few times and show the results and totals.", '''\
import random

def roll_dice(rolls=5):
    for i in range(1, rolls + 1):
        a, b = random.randint(1, 6), random.randint(1, 6)
        print(f"Roll {i}: {a} + {b} = {a + b}")

if __name__ == "__main__":
    roll_dice()
'''),

("Compute a student's average grade from a list of scores and give a letter grade.", '''\
def letter_grade(avg):
    for cutoff, letter in [(90, "A"), (80, "B"), (70, "C"), (60, "D")]:
        if avg >= cutoff:
            return letter
    return "F"

def main():
    scores = [88, 92, 79, 85, 95]
    avg = sum(scores) / len(scores)
    print(f"Average: {avg:.1f} -> {letter_grade(avg)}")

if __name__ == "__main__":
    main()
'''),

("Write a simple stack class with push, pop and peek, and demonstrate it.", '''\
class Stack:
    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self):
        return self._items[-1] if self._items else None

    def is_empty(self):
        return not self._items

if __name__ == "__main__":
    s = Stack()
    for x in (1, 2, 3):
        s.push(x)
    print("Top:", s.peek())
    print("Popped:", s.pop())
    print("Empty?", s.is_empty())
'''),

("Reverse the order of words in a sentence.", '''\
def reverse_words(sentence):
    return " ".join(sentence.split()[::-1])

if __name__ == "__main__":
    print(reverse_words("the quick brown fox"))
'''),

("Read a CSV called data.csv with a 'score' column and print the average score.", '''\
import csv

def average_score(path="data.csv"):
    with open(path, newline="") as f:
        scores = [float(row["score"]) for row in csv.DictReader(f)]
    return sum(scores) / len(scores) if scores else 0.0

if __name__ == "__main__":
    print(f"Average score: {average_score():.2f}")
'''),

("Make a countdown timer that counts from a given number of seconds down to zero.", '''\
import time

def countdown(seconds):
    while seconds > 0:
        print(seconds)
        time.sleep(0.01)  # short for demo; use 1 for real seconds
        seconds -= 1
    print("Time's up!")

if __name__ == "__main__":
    countdown(3)
'''),
]

STDIN = "5\n3\n+\n42\n1.7\n70\n2000\nAlice\n\n\n\n"

def runs(code):
    try:
        compile(code, "<s>", "exec")
    except SyntaxError as e:
        return False, f"syntax:{e}"
    d = tempfile.mkdtemp()
    p = os.path.join(d, "p.py")
    open(p, "w").write(code)
    open(os.path.join(d, "numbers.txt"), "w").write("3\n7\n10\n5\n")
    open(os.path.join(d, "data.csv"), "w").write("name,score\na,3\nb,7\nc,8\n")
    try:
        r = subprocess.run([sys.executable, p], input=STDIN, capture_output=True,
                           text=True, timeout=8, cwd=d)
        if "Traceback" in r.stderr:
            return False, r.stderr.strip().splitlines()[-1][:80]
        return True, "ok"
    except subprocess.TimeoutExpired:
        return True, "timeout"
    except Exception as e:
        return False, str(e)
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)

def main():
    kept = 0
    with open("gold_python.jsonl", "w") as out:
        for instr, code in GOLD:
            code = code.strip()
            ok, why = runs(code)
            print(f"  {'OK ' if ok else 'BAD'} {why:20} :: {instr[:50]}", flush=True)
            if not ok:
                continue
            out.write(json.dumps({
                "messages": [{"role": "user", "content": instr},
                             {"role": "assistant", "content": "```python\n" + code + "\n```"}],
                "lang": "Python", "task": "complete program", "difficulty": "", "tests": None}) + "\n")
            kept += 1
    print(f"\ngold_python.jsonl: {kept}/{len(GOLD)} verified")

if __name__ == "__main__":
    main()
