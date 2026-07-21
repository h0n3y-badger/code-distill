"""Hand-authored, gcc-verified GOLD C examples that deliberately exercise the
headers the teacher-generated C data underweights (<limits.h>, <ctype.h>,
<math.h>, <stdbool.h>, <stdint.h>) plus canonical struct/typedef naming.

Diagnosis (diag_c.py on v2) showed the residual C failures are dominated by
missing_include/decl: the student uses INT_MIN / isalpha / sqrt / bool but never
learned to emit the matching #include, because those headers appear in only
2-4% of training C. Each example here is SELF-CONTAINED (every #include present),
uses the assert-only test convention, and is compiled+run before being kept.

Writes passing rows to gold_c.jsonl in the same schema as clean.jsonl.
"""
import json, os, subprocess, tempfile, shutil

# (instruction, solution, tests-as-main-with-assert-only)
EXAMPLES = [
    # ---- limits.h : integer overflow / bounds --------------------------------
    ("Implement `int safe_add(int a, int b, int *out)` that stores a+b in *out and "
     "returns 0 on success, or -1 (without touching *out) if the addition would "
     "overflow the range of int. Use the limits of int to detect overflow.",
     """#include <limits.h>

int safe_add(int a, int b, int *out) {
    if (b > 0 && a > INT_MAX - b) return -1;
    if (b < 0 && a < INT_MIN - b) return -1;
    *out = a + b;
    return 0;
}""",
     """#include <assert.h>
#include <limits.h>
int main(void) {
    int r;
    assert(safe_add(2, 3, &r) == 0 && r == 5);
    assert(safe_add(INT_MAX, 1, &r) == -1);
    assert(safe_add(INT_MIN, -1, &r) == -1);
    assert(safe_add(INT_MAX, 0, &r) == 0 && r == INT_MAX);
    return 0;
}"""),

    ("Implement `int safe_abs(int x, int *out)` that stores the absolute value of x "
     "in *out and returns 0, but returns -1 if x is INT_MIN (whose absolute value "
     "is not representable as an int).",
     """#include <limits.h>

int safe_abs(int x, int *out) {
    if (x == INT_MIN) return -1;
    *out = x < 0 ? -x : x;
    return 0;
}""",
     """#include <assert.h>
#include <limits.h>
int main(void) {
    int r;
    assert(safe_abs(-7, &r) == 0 && r == 7);
    assert(safe_abs(5, &r) == 0 && r == 5);
    assert(safe_abs(INT_MIN, &r) == -1);
    return 0;
}"""),

    ("Implement `int clamp_int(int v, int lo, int hi)` that clamps v to the inclusive "
     "range [lo, hi]. Also implement `int int_bits(void)` that returns the number of "
     "value bits usable in a signed int on this platform using the values in "
     "<limits.h> (return the count of bits in INT_MAX plus one for the sign).",
     """#include <limits.h>

int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

int int_bits(void) {
    int bits = 1;              /* sign bit */
    unsigned int m = INT_MAX;
    while (m) { bits++; m >>= 1; }
    return bits;
}""",
     """#include <assert.h>
int main(void) {
    assert(clamp_int(5, 0, 10) == 5);
    assert(clamp_int(-3, 0, 10) == 0);
    assert(clamp_int(99, 0, 10) == 10);
    assert(int_bits() == 32 || int_bits() == 64 || int_bits() == 16);
    return 0;
}"""),

    # ---- ctype.h : character classification / case ---------------------------
    ("Implement `int count_vowels(const char *s)` that returns the number of vowel "
     "characters (a, e, i, o, u, case-insensitive) in the null-terminated string s.",
     """#include <ctype.h>

int count_vowels(const char *s) {
    int n = 0;
    for (; *s; s++) {
        int c = tolower((unsigned char)*s);
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') n++;
    }
    return n;
}""",
     """#include <assert.h>
int main(void) {
    assert(count_vowels("Hello World") == 3);
    assert(count_vowels("XYZ") == 0);
    assert(count_vowels("AEIOUaeiou") == 10);
    assert(count_vowels("") == 0);
    return 0;
}"""),

    ("Implement `void to_upper_inplace(char *s)` that converts every alphabetic "
     "character in the null-terminated string s to uppercase, leaving other "
     "characters unchanged.",
     """#include <ctype.h>

void to_upper_inplace(char *s) {
    for (; *s; s++)
        *s = (char)toupper((unsigned char)*s);
}""",
     """#include <assert.h>
#include <string.h>
int main(void) {
    char a[] = "Hello, World! 123";
    to_upper_inplace(a);
    assert(strcmp(a, "HELLO, WORLD! 123") == 0);
    char b[] = "";
    to_upper_inplace(b);
    assert(strcmp(b, "") == 0);
    return 0;
}"""),

    ("Implement `int count_digits(const char *s)` that returns how many decimal digit "
     "characters appear in the null-terminated string s. Use the standard character "
     "classification facilities.",
     """#include <ctype.h>

int count_digits(const char *s) {
    int n = 0;
    for (; *s; s++)
        if (isdigit((unsigned char)*s)) n++;
    return n;
}""",
     """#include <assert.h>
int main(void) {
    assert(count_digits("abc123") == 3);
    assert(count_digits("2026-07-21") == 8);
    assert(count_digits("none") == 0);
    return 0;
}"""),

    ("Implement `char *caesar_shift(char *s, int k)` that shifts each ASCII letter in "
     "the null-terminated string s forward by k positions within its case (wrapping "
     "a-z and A-Z), leaves non-letters unchanged, modifies s in place, and returns s.",
     """#include <ctype.h>

char *caesar_shift(char *s, int k) {
    k = ((k % 26) + 26) % 26;
    for (char *p = s; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (isupper(c)) *p = (char)('A' + (c - 'A' + k) % 26);
        else if (islower(c)) *p = (char)('a' + (c - 'a' + k) % 26);
    }
    return s;
}""",
     """#include <assert.h>
#include <string.h>
int main(void) {
    char a[] = "abcXYZ 1!";
    caesar_shift(a, 3);
    assert(strcmp(a, "defABC 1!") == 0);
    char b[] = "Hello";
    caesar_shift(b, 26);
    assert(strcmp(b, "Hello") == 0);
    return 0;
}"""),

    # ---- math.h --------------------------------------------------------------
    ("Implement `int is_prime(int n)` that returns 1 if n is a prime number and 0 "
     "otherwise, testing divisors only up to the square root of n.",
     """#include <math.h>

int is_prime(int n) {
    if (n < 2) return 0;
    if (n % 2 == 0) return n == 2;
    int limit = (int)sqrt((double)n);
    for (int d = 3; d <= limit; d += 2)
        if (n % d == 0) return 0;
    return 1;
}""",
     """#include <assert.h>
int main(void) {
    assert(is_prime(2) == 1);
    assert(is_prime(17) == 1);
    assert(is_prime(1) == 0);
    assert(is_prime(15) == 0);
    assert(is_prime(97) == 1);
    return 0;
}"""),

    ("Implement `double euclidean(double x1, double y1, double x2, double y2)` that "
     "returns the Euclidean distance between the two points, and `int nearly_equal("
     "double a, double b)` that returns 1 if the two doubles differ by less than "
     "1e-9. Use the standard math library.",
     """#include <math.h>

double euclidean(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1, dy = y2 - y1;
    return sqrt(dx * dx + dy * dy);
}

int nearly_equal(double a, double b) {
    return fabs(a - b) < 1e-9;
}""",
     """#include <assert.h>
int main(void) {
    assert(nearly_equal(euclidean(0, 0, 3, 4), 5.0));
    assert(nearly_equal(euclidean(1, 1, 1, 1), 0.0));
    assert(nearly_equal(1.0, 1.0 + 1e-12));
    assert(!nearly_equal(1.0, 2.0));
    return 0;
}"""),

    # ---- stdbool.h : boolean predicates --------------------------------------
    ("Implement `bool is_power_of_two(unsigned int n)` returning true when n is a "
     "positive power of two, and `bool all_even(const int *a, int len)` returning "
     "true when every element of the array is even (true for an empty array). Return "
     "genuine bool values.",
     """#include <stdbool.h>

bool is_power_of_two(unsigned int n) {
    return n != 0 && (n & (n - 1)) == 0;
}

bool all_even(const int *a, int len) {
    for (int i = 0; i < len; i++)
        if (a[i] % 2 != 0) return false;
    return true;
}""",
     """#include <assert.h>
int main(void) {
    assert(is_power_of_two(1));
    assert(is_power_of_two(1024));
    assert(!is_power_of_two(0));
    assert(!is_power_of_two(6));
    int e[] = {2, 4, 6}; assert(all_even(e, 3));
    int o[] = {2, 3, 4}; assert(!all_even(o, 3));
    assert(all_even(e, 0));
    return 0;
}"""),

    # ---- stdint.h : fixed-width ---------------------------------------------
    ("Implement `int popcount_u32(uint32_t x)` that returns the number of set bits in "
     "the 32-bit unsigned value x, and `uint32_t reverse_bytes_u32(uint32_t x)` that "
     "returns x with its four bytes in reverse order. Use fixed-width integer types.",
     """#include <stdint.h>

int popcount_u32(uint32_t x) {
    int n = 0;
    while (x) { n += (int)(x & 1u); x >>= 1; }
    return n;
}

uint32_t reverse_bytes_u32(uint32_t x) {
    return ((x & 0x000000FFu) << 24) |
           ((x & 0x0000FF00u) << 8)  |
           ((x & 0x00FF0000u) >> 8)  |
           ((x & 0xFF000000u) >> 24);
}""",
     """#include <assert.h>
#include <stdint.h>
int main(void) {
    assert(popcount_u32(0u) == 0);
    assert(popcount_u32(0xFFFFFFFFu) == 32);
    assert(popcount_u32(0b1011u) == 3);
    assert(reverse_bytes_u32(0x11223344u) == 0x44332211u);
    return 0;
}"""),

    # ---- canonical struct naming (unknown_type bucket) -----------------------
    ("Define a singly linked list node exactly as `typedef struct Node { int value; "
     "struct Node *next; } Node;` and implement `void push_front(Node **head, int "
     "value)` which prepends a new node, and `int list_length(const Node *head)` "
     "which returns the number of nodes. Use the exact type name Node.",
     """#include <stdlib.h>
#include <stddef.h>

typedef struct Node {
    int value;
    struct Node *next;
} Node;

void push_front(Node **head, int value) {
    Node *n = (Node *)malloc(sizeof(Node));
    n->value = value;
    n->next = *head;
    *head = n;
}

int list_length(const Node *head) {
    int len = 0;
    for (const Node *p = head; p; p = p->next) len++;
    return len;
}""",
     """#include <assert.h>
#include <stddef.h>
int main(void) {
    Node *head = NULL;
    assert(list_length(head) == 0);
    push_front(&head, 3);
    push_front(&head, 2);
    push_front(&head, 1);
    assert(list_length(head) == 3);
    assert(head->value == 1 && head->next->value == 2);
    return 0;
}"""),

    ("Define `typedef struct { int *data; int size; int capacity; } IntVec;` and "
     "implement `void vec_init(IntVec *v)` (empty vector), `void vec_push(IntVec *v, "
     "int x)` (append, growing capacity as needed), and `int vec_get(const IntVec *v, "
     "int i)` (return element i). Use the exact type name IntVec.",
     """#include <stdlib.h>

typedef struct {
    int *data;
    int size;
    int capacity;
} IntVec;

void vec_init(IntVec *v) {
    v->data = NULL;
    v->size = 0;
    v->capacity = 0;
}

void vec_push(IntVec *v, int x) {
    if (v->size == v->capacity) {
        int cap = v->capacity ? v->capacity * 2 : 4;
        v->data = (int *)realloc(v->data, (size_t)cap * sizeof(int));
        v->capacity = cap;
    }
    v->data[v->size++] = x;
}

int vec_get(const IntVec *v, int i) {
    return v->data[i];
}""",
     """#include <assert.h>
int main(void) {
    IntVec v;
    vec_init(&v);
    for (int i = 0; i < 10; i++) vec_push(&v, i * i);
    assert(v.size == 10);
    assert(vec_get(&v, 0) == 0);
    assert(vec_get(&v, 3) == 9);
    assert(vec_get(&v, 9) == 81);
    return 0;
}"""),

    # ---- string.h + stdlib combo (common but pin it) -------------------------
    ("Implement `char *str_reverse_dup(const char *s)` that returns a newly malloc'd "
     "null-terminated string containing the characters of s in reverse order. The "
     "caller frees it. Return NULL if allocation fails.",
     """#include <stdlib.h>
#include <string.h>

char *str_reverse_dup(const char *s) {
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    for (size_t i = 0; i < n; i++)
        out[i] = s[n - 1 - i];
    out[n] = '\\0';
    return out;
}""",
     """#include <assert.h>
#include <string.h>
#include <stdlib.h>
int main(void) {
    char *r = str_reverse_dup("abcde");
    assert(strcmp(r, "edcba") == 0);
    free(r);
    char *e = str_reverse_dup("");
    assert(strcmp(e, "") == 0);
    free(e);
    return 0;
}"""),

    ("Implement `long parse_long_or(const char *s, long fallback)` that parses s as a "
     "base-10 long using the standard library and returns the parsed value, or "
     "returns fallback if s has no leading number or the whole string is not numeric.",
     """#include <stdlib.h>
#include <ctype.h>

long parse_long_or(const char *s, long fallback) {
    while (*s && isspace((unsigned char)*s)) s++;
    if (*s == '\\0') return fallback;
    char *end;
    long v = strtol(s, &end, 10);
    if (end == s) return fallback;
    return v;
}""",
     """#include <assert.h>
int main(void) {
    assert(parse_long_or("42", -1) == 42);
    assert(parse_long_or("  -7abc", 0) == -7);
    assert(parse_long_or("nope", 99) == 99);
    assert(parse_long_or("", 5) == 5);
    return 0;
}"""),
]


def main():
    kept, out_path = [], "gold_c.jsonl"
    for i, (instr, sol, tests) in enumerate(EXAMPLES, 1):
        src = f"{sol}\n\n{tests}\n"
        d = tempfile.mkdtemp()
        cpath, bpath = os.path.join(d, "p.c"), os.path.join(d, "p")
        try:
            with open(cpath, "w") as f:
                f.write(src)
            comp = subprocess.run(
                ["gcc", "-std=c11", "-O0", "-w", cpath, "-o", bpath, "-lm"],
                capture_output=True, timeout=15, text=True)
            if comp.returncode != 0:
                print(f"[{i:>2}] COMPILE_FAIL\n{comp.stderr[:400]}")
                continue
            run = subprocess.run([bpath], capture_output=True, timeout=15, text=True)
            if run.returncode != 0:
                print(f"[{i:>2}] RUNTIME_FAIL exit={run.returncode} {run.stderr[:200]}")
                continue
        finally:
            shutil.rmtree(d, ignore_errors=True)
        kept.append({
            "messages": [{"role": "user", "content": instr},
                         {"role": "assistant", "content": sol}],
            "lang": "C", "task": "implement a self-contained function",
            "difficulty": "intermediate", "tests": tests})
        print(f"[{i:>2}] PASS")
    with open(out_path, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(kept)}/{len(EXAMPLES)} verified gold C rows -> {out_path}")


if __name__ == "__main__":
    main()
