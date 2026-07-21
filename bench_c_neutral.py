"""Independent C generalization check: novel, plainly-phrased C problems that are
NOT from our teacher and NOT in training — a home-grown but training-disjoint set
to see whether the v3 C gains hold on unfamiliar problems (anti-benchmaxxing).

Each item has a reference solution used ONLY to validate that the hidden test is
correct (gcc compile+run at load time); at eval time the model sees ONLY the
instruction, and we compile model_code + test. Same no-main/assert convention as
our eval so it's scorable. Run the SAME set on v3 and base; compare the gap.

Serve a model on :8091, then: MODEL_TAG=v3 .venv/bin/python bench_c_neutral.py
"""
import os, re, json, subprocess, tempfile, shutil
from openai import OpenAI
import config as C

TAG = os.environ.get("MODEL_TAG", "model")
client = OpenAI(base_url=C.TEACHER_BASE_URL, api_key=C.TEACHER_API_KEY)
HINT = (" Provide ONLY the implementation (functions/types and any needed "
        "#includes); do NOT write a main() function.")

# (instruction, reference_solution, test-main-assert-only)
PROBLEMS = [
    ("Write `int hamming(const char *a, const char *b)` that returns the number of "
     "positions where the two equal-length strings differ.",
     "int hamming(const char*a,const char*b){int n=0;while(*a){if(*a!=*b)n++;a++;b++;}return n;}",
     "#include <assert.h>\nint main(void){assert(hamming(\"karolin\",\"kathrin\")==3);"
     "assert(hamming(\"abc\",\"abc\")==0);assert(hamming(\"\",\"\")==0);return 0;}"),

    ("Write `int is_armstrong(int n)` that returns 1 if n (a non-negative integer) "
     "equals the sum of its own digits each raised to the power of the number of "
     "digits, else 0.",
     "#include <math.h>\nint is_armstrong(int n){int d=0,t=n;while(t){d++;t/=10;}"
     "int s=0;t=n;while(t){int x=t%10;s+=(int)(pow(x,d)+0.5);t/=10;}return s==n;}",
     "#include <assert.h>\nint main(void){assert(is_armstrong(153)==1);"
     "assert(is_armstrong(9474)==1);assert(is_armstrong(10)==0);assert(is_armstrong(0)==1);return 0;}"),

    ("Write `int collatz_steps(int n)` that returns how many steps it takes to reach "
     "1 from n>0 using the Collatz rule (n even -> n/2, n odd -> 3n+1).",
     "int collatz_steps(int n){int s=0;while(n!=1){n=(n%2==0)?n/2:3*n+1;s++;}return s;}",
     "#include <assert.h>\nint main(void){assert(collatz_steps(1)==0);"
     "assert(collatz_steps(6)==8);assert(collatz_steps(27)==111);return 0;}"),

    ("Write `int balanced_brackets(const char *s)` that returns 1 if the brackets "
     "(), [], {} in s are correctly matched and nested, else 0. Non-bracket "
     "characters are ignored.",
     "#include <string.h>\nint balanced_brackets(const char*s){char st[256];int t=0;"
     "for(;*s;s++){char c=*s;if(c=='('||c=='['||c=='{')st[t++]=c;"
     "else if(c==')'||c==']'||c=='}'){if(t==0)return 0;char o=st[--t];"
     "if((c==')'&&o!='(')||(c==']'&&o!='[')||(c=='}'&&o!='{'))return 0;}}return t==0;}",
     "#include <assert.h>\nint main(void){assert(balanced_brackets(\"{[()]}\")==1);"
     "assert(balanced_brackets(\"([)]\")==0);assert(balanced_brackets(\"a(b)c\")==1);"
     "assert(balanced_brackets(\"(\")==0);return 0;}"),

    ("Write `void rotate_left(int *a, int n, int k)` that rotates the array of n "
     "ints left by k positions in place (k may exceed n).",
     "void rotate_left(int*a,int n,int k){if(n==0)return;k%=n;int tmp[1024];"
     "for(int i=0;i<n;i++)tmp[i]=a[(i+k)%n];for(int i=0;i<n;i++)a[i]=tmp[i];}",
     "#include <assert.h>\nint main(void){int a[]={1,2,3,4,5};rotate_left(a,5,2);"
     "int e[]={3,4,5,1,2};for(int i=0;i<5;i++)assert(a[i]==e[i]);"
     "int b[]={1,2,3};rotate_left(b,3,4);int f[]={2,3,1};"
     "for(int i=0;i<3;i++)assert(b[i]==f[i]);return 0;}"),

    ("Write `int gcd_array(const int *a, int n)` that returns the greatest common "
     "divisor of n positive integers.",
     "int gcd_array(const int*a,int n){int g=a[0];for(int i=1;i<n;i++){int x=g,y=a[i];"
     "while(y){int t=x%y;x=y;y=t;}g=x;}return g;}",
     "#include <assert.h>\nint main(void){int a[]={12,18,24};assert(gcd_array(a,3)==6);"
     "int b[]={7,13};assert(gcd_array(b,2)==1);int c[]={100};assert(gcd_array(c,1)==100);return 0;}"),

    ("Write `int count_words(const char *s)` that returns the number of "
     "whitespace-separated words in s.",
     "#include <ctype.h>\nint count_words(const char*s){int n=0,in=0;for(;*s;s++){"
     "if(isspace((unsigned char)*s))in=0;else if(!in){in=1;n++;}}return n;}",
     "#include <assert.h>\nint main(void){assert(count_words(\"hello world\")==2);"
     "assert(count_words(\"  a  b  c \")==3);assert(count_words(\"\")==0);"
     "assert(count_words(\"one\")==1);return 0;}"),

    ("Write `long factorial(int n)` returning n! for 0<=n<=20 as a long.",
     "long factorial(int n){long r=1;for(int i=2;i<=n;i++)r*=i;return r;}",
     "#include <assert.h>\nint main(void){assert(factorial(0)==1);assert(factorial(5)==120);"
     "assert(factorial(10)==3628800L);return 0;}"),

    ("Write `int run_length_encode(const char *in, char *out)` that RLE-encodes in "
     "into out as <char><count> pairs (e.g. \"aaab\" -> \"a3b1\"), returns the number "
     "of characters written to out (out is null-terminated too).",
     "#include <stdio.h>\n#include <string.h>\nint run_length_encode(const char*in,char*out){"
     "int w=0;int i=0;while(in[i]){char c=in[i];int cnt=1;while(in[i+1]==c){cnt++;i++;}"
     "w+=sprintf(out+w,\"%c%d\",c,cnt);i++;}out[w]='\\0';return w;}",
     "#include <assert.h>\n#include <string.h>\nint main(void){char o[128];"
     "int w=run_length_encode(\"aaab\",o);assert(strcmp(o,\"a3b1\")==0);assert(w==4);"
     "run_length_encode(\"\",o);assert(strcmp(o,\"\")==0);return 0;}"),

    ("Write `int is_prime(int n)` returning 1 if n is prime else 0.",
     "int is_prime(int n){if(n<2)return 0;for(int i=2;(long)i*i<=n;i++)if(n%i==0)return 0;return 1;}",
     "#include <assert.h>\nint main(void){assert(is_prime(2));assert(is_prime(29));"
     "assert(!is_prime(1));assert(!is_prime(100));return 0;}"),

    ("Write `void merge_sorted(const int *a, int na, const int *b, int nb, int *out)` "
     "that merges two ascending arrays into out in ascending order.",
     "void merge_sorted(const int*a,int na,const int*b,int nb,int*out){int i=0,j=0,k=0;"
     "while(i<na&&j<nb)out[k++]=(a[i]<=b[j])?a[i++]:b[j++];"
     "while(i<na)out[k++]=a[i++];while(j<nb)out[k++]=b[j++];}",
     "#include <assert.h>\nint main(void){int a[]={1,3,5},b[]={2,4,6},o[6];"
     "merge_sorted(a,3,b,3,o);int e[]={1,2,3,4,5,6};for(int i=0;i<6;i++)assert(o[i]==e[i]);return 0;}"),

    ("Write `int digit_sum(long n)` returning the sum of the decimal digits of the "
     "absolute value of n.",
     "int digit_sum(long n){if(n<0)n=-n;int s=0;while(n){s+=n%10;n/=10;}return s;}",
     "#include <assert.h>\nint main(void){assert(digit_sum(1234)==10);assert(digit_sum(0)==0);"
     "assert(digit_sum(-99)==18);return 0;}"),

    ("Write `int fib(int n)` returning the n-th Fibonacci number, 0-indexed with "
     "fib(0)=0 and fib(1)=1.",
     "int fib(int n){if(n<2)return n;int a=0,b=1;for(int i=2;i<=n;i++){int c=a+b;a=b;b=c;}return b;}",
     "#include <assert.h>\nint main(void){assert(fib(0)==0);assert(fib(1)==1);"
     "assert(fib(10)==55);assert(fib(20)==6765);return 0;}"),

    ("Write `int sum_range(int a, int b)` returning the sum of all integers from a to "
     "b inclusive (assume a <= b).",
     "int sum_range(int a,int b){int s=0;for(int i=a;i<=b;i++)s+=i;return s;}",
     "#include <assert.h>\nint main(void){assert(sum_range(1,5)==15);assert(sum_range(3,3)==3);"
     "assert(sum_range(-2,2)==0);return 0;}"),

    ("Write `void reverse_array(int *a, int n)` that reverses the n-element array in "
     "place.",
     "void reverse_array(int*a,int n){for(int i=0;i<n/2;i++){int t=a[i];a[i]=a[n-1-i];a[n-1-i]=t;}}",
     "#include <assert.h>\nint main(void){int a[]={1,2,3,4};reverse_array(a,4);"
     "int e[]={4,3,2,1};for(int i=0;i<4;i++)assert(a[i]==e[i]);"
     "int b[]={1,2,3};reverse_array(b,3);int f[]={3,2,1};for(int i=0;i<3;i++)assert(b[i]==f[i]);return 0;}"),

    ("Write `int max_subarray(const int *a, int n)` returning the largest sum of any "
     "contiguous non-empty subarray (Kadane's algorithm).",
     "int max_subarray(const int*a,int n){int best=a[0],cur=a[0];for(int i=1;i<n;i++){"
     "cur=a[i]>cur+a[i]?a[i]:cur+a[i];if(cur>best)best=cur;}return best;}",
     "#include <assert.h>\nint main(void){int a[]={-2,1,-3,4,-1,2,1,-5,4};assert(max_subarray(a,9)==6);"
     "int b[]={-1,-2,-3};assert(max_subarray(b,3)==-1);int c[]={5};assert(max_subarray(c,1)==5);return 0;}"),

    ("Write `int count_char(const char *s, char c)` returning how many times character "
     "c appears in the null-terminated string s.",
     "int count_char(const char*s,char c){int n=0;for(;*s;s++)if(*s==c)n++;return n;}",
     "#include <assert.h>\nint main(void){assert(count_char(\"banana\",'a')==3);"
     "assert(count_char(\"xyz\",'a')==0);assert(count_char(\"\",'a')==0);return 0;}"),

    ("Write `int is_palindrome(const char *s)` returning 1 if the string reads the "
     "same forwards and backwards, else 0. The empty string is a palindrome.",
     "#include <string.h>\nint is_palindrome(const char*s){int i=0,j=(int)strlen(s)-1;"
     "while(i<j){if(s[i]!=s[j])return 0;i++;j--;}return 1;}",
     "#include <assert.h>\nint main(void){assert(is_palindrome(\"racecar\")==1);"
     "assert(is_palindrome(\"hello\")==0);assert(is_palindrome(\"\")==1);assert(is_palindrome(\"a\")==1);return 0;}"),

    ("Write `int binary_search(const int *a, int n, int target)` that returns the index "
     "of target in the ascending sorted array a, or -1 if absent.",
     "int binary_search(const int*a,int n,int target){int lo=0,hi=n-1;while(lo<=hi){"
     "int mid=lo+(hi-lo)/2;if(a[mid]==target)return mid;if(a[mid]<target)lo=mid+1;else hi=mid-1;}return -1;}",
     "#include <assert.h>\nint main(void){int a[]={1,3,5,7,9};assert(binary_search(a,5,5)==2);"
     "assert(binary_search(a,5,1)==0);assert(binary_search(a,5,9)==4);assert(binary_search(a,5,4)==-1);return 0;}"),

    ("Write `long power_int(int base, int exp)` returning base raised to exp for "
     "exp >= 0, computed iteratively.",
     "long power_int(int base,int exp){long r=1;for(int i=0;i<exp;i++)r*=base;return r;}",
     "#include <assert.h>\nint main(void){assert(power_int(2,10)==1024L);assert(power_int(3,0)==1L);"
     "assert(power_int(5,3)==125L);return 0;}"),

    ("Write `int str_to_int(const char *s)` that parses a string of decimal digits into "
     "a non-negative int, stopping at the first non-digit. Do not use library "
     "conversion functions.",
     "int str_to_int(const char*s){int r=0;while(*s>='0'&&*s<='9'){r=r*10+(*s-'0');s++;}return r;}",
     "#include <assert.h>\nint main(void){assert(str_to_int(\"123\")==123);assert(str_to_int(\"0\")==0);"
     "assert(str_to_int(\"42abc\")==42);return 0;}"),

    ("Write `int digital_root(int n)` that repeatedly sums the decimal digits of "
     "n >= 0 until a single digit remains, and returns it.",
     "int digital_root(int n){while(n>=10){int s=0;while(n){s+=n%10;n/=10;}n=s;}return n;}",
     "#include <assert.h>\nint main(void){assert(digital_root(942)==6);assert(digital_root(0)==0);"
     "assert(digital_root(9)==9);assert(digital_root(12345)==6);return 0;}"),

    ("Write `void bubble_sort(int *a, int n)` that sorts the n-element array into "
     "ascending order in place.",
     "void bubble_sort(int*a,int n){for(int i=0;i<n-1;i++)for(int j=0;j<n-1-i;j++)"
     "if(a[j]>a[j+1]){int t=a[j];a[j]=a[j+1];a[j+1]=t;}}",
     "#include <assert.h>\nint main(void){int a[]={5,4,3,2,1};bubble_sort(a,5);"
     "for(int i=0;i<5;i++)assert(a[i]==i+1);int b[]={3,1,2};bubble_sort(b,3);"
     "assert(b[0]==1&&b[1]==2&&b[2]==3);return 0;}"),

    ("Write `int count_set_bits(unsigned int x)` returning the number of 1-bits in x.",
     "int count_set_bits(unsigned int x){int n=0;while(x){n+=(int)(x&1u);x>>=1;}return n;}",
     "#include <assert.h>\nint main(void){assert(count_set_bits(0u)==0);assert(count_set_bits(7u)==3);"
     "assert(count_set_bits(255u)==8);return 0;}"),

    ("Write `int min_of_array(const int *a, int n)` returning the smallest element of "
     "the n-element array (n >= 1).",
     "int min_of_array(const int*a,int n){int m=a[0];for(int i=1;i<n;i++)if(a[i]<m)m=a[i];return m;}",
     "#include <assert.h>\nint main(void){int a[]={3,1,2};assert(min_of_array(a,3)==1);"
     "int b[]={-5,0,5};assert(min_of_array(b,3)==-5);int c[]={7};assert(min_of_array(c,1)==7);return 0;}"),

    ("Write `double average(const int *a, int n)` returning the arithmetic mean of the "
     "n integers as a double (n >= 1).",
     "double average(const int*a,int n){long s=0;for(int i=0;i<n;i++)s+=a[i];return (double)s/n;}",
     "#include <assert.h>\n#include <math.h>\nint main(void){int a[]={1,2,3,4};"
     "assert(fabs(average(a,4)-2.5)<1e-9);int b[]={10};assert(fabs(average(b,1)-10.0)<1e-9);return 0;}"),

    ("Write `int nth_prime(int n)` returning the n-th prime number, 1-indexed so "
     "nth_prime(1) == 2.",
     "int nth_prime(int n){int count=0,cand=1;while(count<n){cand++;int p=1;"
     "for(int i=2;(long)i*i<=cand;i++)if(cand%i==0){p=0;break;}if(p)count++;}return cand;}",
     "#include <assert.h>\nint main(void){assert(nth_prime(1)==2);assert(nth_prime(5)==11);"
     "assert(nth_prime(10)==29);return 0;}"),

    ("Write `void swap_case(char *s)` that swaps the case of every alphabetic character "
     "in s in place, leaving other characters unchanged.",
     "#include <ctype.h>\nvoid swap_case(char*s){for(;*s;s++){unsigned char c=(unsigned char)*s;"
     "if(isupper(c))*s=(char)tolower(c);else if(islower(c))*s=(char)toupper(c);}}",
     "#include <assert.h>\n#include <string.h>\nint main(void){char a[]=\"Hello World 42\";"
     "swap_case(a);assert(strcmp(a,\"hELLO wORLD 42\")==0);return 0;}"),

    ("Write `int lcm(int a, int b)` returning the least common multiple of two positive "
     "integers.",
     "int lcm(int a,int b){int x=a,y=b;while(y){int t=x%y;x=y;y=t;}return a/x*b;}",
     "#include <assert.h>\nint main(void){assert(lcm(4,6)==12);assert(lcm(3,7)==21);"
     "assert(lcm(5,5)==5);return 0;}"),

    ("Write `int count_substrings(const char *s, const char *sub)` returning the number "
     "of non-overlapping occurrences of sub in s. If sub is empty, return 0.",
     "#include <string.h>\nint count_substrings(const char*s,const char*sub){"
     "size_t L=strlen(sub);if(L==0)return 0;int n=0;const char*p=s;"
     "while((p=strstr(p,sub))){n++;p+=L;}return n;}",
     "#include <assert.h>\nint main(void){assert(count_substrings(\"aaaa\",\"aa\")==2);"
     "assert(count_substrings(\"ababab\",\"ab\")==3);assert(count_substrings(\"hello\",\"xyz\")==0);return 0;}"),
]


def compile_run(sol, tests):
    src = f"{sol}\n\n{tests}\n"
    d = tempfile.mkdtemp()
    cp, bp = os.path.join(d, "p.c"), os.path.join(d, "p")
    try:
        open(cp, "w").write(src)
        c = subprocess.run(["gcc", "-std=c11", "-O0", "-w", cp, "-o", bp, "-lm"],
                           capture_output=True, timeout=15, text=True)
        if c.returncode != 0:
            return False
        r = subprocess.run([bp], capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def extract(t):
    m = re.search(r"```[a-zA-Z0-9+#]*\n(.*?)```", t, re.S)
    return m.group(1) if m else t


def main():
    # 1) self-check: every reference must pass its own test (guards the benchmark)
    for i, (instr, ref, test) in enumerate(PROBLEMS, 1):
        if not compile_run(ref, test):
            raise SystemExit(f"BENCH BUG: reference {i} fails its own test")
    print(f"benchmark valid: {len(PROBLEMS)} references pass their tests\n")

    # 2) score the served model (instruction only, no reference)
    npass = 0
    for i, (instr, ref, test) in enumerate(PROBLEMS, 1):
        r = client.chat.completions.create(model=TAG, temperature=0.2,
            messages=[{"role": "user", "content": instr + HINT}], max_tokens=1024)
        code = extract(r.choices[0].message.content)
        ok = compile_run(code, test)
        npass += ok
        print(f"[{i:>2}/{len(PROBLEMS)}] {'PASS' if ok else 'FAIL'}  "
              f"running={npass/i:.1%}", flush=True)
    print(f"\n=== neutral-C [{TAG}] pass@1 = {npass}/{len(PROBLEMS)} = "
          f"{npass/len(PROBLEMS):.1%} ===")
    os.makedirs("bench", exist_ok=True)
    json.dump({"tag": TAG, "n": len(PROBLEMS), "pass": npass},
              open(f"bench/cneutral_{TAG}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
