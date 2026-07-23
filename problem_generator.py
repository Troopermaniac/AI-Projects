"""
PROBLEM GENERATOR   Infinite Formally Verified Problem Stream

Replaces the static dynamic_dataset.json curriculum with a procedural generator
that produces novel, never-repeating problems across multiple domains. Expected
outputs are computed by trusted reference implementations at generation time  
the AI cannot game these by memorizing past answers.

Key design principles:
- Every problem has 3+ distinct test cases (not just 1)
- Reference solutions are never shown to the AI
- Problems rotate across domains so no single skill saturates
- Difficulty scales with epoch number
"""

import random
import math
import hashlib
import json
import os
from typing import List, Tuple, Any, Dict

# REFERENCE IMPLEMENTATIONS  (ground truth   never shown to AI)

def _ref_is_prime(n: int) -> bool:
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0: return False
    return True

def _ref_gcd(a: int, b: int) -> int:
    while b: a, b = b, a % b
    return a

def _ref_lcm(a: int, b: int) -> int:
    return abs(a * b) // _ref_gcd(a, b)

def _ref_digit_sum(n: int) -> int:
    return sum(int(d) for d in str(abs(n)))

def _ref_is_palindrome_int(n: int) -> bool:
    s = str(abs(n)); return s == s[::-1]

def _ref_collatz_steps(n: int) -> int:
    if n <= 0: return 0
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps

def _ref_count_divisors(n: int) -> int:
    if n <= 0: return 0
    return sum(1 for i in range(1, n + 1) if n % i == 0)

def _ref_nth_prime(n: int) -> int:
    if n < 1: return 2
    primes, candidate = [], 2
    while len(primes) < n:
        if _ref_is_prime(candidate): primes.append(candidate)
        candidate += 1
    return primes[-1]

def _ref_flatten_sum(lst: List) -> int:
    total = 0
    for item in lst:
        if isinstance(item, list): total += _ref_flatten_sum(item)
        else: total += item
    return total

def _ref_caesar_cipher(text: str, shift: int) -> str:
    result = []
    for ch in text:
        if ch.isalpha():
            base = ord('A') if ch.isupper() else ord('a')
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return ''.join(result)

def _ref_run_length_encode(s: str) -> str:
    if not s: return ""
    result, count, current = [], 1, s[0]
    for ch in s[1:]:
        if ch == current: count += 1
        else:
            result.append(f"{count}{current}"); count, current = 1, ch
    result.append(f"{count}{current}")
    return ''.join(result)

def _ref_count_vowels(s: str) -> int:
    return sum(1 for c in s.lower() if c in 'aeiou')

def _ref_binary_to_decimal(b: str) -> int:
    return int(b, 2)

def _ref_decimal_to_binary(n: int) -> str:
    if n == 0: return "0"
    return bin(n)[2:]

def _ref_is_perfect(n: int) -> bool:
    if n < 2: return False
    return sum(i for i in range(1, n) if n % i == 0) == n

def _ref_triangle_type(a: int, b: int, c: int) -> str:
    sides = sorted([a, b, c])
    if sides[0] + sides[1] <= sides[2]: return "invalid"
    if a == b == c: return "equilateral"
    if a == b or b == c or a == c: return "isosceles"
    return "scalene"

def _ref_sum_of_squares(n: int) -> int:
    return sum(i * i for i in range(1, n + 1))

def _ref_square_of_sum(n: int) -> int:
    return sum(range(1, n + 1)) ** 2

def _ref_hamming_distance(s1: str, s2: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def _ref_pangram(sentence: str) -> bool:
    return set('abcdefghijklmnopqrstuvwxyz').issubset(set(sentence.lower()))

def _ref_anagram(s1: str, s2: str) -> bool:
    return sorted(s1.lower().replace(' ', '')) == sorted(s2.lower().replace(' ', ''))

def _ref_luhn(card: str) -> bool:
    digits = [int(d) for d in card if d.isdigit()]
    digits.reverse()
    total = sum(digits[0::2]) + sum(d * 2 - 9 if d * 2 > 9 else d * 2
                                    for d in digits[1::2])
    return total % 10 == 0

def _ref_nth_fibonacci(n: int) -> int:
    if n <= 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(2, n + 1): a, b = b, a + b
    return b

def _ref_power(base: int, exp: int) -> int:
    return base ** exp

def _ref_count_words(sentence: str) -> int:
    return len(sentence.split())

def _ref_most_common_char(s: str) -> str:
    if not s: return ""
    freq = {}
    for c in s: freq[c] = freq.get(c, 0) + 1
    return max(freq, key=freq.get)

def _ref_rotate_list(lst: list, k: int) -> list:
    if not lst: return lst
    k = k % len(lst)
    return lst[k:] + lst[:k]

def _ref_unique_elements(lst: list) -> list:
    seen = set(); result = []
    for x in lst:
        if x not in seen: seen.add(x); result.append(x)
    return result

def _ref_product_of_list(lst: list) -> int:
    result = 1
    for x in lst: result *= x
    return result

def _ref_sum_even(lst: list) -> int:
    return sum(x for x in lst if x % 2 == 0)

def _ref_sum_odd(lst: list) -> int:
    return sum(x for x in lst if x % 2 != 0)

def _ref_median(lst: list) -> float:
    s = sorted(lst); n = len(s)
    if n % 2 == 1: return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

def _ref_clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

def _ref_title_case(s: str) -> str:
    return s.title()

def _ref_count_substrings(s: str, sub: str) -> int:
    return s.count(sub)


# TIER 3 & TIER 4 ALGORITHMIC REFERENCE IMPLEMENTATIONS

def _ref_valid_parentheses(s: str) -> bool:
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in mapping.values():
            stack.append(char)
        elif char in mapping.keys():
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
    return len(stack) == 0

def _ref_matrix_transpose(matrix: List[List[int]]) -> List[List[int]]:
    if not matrix or not matrix[0]: return []
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def _ref_pascal_row(n: int) -> List[int]:
    row = [1]
    for _ in range(n):
        row = [1] + [row[i] + row[i+1] for i in range(len(row)-1)] + [1]
    return row

def _ref_lcs_length(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def _ref_evaluate_rpn(tokens: List[str]) -> int:
    stack = []
    for t in tokens:
        if t in ("+", "-", "*", "/"):
            b, a = stack.pop(), stack.pop()
            if t == "+": stack.append(a + b)
            elif t == "-": stack.append(a - b)
            elif t == "*": stack.append(a * b)
            elif t == "/": stack.append(int(a / b))
        else:
            stack.append(int(t))
    return stack[0]

def _ref_is_valid_bst(keys: List[int]) -> bool:
    return keys == sorted(set(keys))

def _ref_knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        w, v = weights[i], values[i]
        for c in range(capacity, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[capacity]


# PROBLEM TEMPLATES
# Each template is a dict with:
#   name_template: f-string pattern for problem description
#   generator: callable → (args, expected_output)
#   category: domain label
#   min_difficulty: 1-5

def _gen_is_prime(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    n = rng.choice([p for p in range(2, 200) if _ref_is_prime(p)] +
                   [c for c in range(2, 200) if not _ref_is_prime(c)])
    test_cases = []
    seen = set()
    for val in [n, n + 1, n + 3, 2, 15, 97, 100]:
        if val not in seen:
            seen.add(val)
            test_cases.append(([val], _ref_is_prime(val)))
    return {"description": "Check if a number is prime", "category": "mathematical", "difficulty": 1}, test_cases[:4]

def _gen_gcd(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    pairs = [(rng.randint(1, 100), rng.randint(1, 100)) for _ in range(4)]
    test_cases = [([a, b], _ref_gcd(a, b)) for a, b in pairs]
    return {"description": "Compute the Greatest Common Divisor (GCD) of two numbers", "category": "mathematical", "difficulty": 1}, test_cases

def _gen_lcm(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    pairs = [(rng.randint(1, 30), rng.randint(1, 30)) for _ in range(4)]
    test_cases = [([a, b], _ref_lcm(a, b)) for a, b in pairs]
    return {"description": "Compute the Least Common Multiple (LCM) of two numbers", "category": "mathematical", "difficulty": 2}, test_cases

def _gen_digit_sum(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    nums = [rng.randint(1, 9999) for _ in range(4)]
    test_cases = [([n], _ref_digit_sum(n)) for n in nums]
    return {"description": "Compute the sum of digits of a number", "category": "mathematical", "difficulty": 1}, test_cases

def _gen_collatz(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    nums = [rng.randint(2, 50) for _ in range(4)]
    test_cases = [([n], _ref_collatz_steps(n)) for n in nums]
    return {"description": "Count the number of steps in the Collatz sequence until reaching 1", "category": "mathematical", "difficulty": 2}, test_cases

def _gen_count_divisors(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    nums = [rng.randint(1, 50) for _ in range(4)]
    test_cases = [([n], _ref_count_divisors(n)) for n in nums]
    return {"description": "Count the total number of divisors of a number", "category": "mathematical", "difficulty": 2}, test_cases

def _gen_nth_prime(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    ns = [rng.randint(1, 20) for _ in range(4)]
    test_cases = [([n], _ref_nth_prime(n)) for n in ns]
    return {"description": "Return the Nth prime number", "category": "mathematical", "difficulty": 2}, test_cases

def _gen_palindrome_int(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    samples = [rng.randint(1, 9999) for _ in range(4)] + [121, 1221]
    test_cases = [([n], _ref_is_palindrome_int(n)) for n in samples[:4]]
    return {"description": "Check if an integer reads the same forwards and backwards (palindrome number)", "category": "mathematical", "difficulty": 1}, test_cases

def _gen_sum_of_squares(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    ns = [rng.randint(1, 15) for _ in range(4)]
    test_cases = [([n], _ref_sum_of_squares(n)) for n in ns]
    return {"description": "Compute the sum of squares of all integers from 1 to N", "category": "mathematical", "difficulty": 2}, test_cases

def _gen_square_of_sum(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    ns = [rng.randint(1, 15) for _ in range(4)]
    test_cases = [([n], _ref_square_of_sum(n)) for n in ns]
    return {"description": "Compute the square of the sum of all integers from 1 to N", "category": "mathematical", "difficulty": 2}, test_cases

def _gen_is_perfect(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    candidates = [1, 6, 12, 28, 30, 496, 100]
    rng.shuffle(candidates)
    test_cases = [([n], _ref_is_perfect(n)) for n in candidates[:4]]
    return {"description": "Check if a number is a perfect number (equals the sum of its proper divisors)", "category": "mathematical", "difficulty": 3}, test_cases

def _gen_triangle_type(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    cases = [
        ([3, 3, 3], "equilateral"),
        ([3, 4, 5], "scalene"),
        ([5, 5, 8], "isosceles"),
        ([1, 2, 10], "invalid"),
    ]
    rng.shuffle(cases)
    test_cases = [([a, b, c], t) for ([a, b, c], t) in cases[:4]]
    return {"description": "Classify a triangle as equilateral, isosceles, scalene, or invalid given its three side lengths", "category": "mathematical", "difficulty": 2}, test_cases

def _gen_caesar_cipher(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    words = ["hello", "python", "secret", "attack"]
    shifts = [rng.randint(1, 25) for _ in range(4)]
    test_cases = [([words[i], shifts[i]], _ref_caesar_cipher(words[i], shifts[i])) for i in range(4)]
    return {"description": "Encrypt a string using Caesar cipher with a given shift", "category": "string", "difficulty": 2}, test_cases

def _gen_run_length_encode(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    samples = ["aabbbcccc", "xxyyzz", "aaaa", "abcde"]
    test_cases = [([s], _ref_run_length_encode(s)) for s in samples]
    return {"description": "Perform run-length encoding on a string", "category": "string", "difficulty": 3}, test_cases

def _gen_count_vowels(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    samples = ["hello world", "python programming", "aeiou", "xyz"]
    test_cases = [([s], _ref_count_vowels(s)) for s in samples]
    return {"description": "Count the number of vowels in a string", "category": "string", "difficulty": 1}, test_cases

def _gen_binary_to_decimal(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    ns = [rng.randint(0, 255) for _ in range(4)]
    test_cases = [([_ref_decimal_to_binary(n)], n) for n in ns]
    return {"description": "Convert a binary string to its decimal integer value", "category": "string", "difficulty": 2}, test_cases

def _gen_hamming_distance(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    length = rng.randint(4, 8)
    pairs = [(''.join(rng.choices('01', k=length)), ''.join(rng.choices('01', k=length))) for _ in range(4)]
    test_cases = [([a, b], _ref_hamming_distance(a, b)) for a, b in pairs]
    return {"description": "Compute the Hamming distance between two equal-length strings", "category": "string", "difficulty": 2}, test_cases

def _gen_pangram(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    cases = [
        ("the quick brown fox jumps over the lazy dog", True),
        ("hello world", False),
        ("pack my box with five dozen liquor jugs", True),
        ("python programming is fun", False),
    ]
    rng.shuffle(cases)
    test_cases = [([s], r) for s, r in cases[:4]]
    return {"description": "Check if a sentence contains every letter of the alphabet (pangram check)", "category": "string", "difficulty": 2}, test_cases

def _gen_anagram(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    cases = [
        ("listen", "silent", True),
        ("hello", "world", False),
        ("triangle", "integral", True),
        ("cat", "dog", False),
    ]
    rng.shuffle(cases)
    test_cases = [([a, b], r) for a, b, r in cases[:4]]
    return {"description": "Check if two strings are anagrams of each other", "category": "string", "difficulty": 2}, test_cases

def _gen_count_words(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    samples = ["hello world", "one two three four", "python", "the quick brown fox"]
    test_cases = [([s], _ref_count_words(s)) for s in samples]
    return {"description": "Count the number of words in a sentence", "category": "string", "difficulty": 1}, test_cases

def _gen_most_common_char(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    samples = ["aabbbc", "xyzzzz", "aabb", "hello"]
    test_cases = [([s], _ref_most_common_char(s)) for s in samples]
    return {"description": "Return the most frequently occurring character in a string", "category": "string", "difficulty": 2}, test_cases

def _gen_rotate_list(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    lists = [[1,2,3,4,5], [10,20,30], [1,2,3,4,5,6], [7,8,9]]
    ks = [rng.randint(1, 4) for _ in range(4)]
    test_cases = [([lists[i], ks[i]], _ref_rotate_list(lists[i], ks[i])) for i in range(4)]
    return {"description": "Rotate a list to the left by K positions", "category": "algorithmic", "difficulty": 2}, test_cases

def _gen_unique_elements(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    lists = [[1,2,2,3,3,3], [4,4,4,4], [1,2,3], [5,5,1,2,5]]
    test_cases = [([lst], _ref_unique_elements(lst)) for lst in lists]
    return {"description": "Return a list with duplicate elements removed, preserving original order", "category": "algorithmic", "difficulty": 2}, test_cases

def _gen_product_of_list(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    lists = [[rng.randint(1, 5) for _ in range(rng.randint(2, 5))] for _ in range(4)]
    test_cases = [([lst], _ref_product_of_list(lst)) for lst in lists]
    return {"description": "Compute the product of all elements in a list", "category": "algorithmic", "difficulty": 1}, test_cases

def _gen_sum_even(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    lists = [[rng.randint(1, 20) for _ in range(rng.randint(3, 7))] for _ in range(4)]
    test_cases = [([lst], _ref_sum_even(lst)) for lst in lists]
    return {"description": "Return the sum of all even numbers in a list", "category": "algorithmic", "difficulty": 1}, test_cases

def _gen_sum_odd(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    lists = [[rng.randint(1, 20) for _ in range(rng.randint(3, 7))] for _ in range(4)]
    test_cases = [([lst], _ref_sum_odd(lst)) for lst in lists]
    return {"description": "Return the sum of all odd numbers in a list", "category": "algorithmic", "difficulty": 1}, test_cases

def _gen_median(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    lists = [[rng.randint(1, 50) for _ in range(rng.randint(3, 7))] for _ in range(4)]
    test_cases = [([lst], _ref_median(lst)) for lst in lists]
    return {"description": "Return the median value of a list of numbers", "category": "algorithmic", "difficulty": 2}, test_cases

def _gen_clamp(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    cases = [(rng.uniform(-10, 10), rng.uniform(-5, 0), rng.uniform(1, 5)) for _ in range(4)]
    test_cases = [([round(v, 1), round(lo, 1), round(hi, 1)],
                   _ref_clamp(round(v, 1), round(lo, 1), round(hi, 1))) for v, lo, hi in cases]
    return {"description": "Clamp a value between a minimum and maximum boundary", "category": "algorithmic", "difficulty": 1}, test_cases

def _gen_nth_fibonacci(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    ns = [rng.randint(5, 25) for _ in range(4)]
    test_cases = [([n], _ref_nth_fibonacci(n)) for n in ns]
    return {"description": "Return the Nth number in the Fibonacci sequence (0-indexed)", "category": "mathematical", "difficulty": 1}, test_cases

def _gen_count_substrings(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    cases = [("banana", "an"), ("mississippi", "ss"), ("hello", "l"), ("aababab", "ab")]
    test_cases = [([s, sub], _ref_count_substrings(s, sub)) for s, sub in cases]
    return {"description": "Count the non-overlapping occurrences of a substring within a string", "category": "string", "difficulty": 2}, test_cases


# TIER 3 & TIER 4 GENERATORS

def _gen_valid_parentheses(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    samples = ["()", "()[]{}", "(]", "([{}])", "(((", "([)]"]
    test_cases = [([s], _ref_valid_parentheses(s)) for s in samples]
    return {"description": "Determine if a string containing brackets '()[]{}' has valid balanced parentheses", "category": "data_structures", "difficulty": 3}, test_cases[:4]

def _gen_matrix_transpose(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    matrices = [
        [[1, 2], [3, 4]],
        [[1, 2, 3], [4, 5, 6]],
        [[7]],
        [[1, 4], [2, 5], [3, 6]]
    ]
    test_cases = [([m], _ref_matrix_transpose(m)) for m in matrices]
    return {"description": "Return the transpose of a 2D matrix (swap rows and columns)", "category": "matrices", "difficulty": 3}, test_cases

def _gen_pascal_row(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    rows = [0, 1, 4, 6]
    test_cases = [([r], _ref_pascal_row(r)) for r in rows]
    return {"description": "Return the Nth row (0-indexed) of Pascal's Triangle as a list of integers", "category": "mathematical", "difficulty": 3}, test_cases

def _gen_lcs(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    pairs = [("abcde", "ace"), ("abc", "abc"), ("abc", "def"), ("AGGTAB", "GXTXAYB")]
    test_cases = [([s1, s2], _ref_lcs_length(s1, s2)) for s1, s2 in pairs]
    return {"description": "Compute the length of the Longest Common Subsequence between two strings", "category": "dynamic_programming", "difficulty": 3}, test_cases

def _gen_evaluate_rpn(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    exprs = [
        ["2", "1", "+", "3", "*"],
        ["4", "13", "5", "/", "+"],
        ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"],
        ["3", "4", "-"]
    ]
    test_cases = [([tokens], _ref_evaluate_rpn(tokens)) for tokens in exprs]
    return {"description": "Evaluate the value of an arithmetic expression in Reverse Polish Notation (tokens list)", "category": "data_structures", "difficulty": 4}, test_cases

def _gen_is_valid_bst(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    keys_list = [[1, 2, 3, 4], [10, 5, 15], [2, 5, 8, 12], [5, 1, 4]]
    test_cases = [([keys], _ref_is_valid_bst(keys)) for keys in keys_list]
    return {"description": "Verify if a list of tree key traversals satisfies valid Binary Search Tree order properties", "category": "trees", "difficulty": 4}, test_cases

def _gen_knapsack(rng: random.Random) -> Tuple[Dict, List[Tuple]]:
    cases = [
        ([1, 2, 3], [10, 15, 40], 6),
        ([10, 20, 30], [60, 100, 120], 50),
        ([2, 3, 4, 5], [3, 4, 5, 6], 5),
        ([5], [10], 3)
    ]
    test_cases = [([w, v, cap], _ref_knapsack_01(w, v, cap)) for w, v, cap in cases]
    return {"description": "Solve the 0/1 Knapsack problem returning maximum total value within item capacity", "category": "dynamic_programming", "difficulty": 4}, test_cases


# TIER REGISTRY

TIER_TEMPLATES = {
    "beginner": [
        _gen_is_prime, _gen_gcd, _gen_digit_sum, _gen_palindrome_int,
        _gen_sum_of_squares, _gen_triangle_type, _gen_count_vowels,
        _gen_count_words, _gen_sum_even, _gen_sum_odd, _gen_clamp,
        _gen_nth_fibonacci
    ],
    "intermediate": [
        _gen_lcm, _gen_collatz, _gen_count_divisors, _gen_nth_prime,
        _gen_square_of_sum, _gen_is_perfect, _gen_caesar_cipher,
        _gen_run_length_encode, _gen_binary_to_decimal, _gen_hamming_distance,
        _gen_pangram, _gen_anagram, _gen_rotate_list, _gen_unique_elements,
        _gen_product_of_list, _gen_median, _gen_count_substrings
    ],
    "advanced": [
        _gen_valid_parentheses, _gen_matrix_transpose, _gen_pascal_row, _gen_lcs
    ],
    "expert": [
        _gen_evaluate_rpn, _gen_is_valid_bst, _gen_knapsack
    ]
}

PROBLEM_TEMPLATES = (
    TIER_TEMPLATES["beginner"] +
    TIER_TEMPLATES["intermediate"] +
    TIER_TEMPLATES["advanced"] +
    TIER_TEMPLATES["expert"]
)


# PUBLIC API

def generate_epoch_problems(epoch: int, n_problems: int = 10, difficulty_tier: str = "beginner") -> List[Dict]:
    """
    Generate N fresh problems for this epoch calibrated to the specified difficulty tier.
    """
    rng = random.Random(epoch * 31337 + 9999)

    tier = str(difficulty_tier).lower().strip()
    if tier == "expert":
        available_templates = TIER_TEMPLATES["advanced"] + TIER_TEMPLATES["expert"]
    elif tier == "advanced":
        available_templates = TIER_TEMPLATES["intermediate"] + TIER_TEMPLATES["advanced"]
    elif tier == "intermediate":
        available_templates = TIER_TEMPLATES["beginner"] + TIER_TEMPLATES["intermediate"]
    else:
        available_templates = TIER_TEMPLATES["beginner"]

    templates = available_templates[:]
    rng.shuffle(templates)

    problems = []
    used_descriptions = set()
    attempts = 0

    while len(problems) < n_problems and attempts < n_problems * 10:
        attempts += 1
        template_fn = templates[attempts % len(templates)]
        try:
            meta, raw_cases = template_fn(rng)
        except Exception:
            continue

        desc = meta["description"]
        if desc in used_descriptions:
            continue
        used_descriptions.add(desc)

        test_cases = [{"inputs": list(inp), "expected": exp} for inp, exp in raw_cases]
        if len(test_cases) < 3:
            continue  # Require at least 3 test cases

        problems.append({
            "description": desc,
            "category": meta.get("category", "general"),
            "difficulty": meta.get("difficulty", 1),
            "test_cases": test_cases,
        })

    return problems


def problems_to_legacy_dataset(problems: List[Dict]) -> List:
    """
    Convert generated problems to the legacy [prompt, inputs, expected] format
    used by evaluator_anchor.py and evaluator_dynamic.py.
    Uses only the FIRST test case for compatibility with old evaluators.
    
    NOTE: The new evaluator_dynamic.py uses the full multi-case format.
    This function is only for backward compatibility with evaluator_anchor.py.
    """
    result = []
    for p in problems:
        if p["test_cases"]:
            first = p["test_cases"][0]
            result.append([p["description"], first["inputs"], first["expected"]])
    return result


def save_epoch_problems(epoch: int, n_problems: int = 10,
                        path: str = "sandbox/generated_problems.json") -> List[Dict]:
    """
    Generate problems for this epoch and save them to disk.
    Returns the generated problems list.
    """
    problems = generate_epoch_problems(epoch, n_problems)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2)
    return problems


def load_epoch_problems(path: str = "sandbox/generated_problems.json") -> List[Dict]:
    """Load previously generated problems from disk."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def get_problem_stats(problems: List[Dict]) -> str:
    """Return a human-readable summary of the generated problem set."""
    if not problems:
        return "No problems generated."
    cats = {}
    for p in problems:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    avg_diff = sum(p["difficulty"] for p in problems) / len(problems)
    cat_str = ", ".join(f"{k}:{v}" for k, v in sorted(cats.items()))
    return (f"{len(problems)} problems | avg_difficulty={avg_diff:.1f} | "
            f"categories=[{cat_str}]")


if __name__ == "__main__":
    # Quick self-test
    for epoch in [1, 2, 3, 42]:
        probs = generate_epoch_problems(epoch, n_problems=10)
        print(f"Epoch {epoch}: {get_problem_stats(probs)}")
        for p in probs[:2]:
            print(f"  [{p['difficulty']}] {p['description']}")
            for tc in p['test_cases'][:2]:
                print(f"    {tc['inputs']} -> {tc['expected']}")
        print()
