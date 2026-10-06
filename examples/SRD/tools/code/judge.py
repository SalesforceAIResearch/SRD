"""LiveCodeBench-style code judge for SDPO_ReAct.

Debug from repo root (needs the sandbox up: bash tools/run_sandbox.sh):
    python -m examples.SRD.tools.code.judge --demo
"""

import json

from miles.utils.http_utils import post

SANDBOX_URL = None  # set lazily from env so imports don't require it


def _sandbox_url() -> str:
    import os

    return os.environ.get("SDPO_REACT_SANDBOX_URL", "http://127.0.0.1:8420")


# Per-test wall-clock budget inside the harness (the sandbox's own /execute
# timeout must be >= n_tests * this; we pass a generous overall timeout below).
_PER_TEST_TIMEOUT = 6
_MAX_TESTS = 15  # cap tests actually run so one pathological problem can't hang the batch
# The sandbox refuses a per-call timeout above this (sandbox_server.py's
# MAX_TIMEOUT_SECONDS), so the tests are graded in chunks small enough that the
# harness's worst case (n * _PER_TEST_TIMEOUT) still fits in one call.
_MAX_SANDBOX_TIMEOUT = 60.0
_TESTS_PER_CALL = max(1, int((_MAX_SANDBOX_TIMEOUT - 5) / _PER_TEST_TIMEOUT))


def _extract_code_block(response: str) -> str:
    """Pull the candidate program out of a model response. Prefer a fenced
    ```python block; else the <answer> tag; else the whole response. The model
    is prompted (code system prompt) to put its final program in ```python."""
    import re

    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m[-1].strip()
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return response.strip()


def _build_harness(candidate: str, test_cases: list[dict]) -> str:
    """A self-contained program: run `candidate` (as a fresh subprocess-like
    exec with patched stdin) against each test case, print JSON {passed,total}.
    Runs entirely inside the sandbox via one /execute call."""
    tests = [{"input": t.get("input", ""), "output": t.get("output", "")} for t in test_cases[:_MAX_TESTS]]
    return (
        "import sys, io, json, contextlib, multiprocessing as mp\n"
        f"CANDIDATE = {candidate!r}\n"
        f"TESTS = {json.dumps(tests)}\n"
        f"PER_TEST_TIMEOUT = {_PER_TEST_TIMEOUT}\n"
        "def _run(inp, q):\n"
        "    buf = io.StringIO()\n"
        "    sys.stdin = io.StringIO(inp)\n"
        "    try:\n"
        "        with contextlib.redirect_stdout(buf):\n"
        "            exec(compile(CANDIDATE, '<candidate>', 'exec'), {'__name__': '__main__'})\n"
        "        q.put(buf.getvalue())\n"
        "    except Exception as e:\n"
        "        q.put('__ERROR__' + repr(e))\n"
        "def _norm(s):\n"
        "    return '\\n'.join(line.rstrip() for line in s.strip().splitlines())\n"
        "passed = 0\n"
        "for t in TESTS:\n"
        "    q = mp.Queue()\n"
        "    p = mp.Process(target=_run, args=(t['input'], q))\n"
        "    p.start(); p.join(PER_TEST_TIMEOUT)\n"
        "    if p.is_alive():\n"
        "        p.terminate(); p.join(); continue\n"
        "    try:\n"
        "        out = q.get_nowait()\n"
        "    except Exception:\n"
        "        continue\n"
        "    if isinstance(out, str) and not out.startswith('__ERROR__') and _norm(out) == _norm(t['output']):\n"
        "        passed += 1\n"
        "print(json.dumps({'passed': passed, 'total': len(TESTS)}))\n"
    )


# Imports injected in front of the candidate for FUNCTIONAL problems.
# LiveCodeBench's leetcode starter code annotates its signature
# (`nums: List[int]`), and annotations are evaluated when the method is defined,
# so `List` MUST already exist or the class body raises NameError before a
# single test runs. Same idea as LiveCodeBench's own call-based harness
# preamble; optional third-party names are guarded so a slimmer sandbox image
# (tools/docker/requirements.txt has no sortedcontainers) still works.
_FUNCTIONAL_PREAMBLE = (
    "import sys, math, re, json, string, bisect, heapq, itertools, functools, collections, random, copy\n"
    "from typing import List, Tuple, Dict, Set, FrozenSet, Optional, Any\n"
    "from collections import Counter, defaultdict, deque, OrderedDict\n"
    "from itertools import accumulate, permutations, combinations, product\n"
    "from functools import lru_cache, cache, cmp_to_key, reduce\n"
    "from heapq import heappush, heappop, heapify, nlargest, nsmallest\n"
    "from bisect import bisect_left, bisect_right, insort\n"
    "from math import inf, gcd, lcm, sqrt, ceil, floor, comb, perm, isqrt\n"
    "try:\n    from sortedcontainers import SortedList, SortedDict, SortedSet\nexcept Exception:\n    pass\n"
    "try:\n    import numpy as np\nexcept Exception:\n    pass\n"
)


def _build_functional_harness(candidate: str, test_cases: list[dict], fn_name: str) -> str:
    """LeetCode-style counterpart of `_build_harness`: exec the candidate, take
    `Solution().<fn_name>` (or a top-level `fn_name`, or -- if the dataset has
    no func_name -- the sole public method of Solution), call it once per test
    with the JSON-decoded argument lines, and structurally compare the RETURN
    VALUE against the JSON-decoded expected output. Same one-/execute-call,
    one-process-per-test, print-a-JSON-verdict shape as the stdin harness.

    Two deliberate leniencies, both because the graded candidate is whatever
    the model last RAN through code_interpreter (reward.py::_code_candidate),
    not a clean submission:
      - module-level exceptions are swallowed and the entrypoint is looked up
        anyway (the model's own driver/test code often sits after the class and
        may raise, e.g. on input() with no stdin -- that must not void a
        correct Solution);
      - the candidate's own prints are captured and discarded so they can't
        corrupt the verdict line.
    """
    tests = [{"input": t.get("input", ""), "output": t.get("output", "")} for t in test_cases[:_MAX_TESTS]]
    return (
        "import sys, io, json, contextlib, copy, multiprocessing as mp\n"
        f"CANDIDATE = {candidate!r}\n"
        f"PREAMBLE = {_FUNCTIONAL_PREAMBLE!r}\n"
        f"TESTS = {json.dumps(tests)}\n"
        f"FN_NAME = {fn_name!r}\n"
        f"PER_TEST_TIMEOUT = {_PER_TEST_TIMEOUT}\n"
        "def _resolve(ns):\n"
        "    sol = ns.get('Solution')\n"
        "    if isinstance(sol, type):\n"
        "        try:\n"
        "            inst = sol()\n"
        "        except Exception:\n"
        "            inst = None\n"
        "        if inst is not None:\n"
        "            f = getattr(inst, FN_NAME, None) if FN_NAME else None\n"
        "            if callable(f):\n"
        "                return f\n"
        "            pub = [m for m in dir(inst) if not m.startswith('_') and callable(getattr(inst, m, None))]\n"
        "            if len(pub) == 1:\n"
        "                return getattr(inst, pub[0])\n"
        "    f = ns.get(FN_NAME) if FN_NAME else None\n"
        "    return f if callable(f) else None\n"
        "def _eq(got, exp):\n"
        "    if isinstance(got, tuple):\n"
        "        got = list(got)\n"
        "    try:\n"
        "        if got == exp:\n"
        "            return True\n"
        "    except Exception:\n"
        "        pass\n"
        "    if isinstance(got, bool) or isinstance(exp, bool):\n"
        "        return False\n"
        "    if isinstance(got, (int, float)) and isinstance(exp, (int, float)):\n"
        "        return abs(got - exp) <= 1e-6 * max(1.0, abs(exp))\n"
        "    if isinstance(got, (list, tuple)) and isinstance(exp, list) and len(got) == len(exp):\n"
        "        return all(_eq(a, b) for a, b in zip(got, exp))\n"
        "    return False\n"
        "def _run(inp, exp_raw, q):\n"
        "    buf = io.StringIO()\n"
        "    try:\n"
        "        args = [json.loads(l) for l in inp.split('\\n') if l.strip() != '']\n"
        "        exp = json.loads(exp_raw)\n"
        "    except Exception as e:\n"
        "        q.put('__ERROR__bad_test:' + repr(e)); return\n"
        "    ns = {'__name__': '__candidate__'}\n"
        "    try:\n"
        "        with contextlib.redirect_stdout(buf):\n"
        "            exec(compile(PREAMBLE + CANDIDATE, '<candidate>', 'exec'), ns)\n"
        "    except Exception:\n"
        "        pass\n"
        "    fn = _resolve(ns)\n"
        "    if fn is None:\n"
        "        q.put('__ERROR__no_entrypoint'); return\n"
        "    try:\n"
        "        with contextlib.redirect_stdout(buf):\n"
        "            got = fn(*copy.deepcopy(args))\n"
        "    except Exception as e:\n"
        "        q.put('__ERROR__' + repr(e)); return\n"
        "    try:\n"
        "        q.put(bool(_eq(got, exp)))\n"
        "    except Exception as e:\n"
        "        q.put('__ERROR__cmp:' + repr(e))\n"
        "passed = 0\n"
        "for t in TESTS:\n"
        "    q = mp.Queue()\n"
        "    p = mp.Process(target=_run, args=(t['input'], t['output'], q))\n"
        "    p.start(); p.join(PER_TEST_TIMEOUT)\n"
        "    if p.is_alive():\n"
        "        p.terminate(); p.join(); continue\n"
        "    try:\n"
        "        ok = q.get_nowait()\n"
        "    except Exception:\n"
        "        continue\n"
        "    if ok is True:\n"
        "        passed += 1\n"
        "print(json.dumps({'passed': passed, 'total': len(TESTS)}))\n"
    )


def _testtype(test_cases: list[dict]) -> str:
    """"stdin" | "functional" for a test-case list. First non-empty testtype
    wins; absent -> stdin (older lists have no field and are all stdin)."""
    for t in test_cases:
        tt = str(t.get("testtype") or "").strip().lower()
        if tt:
            return tt
    return "stdin"


def _fn_name(test_cases: list[dict]) -> str:
    """The functional entrypoint name carried on the test cases (LiveCodeBench's
    metadata.func_name, stamped per test by build_code_data.py). Empty when the
    dataset didn't have one -- the harness then falls back to Solution's sole
    public method."""
    for t in test_cases:
        n = t.get("fn_name") or t.get("func_name") or ""
        if n:
            return str(n)
    return ""


async def _run_harness(harness: str, n_tests: int) -> tuple[int, int] | None:
    """Run one harness program in the sandbox; return (passed, total) or None on
    any sandbox/timeout/parse failure."""
    timeout = min(_MAX_SANDBOX_TIMEOUT, _PER_TEST_TIMEOUT * n_tests + 5)
    try:
        payload = await post(
            f"{_sandbox_url()}/execute",
            {"code": harness, "timeout": timeout},
            max_retries=3,
            action="post",
        )
    except Exception:
        return None
    if payload.get("timed_out"):
        return None
    stdout = (payload.get("stdout") or "").strip()
    # the harness prints exactly one JSON line last
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                v = json.loads(line)
                total = int(v.get("total", 0))
                return (int(v.get("passed", 0)), total) if total else None
            except Exception:
                return None
    return None


async def grade_code(response: str, test_cases: list[dict]) -> float:
    """Fraction of test cases the response's program passes, in [0,1]. Returns
    0.0 on any sandbox/parse failure (a broken submission is simply wrong).

    The tests are split into chunks of `_TESTS_PER_CALL` and graded with one
    /execute call each: the sandbox caps a single call at 60s
    (sandbox_server.py::MAX_TIMEOUT_SECONDS) while the harness's own worst case
    is n_tests * _PER_TEST_TIMEOUT, so a single call over ~9 tests could be
    killed mid-run and score a fully correct submission 0."""
    if not test_cases:
        return 0.0
    candidate = _extract_code_block(response)
    if not candidate.strip():
        return 0.0
    tests = test_cases[:_MAX_TESTS]
    functional = _testtype(tests) == "functional"
    fn_name = _fn_name(tests) if functional else ""
    passed = 0
    for i in range(0, len(tests), _TESTS_PER_CALL):
        chunk = tests[i : i + _TESTS_PER_CALL]
        if functional:
            harness = _build_functional_harness(candidate, chunk, fn_name)
        else:
            harness = _build_harness(candidate, chunk)
        got = await _run_harness(harness, len(chunk))
        if got is None:
            return 0.0
        passed += got[0]
    return passed / len(tests)


def main() -> None:
    import argparse
    import asyncio

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.parse_args()

    # http_utils.post() uses the process-wide httpx client the training runtime
    # builds in init_http_client(args); standalone there is no runtime, so every
    # call would fail on a None client and silently grade 0.0. Build a plain one.
    import httpx

    from miles.utils import http_utils

    if http_utils._http_client is None:
        http_utils._http_client = httpx.AsyncClient(timeout=httpx.Timeout(None))

    # toy stdin problem: read two ints, print their sum
    good = "```python\na,b=map(int,input().split())\nprint(a+b)\n```"
    bad = "```python\na,b=map(int,input().split())\nprint(a*b)\n```"
    tests = [{"input": "2 3\n", "output": "5"}, {"input": "10 20\n", "output": "30"}]
    print("stdin good ->", asyncio.run(grade_code(good, tests)), "(expect 1.0)")
    print("stdin bad  ->", asyncio.run(grade_code(bad, tests)), "(expect 0.0)")

    # toy functional problem: leetcode two-sum-ish `Solution.pairSum(nums, k)`.
    # The "good" candidate is deliberately shaped like what the model actually
    # runs through code_interpreter (List[int] annotation + its own driver
    # print), to exercise the preamble and the swallowed-module-level-code path.
    fgood = (
        "```python\nclass Solution:\n"
        "    def pairSum(self, nums: List[int], k: int) -> List[int]:\n"
        "        seen = {}\n"
        "        for i, v in enumerate(nums):\n"
        "            if k - v in seen:\n"
        "                return [seen[k - v], i]\n"
        "            seen[v] = i\n"
        "        return []\n"
        "print(Solution().pairSum([2, 7, 11, 15], 9))\n```"
    )
    fbad = (
        "```python\nclass Solution:\n"
        "    def pairSum(self, nums: List[int], k: int) -> List[int]:\n"
        "        return [0, 0]\n```"
    )
    ftests = [
        {"input": "[2, 7, 11, 15]\n9", "output": "[0, 1]", "testtype": "functional", "fn_name": "pairSum"},
        {"input": "[3, 2, 4]\n6", "output": "[1, 2]", "testtype": "functional", "fn_name": "pairSum"},
    ]
    print("func  good ->", asyncio.run(grade_code(fgood, ftests)), "(expect 1.0)")
    print("func  bad  ->", asyncio.run(grade_code(fbad, ftests)), "(expect 0.0)")


if __name__ == "__main__":
    main()
