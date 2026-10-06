#!/usr/bin/env python3
"""Standalone smoke test for the SDPO_ReAct tool sidecars (no miles / no GPU).

Usage:
    # start the code sandbox first (idempotent):
    bash examples/SRD/tools/run_sandbox.sh
    # then:
    python examples/SRD/tools/test_tools_docker.py
    # options:
    python examples/SRD/tools/test_tools_docker.py \
        --sandbox-url http://127.0.0.1:8420 --search-url http://127.0.0.1:8000/retrieve
"""

import argparse
import json
import urllib.error
import urllib.request

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"


def _post(url: str, payload: dict, timeout: float = 90.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _execute(sandbox_url: str, code: str, timeout: float | None = None) -> dict:
    payload = {"code": code}
    if timeout is not None:
        payload["timeout"] = timeout
    return _post(f"{sandbox_url}/execute", payload)


class Runner:
    def __init__(self):
        self.n_pass = 0
        self.n_fail = 0
        self.n_skip = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        tag = PASS if ok else FAIL
        self.n_pass += ok
        self.n_fail += not ok
        print(f"[{tag}] {name}" + (f"  --  {detail}" if detail else ""))

    def skip(self, name: str, why: str) -> None:
        self.n_skip += 1
        print(f"[{SKIP}] {name}  --  {why}")

    def summary(self) -> int:
        print(f"\n{self.n_pass} passed, {self.n_fail} failed, {self.n_skip} skipped")
        return 1 if self.n_fail else 0


def test_sandbox(r: Runner, url: str) -> None:
    print(f"\n=== code_interpreter sandbox @ {url} ===")

    # reachability
    try:
        out = _execute(url, "print('hello')")
    except Exception as e:
        r.check("sandbox reachable", False, f"{e!r} -- is it up? run tools/run_sandbox.sh")
        return
    r.check("1. basic exec + stdout", out.get("stdout", "").strip() == "hello", repr(out.get("stdout")))

    out = _execute(url, "x = 6 * 7\nx")
    r.check("2. auto-print trailing bare expr", "42" in out.get("stdout", ""), repr(out.get("stdout")))

    out = _execute(url, "import sympy, numpy, scipy\nprint(sympy.sqrt(8))")
    r.check("3. sympy/numpy/scipy import", "2*sqrt(2)" in out.get("stdout", ""), repr(out.get("stdout")))

    out = _execute(url, "1/0")
    r.check("4. runtime error surfaced (sidecar alive)", bool(out.get("error")), repr(out.get("error"))[:80])
    # sidecar still alive after the error?
    out2 = _execute(url, "print('still alive')")
    r.check("4b. sidecar survives error", out2.get("stdout", "").strip() == "still alive")

    out = _execute(url, "while True:\n    pass", timeout=3)
    r.check("5. timeout enforced", bool(out.get("timed_out")), repr(out.get("timed_out")))

    mp_code = (
        "import multiprocessing as mp\n"
        "def f(q): q.put(123)\n"
        "q = mp.Queue(); p = mp.Process(target=f, args=(q,)); p.start(); p.join()\n"
        "print(q.get())\n"
    )
    out = _execute(url, mp_code, timeout=15)
    r.check("6. multiprocessing works (code_judge relies on it)", "123" in out.get("stdout", ""), repr(out.get("stdout")))

    # 7. full code_judge harness round-trip
    try:
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        from examples.SRD.tools.code.judge import _build_harness

        cand = "a,b=map(int,input().split())\nprint(a+b)"
        tests = [{"input": "2 3\n", "output": "5"}, {"input": "10 20\n", "output": "30"}]
        harness = _build_harness(cand, tests)
        out = _execute(url, harness, timeout=40)
        verdict = {}
        for line in reversed(out.get("stdout", "").splitlines()):
            if line.strip().startswith("{"):
                verdict = json.loads(line)
                break
        r.check("7. code_judge harness round-trip (2/2)", verdict.get("passed") == 2 and verdict.get("total") == 2, str(verdict))
    except Exception as e:
        r.check("7. code_judge harness round-trip", False, f"{e!r}")


def test_search(r: Runner, url: str) -> None:
    print(f"\n=== web_search retrieval @ {url} ===")
    try:
        out = _post(url, {"queries": ["Who wrote Hamlet?"], "topk": 2}, timeout=30)
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        r.skip("8. retrieval /retrieve", f"not up ({e.__class__.__name__}). Start it: bash tools/search/run_retrieval.sh (needs wiki-18 index)")
        return
    except Exception as e:
        r.skip("8. retrieval /retrieve", f"unexpected: {e!r}")
        return
    hits = (out.get("result") or [[]])[0]
    r.check("8. retrieval returns passages", len(hits) > 0, f"{len(hits)} hits")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox-url", default="http://127.0.0.1:8420")
    ap.add_argument("--search-url", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument("--skip-search", action="store_true")
    args = ap.parse_args()

    r = Runner()
    test_sandbox(r, args.sandbox_url.rstrip("/"))
    if not args.skip_search:
        test_search(r, args.search_url)
    raise SystemExit(r.summary())


if __name__ == "__main__":
    main()
