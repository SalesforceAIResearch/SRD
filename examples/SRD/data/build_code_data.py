"""Build LiveCodeBench train/eval jsonl for the SDPO_ReAct CODE domain.

Usage (module, from repo root):
    python -m examples.SRD.build_code_data --out-dir /root/code_data \
        --n-train 2000 --n-eval 100
"""

import argparse
import json
import os

from examples.SRD.tools.registry import MINIMAL_SYSTEM_PROMPT, all_tool_specs
from examples.SRD.prompt.code import CODE_SYSTEM_PROMPT_SUFFIX, FUNCTIONAL_SYSTEM_PROMPT_SUFFIX

tool_specs = all_tool_specs  # every row exposes all tools; model chooses

# Code system prompt for stdin/stdout problems (graded artifact = last program run through code_interpreter).
CODE_SYSTEM_PROMPT = MINIMAL_SYSTEM_PROMPT + CODE_SYSTEM_PROMPT_SUFFIX


# System prompt for FUNCTIONAL (leetcode-style) problems (graded artifact = a class + method, no stdin).
FUNCTIONAL_SYSTEM_PROMPT = MINIMAL_SYSTEM_PROMPT + FUNCTIONAL_SYSTEM_PROMPT_SUFFIX


def _decode_private_tests(val) -> list:
    """Decode LiveCodeBench's private_test_cases (plain JSON string or base64(zlib(pickle(json)))); [] on anything else."""
    if isinstance(val, list):
        return val
    if not isinstance(val, str) or not val:
        return []
    try:
        return json.loads(val)
    except Exception:
        pass
    try:
        import base64
        import pickle
        import zlib

        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(val.encode("utf-8")))))
    except Exception:
        return []


def _normalize_tests(row: dict, include_private: bool = False, max_test_chars: int = 0) -> list[dict]:
    """Extract test cases from a LiveCodeBench-style row into [{"input","output","testtype"[,"fn_name"]}].

    include_private (default OFF): also pull private_test_cases (public first).
    max_test_chars (0 = no limit): drop test cases whose input+output exceeds it.
    """
    tests: list[dict] = []
    keys = ("public_test_cases", "test_cases", "tests")
    if include_private:
        keys = keys + ("private_test_cases",)
    # LeetCode-style rows carry the graded entrypoint in metadata.func_name;
    # stamp it on each test case so judge.py's functional harness can find it.
    md = row.get("metadata")
    if isinstance(md, str):
        try:
            md = json.loads(md)
        except Exception:
            md = {}
    fn_name = str((md or {}).get("func_name") or "") if isinstance(md, dict) else ""
    for key in keys:
        val = row.get(key)
        if not val:
            continue
        if key == "private_test_cases":
            val = _decode_private_tests(val)
        elif isinstance(val, str):
            try:
                val = json.loads(val)
            except Exception:
                continue
        if isinstance(val, list):
            for t in val:
                if isinstance(t, dict) and "input" in t and "output" in t:
                    inp, out = str(t["input"]), str(t["output"])
                    if max_test_chars and len(inp) + len(out) > max_test_chars:
                        continue
                    # Keep testtype (stdin | functional); code_judge grades the
                    # two differently. Default stdin for older rows.
                    tc = {"input": inp, "output": out, "testtype": t.get("testtype", "stdin")}
                    if tc["testtype"] == "functional" and fn_name:
                        tc["fn_name"] = fn_name
                    tests.append(tc)
    # dedup while preserving order
    seen = set()
    uniq = []
    for t in tests:
        k = (t["input"], t["output"])
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    return uniq


def _question(row: dict) -> str:
    for key in ("question_content", "question", "problem", "prompt", "content"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            q = v.strip()
            # Functional (leetcode) problems need the exact signature the judge
            # calls -- append the starter code, as LiveCodeBench's prompt does.
            starter = row.get("starter_code")
            if isinstance(starter, str) and starter.strip():
                q += (
                    "\n\nComplete the following starter code (keep the class and method "
                    "signature exactly as given, and RETURN the answer):\n"
                    f"```python\n{starter.rstrip()}\n```"
                )
            return q
    return ""


def _build_row(question: str, tests: list[dict], system_prompt: str = CODE_SYSTEM_PROMPT) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "label": "",  # code correctness is from test_cases, not a label string
        "tools": tool_specs,
        "metadata": {"domain": "code", "test_cases": tests},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/code_data")
    ap.add_argument("--hf-repo", default="livecodebench/code_generation_lite")
    ap.add_argument(
        "--jsonl-files",
        nargs="*",
        default=["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
        help="jsonl files in the repo to load directly. test6.jsonl = 2025-01..2025-04 (LCB v6).",
    )
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--max-tests", type=int, default=15, help="cap test cases kept per problem")
    ap.add_argument(
        "--include-private-tests",
        action="store_true",
        help="also use LiveCodeBench's private_test_cases (base64+zlib+pickle-encoded). OFF by "
        "default so existing builds reproduce exactly. Turn ON for functional/leetcode problems.",
    )
    ap.add_argument(
        "--max-test-chars",
        type=int,
        default=0,
        help="drop test cases whose input+output exceeds this many chars (0 = no limit). Use with "
        "--include-private-tests, some of which have multi-MB inputs that overflow the judge's ARG_MAX.",
    )
    ap.add_argument(
        "--testtype",
        default="stdin",
        choices=["stdin", "functional", "both"],
        help="which LiveCodeBench problem type to keep. 'stdin' (default) = codeforces-style "
        "stdin/stdout; 'functional' = LeetCode-style function-call; 'both' = all.",
    )
    ap.add_argument(
        "--difficulty",
        default="medium,hard",
        help="comma-separated LiveCodeBench difficulties to keep (easy|medium|hard). Default "
        "'medium,hard' (easy is trivially one-shot). Use 'easy,medium,hard' for all.",
    )
    ap.add_argument(
        "--min-date",
        default="",
        help="keep only problems with contest_date >= this YYYY-MM (e.g. 2025-02 for LCB v6). Empty = no filter.",
    )
    args = ap.parse_args()
    keep_diff = {d.strip().lower() for d in args.difficulty.split(",") if d.strip()}

    from huggingface_hub import hf_hub_download

    os.makedirs(args.out_dir, exist_ok=True)

    # Load the repo's jsonl files directly (newer `datasets` refuses the dataset
    # script). Each row has question_content + public_test_cases.
    ds = []
    for fname in args.jsonl_files:
        try:
            path = hf_hub_download(args.hf_repo, fname, repo_type="dataset")
        except Exception as e:
            print(f"skip {fname}: {e!r}")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ds.append(json.loads(line))
    print(f"loaded {len(ds)} raw problems from {args.hf_repo}")

    rows = []
    kept_type = {"stdin": 0, "functional": 0}
    kept_diff = {}
    for row in ds:
        q = _question(row)
        tests = _normalize_tests(row, include_private=args.include_private_tests, max_test_chars=args.max_test_chars)
        if not (q and tests):
            continue
        ttype = tests[0].get("testtype", "stdin")
        if args.testtype != "both" and ttype != args.testtype:
            continue
        diff = (row.get("difficulty") or "unknown").strip().lower()
        if diff not in keep_diff:
            continue
        if args.min_date and (row.get("contest_date", "")[:7] < args.min_date):
            continue
        kept_type[ttype] = kept_type.get(ttype, 0) + 1
        kept_diff[diff] = kept_diff.get(diff, 0) + 1
        # functional rows are graded by a function call, not stdin/stdout, so
        # the system prompt must match. 'both' mode mixes the two per row.
        prompt = FUNCTIONAL_SYSTEM_PROMPT if ttype == "functional" else CODE_SYSTEM_PROMPT
        # carry difficulty in metadata for later analysis / stratified eval
        r = _build_row(q, tests[: args.max_tests], system_prompt=prompt)
        r["metadata"]["difficulty"] = diff
        # provenance for a future contamination check
        r["metadata"]["testtype"] = ttype
        for k in ("question_id", "platform", "contest_date"):
            if row.get(k):
                r["metadata"][k] = str(row[k])
        rows.append(r)
    print(f"kept {len(rows)} problems (testtype={args.testtype}, difficulty={sorted(keep_diff)}): types={kept_type} diffs={kept_diff}")

    # Eval-only set (n_train==0): take up to n_eval directly. Otherwise cap
    # eval at 20% so train keeps the bulk.
    n_eval = min(args.n_eval, len(rows)) if args.n_train == 0 else min(args.n_eval, len(rows) // 5)
    eval_rows, train_rows = rows[:n_eval], rows[n_eval : n_eval + args.n_train]

    train_path = os.path.join(args.out_dir, "livecodebench_train.jsonl")
    eval_path = os.path.join(args.out_dir, "livecodebench_eval.jsonl")
    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(eval_path, "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train_rows)} train -> {train_path}")
    print(f"Wrote {len(eval_rows)} eval  -> {eval_path}")
    if train_rows:
        ex = train_rows[0]
        print(f"example: {len(ex['metadata']['test_cases'])} test cases, question {len(ex['prompt'][1]['content'])} chars")


if __name__ == "__main__":
    main()
