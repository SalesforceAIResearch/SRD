"""Build OJBench (He-Ren/OJBench_testdata, Python NOI+ICPC competition problems) eval jsonl for native tool-calling SDPO_ReAct code eval.

Usage:
    python -m examples.SRD.data.build_ojbench --out-dir /root/data/ojbench
    # For quick subset:
    python -m examples.SRD.data.build_ojbench --out-dir /root/data/ojbench --max-problems 50
"""

import argparse
import json
import os
import yaml
import zipfile

from huggingface_hub import hf_hub_download


def _load_test_cases(problem_id, dataset, max_tests=10, max_input_bytes=50000):
    """Download and extract test cases for one problem, skipping inputs > max_input_bytes."""
    if dataset == "NOI":
        folder = f"NOI/loj-{problem_id}"
    else:
        folder = f"ICPC/{problem_id}"
    try:
        yml_path = hf_hub_download("He-Ren/OJBench_testdata", f"{folder}/init.yml", repo_type="dataset")
    except Exception:
        return []

    with open(yml_path) as f:
        config = yaml.safe_load(f)

    # Archive name varies (NOI: tests.zip, ICPC: data.zip); init.yml names it in
    # its "archive" key. Always trust init.yml rather than hardcoding tests.zip.
    archive = config.get("archive") or "tests.zip"
    try:
        zip_path = hf_hub_download("He-Ren/OJBench_testdata", f"{folder}/{archive}", repo_type="dataset")
    except Exception as e:
        print(f" [no {archive}: {type(e).__name__}]", end="")
        return []

    test_cases = []
    with zipfile.ZipFile(zip_path) as z:
        for tc in config.get("test_cases", []):
            if len(test_cases) >= max_tests:
                break
            in_file = tc.get("in", "")
            out_file = tc.get("out", "")
            try:
                inp_bytes = z.read(in_file)
                if len(inp_bytes) > max_input_bytes:
                    continue
                inp = inp_bytes.decode()
                out = z.read(out_file).decode()
                test_cases.append({"input": inp, "output": out, "testtype": "stdin"})
            except Exception:
                continue

    return test_cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/data/ojbench")
    ap.add_argument("--max-problems", type=int, default=None, help="Limit number of problems (for quick test)")
    ap.add_argument("--max-tests-per-problem", type=int, default=10)
    ap.add_argument("--max-input-bytes", type=int, default=50000, help="Skip test cases with input > this size (OJBench has 70MB inputs designed for C++)")
    ap.add_argument("--difficulty", type=str, default=None,
                    help="Comma-separated difficulty filter, e.g. 'easy,medium'. "
                         "Dropping 'hard' is usually right (no model scores on NOI-hard).")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Comma-separated dataset filter: NOI,ICPC (default: both)")
    args = ap.parse_args()

    from examples.SRD.data.build_code_data import CODE_SYSTEM_PROMPT
    from examples.SRD.tools.registry import all_tool_specs as tool_specs

    os.makedirs(args.out_dir, exist_ok=True)

    # Load prompts
    prompts_path = hf_hub_download("He-Ren/OJBench_testdata", "prompts/full.jsonl", repo_type="dataset")
    with open(prompts_path) as f:
        all_rows = [json.loads(l) for l in f]

    py_rows = [r for r in all_rows if r.get("language") == "python"]
    if args.difficulty:
        keep = {d.strip() for d in args.difficulty.split(",") if d.strip()}
        py_rows = [r for r in py_rows if r.get("difficulty") in keep]
    if args.dataset:
        keep = {d.strip() for d in args.dataset.split(",") if d.strip()}
        py_rows = [r for r in py_rows if r.get("dataset") in keep]
    if args.max_problems:
        py_rows = py_rows[:args.max_problems]

    from collections import Counter
    breakdown = Counter((r["dataset"], r.get("difficulty", "")) for r in py_rows)
    print(f"Building OJBench eval: {len(py_rows)} Python problems "
          f"({', '.join(f'{k[0]}/{k[1]}={v}' for k, v in sorted(breakdown.items()))})")

    out_path = os.path.join(args.out_dir, "ojbench_eval.jsonl")
    n_ok = 0
    with open(out_path, "w") as f:
        for i, row in enumerate(py_rows):
            pid = row["id"]
            dataset = row["dataset"]
            print(f"  [{i+1}/{len(py_rows)}] {pid} ({dataset}, {row['difficulty']})...", end="", flush=True)

            test_cases = _load_test_cases(pid, dataset, args.max_tests_per_problem, args.max_input_bytes)
            if not test_cases:
                print(" SKIP (no tests)")
                continue

            record = {
                "prompt": [
                    {"role": "system", "content": CODE_SYSTEM_PROMPT},
                    {"role": "user", "content": row["prompt"]},
                ],
                "label": "",
                "tools": tool_specs,
                "metadata": {
                    "domain": "code",
                    "test_cases": test_cases,
                    "ojbench_id": pid,
                    "ojbench_dataset": dataset,
                    "ojbench_difficulty": row.get("difficulty", ""),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_ok += 1
            print(f" OK ({len(test_cases)} tests)")

    print(f"\nWrote {n_ok}/{len(py_rows)} problems -> {out_path}")


if __name__ == "__main__":
    main()
