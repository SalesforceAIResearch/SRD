"""Build AIME eval jsonls (2024 / 2025 / 2026) in the native tool-calling SDPO_ReAct shape.

Usage:
    python -m examples.SRD.data.build_aime --out-dir /root/data/aime
    python -m examples.SRD.data.build_aime --out-dir /root/data/aime --years 24 26
"""

import argparse
import json
import os

# year -> (HF dataset id, question key, answer key)
_SOURCES = {
    "24": ("zhuzilin/aime-2024", "prompt", "label"),
    "25": ("zhuzilin/aime-2025", "prompt", "label"),
    "26": ("MathArena/aime_2026", "problem", "answer"),
}


def _question(row, qkey):
    v = row[qkey]
    # zhuzilin's `prompt` is a [{role, content}] list; MathArena's `problem` is a str.
    if isinstance(v, list):
        return v[0]["content"]
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/data/aime")
    ap.add_argument(
        "--years", nargs="*", default=list(_SOURCES), choices=list(_SOURCES),
        help="AIME years to build (default: 24 25 26).",
    )
    args = ap.parse_args()

    from datasets import load_dataset

    from examples.SRD.native_prompt import build_native_messages
    from examples.SRD.tools.registry import all_tool_specs as tool_specs

    os.makedirs(args.out_dir, exist_ok=True)

    for year in args.years:
        repo, qkey, akey = _SOURCES[year]
        ds = load_dataset(repo, split="train")
        out_path = os.path.join(args.out_dir, f"aime{year}_eval.jsonl")
        with open(out_path, "w") as f:
            for row in ds:
                question = _question(row, qkey)
                record = {
                    "prompt": build_native_messages(question),
                    "label": str(row[akey]).strip(),
                    "tools": tool_specs,
                    "metadata": {
                        "domain": "math",
                        "question": question,
                        "problem_idx": row.get("problem_idx"),
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(ds)} rows: aime20{year} ({repo}) -> {out_path}")


if __name__ == "__main__":
    main()
