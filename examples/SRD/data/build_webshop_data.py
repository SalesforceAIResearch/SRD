"""Build WebShop train/eval jsonl for the SDPO_ReAct WEBSHOP domain, one row per fixed task (session) index.

Train/eval split reuses SDAR's convention (task indices >=500 train, <500 eval).
Must run inside the webshop sidecar container (needs `web_agent_site.envs`), not
the main training container.

Usage (run inside the webshop sidecar image, e.g.
`docker run --rm -v $(pwd)/out:/out sdpo-react-webshop python build_webshop_data.py ...`):
    python build_webshop_data.py --out-dir /root/data/webshop_data \\
        --n-train 400 --n-eval 100
"""

import argparse
import json
import os
import random

import gym

# Registers WebAgentTextEnv-v0 with gym.
import web_agent_site.envs  # noqa: F401

# Standalone by design -- runs inside the webshop sidecar container, which lacks
# miles/sglang and the rest of the repo (see build_alfworld_data.py).
WEBSHOP_STEP_SPEC = {
    "type": "function",
    "function": {
        "name": "webshop_step",
        "description": (
            "Take one action in the WebShop online-shopping environment. The observation "
            "returned after each call is the current page text (search results, product "
            "page, or product options) plus the list of currently clickable buttons -- "
            "choose your next action from exactly two forms: 'search[query]' to search "
            "for products, or 'click[button text]' to click a button/link shown on the "
            "current page (e.g. 'click[Buy Now]', 'click[< Prev]', a product's ASIN, or an "
            "option like a color/size). The episode ends when you buy a product or the "
            "turn budget runs out."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to take, e.g. 'search[wireless mouse]' or 'click[Buy Now]'.",
                },
            },
            "required": ["action"],
        },
    },
}
from examples.SRD.prompt.system import MINIMAL_SYSTEM_PROMPT
from examples.SRD.prompt.agentic import WEBSHOP_SYSTEM_PROMPT_SUFFIX
WEBSHOP_SYSTEM_PROMPT = MINIMAL_SYSTEM_PROMPT + WEBSHOP_SYSTEM_PROMPT_SUFFIX
_WEBSHOP_TOOLS = [WEBSHOP_STEP_SPEC]


def _build_row(task_id: int, instruction_text: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": WEBSHOP_SYSTEM_PROMPT},
            {"role": "user", "content": instruction_text},
        ],
        "label": "",
        "tools": _WEBSHOP_TOOLS,
        "metadata": {"domain": "webshop", "webshop_task_id": task_id},
    }


def _rows_for_task_ids(task_ids: list[int]) -> list[dict]:
    rows = []
    for task_id in task_ids:
        # Seed before gym.make() (same fix as the sidecar) so each row's
        # captured instruction text has a reproducible price threshold.
        random.seed(0)
        env = gym.make("WebAgentTextEnv-v0", observation_mode="text")
        obs, _info = env.reset(session=task_id)
        env.close()
        rows.append(_build_row(task_id, obs))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/root/data/webshop_data")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument(
        "--eval-start",
        type=int,
        default=0,
        help="held-out eval task-id range starts here (SDAR convention: <500)",
    )
    ap.add_argument(
        "--train-start",
        type=int,
        default=500,
        help="train task-id range starts here (SDAR convention: >=500)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    eval_ids = list(range(args.eval_start, args.eval_start + args.n_eval))
    train_ids = list(range(args.train_start, args.train_start + args.n_train))

    train_rows = _rows_for_task_ids(train_ids)
    train_path = os.path.join(args.out_dir, "webshop_train.jsonl")
    with open(train_path, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(train_rows)} train rows -> {train_path}")

    eval_rows = _rows_for_task_ids(eval_ids)
    eval_path = os.path.join(args.out_dir, "webshop_eval.jsonl")
    with open(eval_path, "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(eval_rows)} eval rows -> {eval_path}")


if __name__ == "__main__":
    main()
