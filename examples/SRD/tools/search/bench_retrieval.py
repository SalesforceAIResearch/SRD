"""Concurrency/latency benchmark for the web_search retrieval sidecar.

Usage:
    python examples/SRD/tools/search/bench_retrieval.py            # default sweep
    python examples/SRD/tools/search/bench_retrieval.py --url http://127.0.0.1:8000/retrieve \\
        --concurrency 1 8 32 64 128 --requests 256 --topk 3
"""

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

_QUERIES = [
    "Who wrote the novel Nineteen Eighty-Four?",
    "What is the capital of Australia?",
    "When did the French Revolution begin?",
    "Who discovered penicillin?",
    "What is the boiling point of water at sea level?",
    "Which planet is known as the red planet?",
    "Who painted the Mona Lisa?",
    "What year did World War II end?",
]


def _one(url: str, query: str, topk: int) -> float:
    payload = json.dumps({"queries": [query], "topk": topk, "return_scores": False}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()
    return time.monotonic() - t0


def _bench(url: str, concurrency: int, n_requests: int, topk: int) -> dict:
    lat = []
    errs = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_one, url, _QUERIES[i % len(_QUERIES)], topk) for i in range(n_requests)]
        for f in futs:
            try:
                lat.append(f.result())
            except Exception:
                errs += 1
    wall = time.monotonic() - t0
    lat.sort()

    def pct(p):
        return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else float("nan")

    return {
        "concurrency": concurrency,
        "requests": n_requests,
        "errors": errs,
        "wall_s": round(wall, 2),
        "throughput_qps": round((len(lat)) / wall, 1) if wall > 0 else 0,
        "lat_p50_ms": round(pct(0.50) * 1000, 1),
        "lat_p95_ms": round(pct(0.95) * 1000, 1),
        "lat_max_ms": round((lat[-1] if lat else 0) * 1000, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument("--concurrency", type=int, nargs="*", default=[1, 8, 32, 64, 128])
    ap.add_argument("--requests", type=int, default=256)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    # reachability
    try:
        _one(args.url, "warmup", args.topk)
    except Exception as e:
        print(f"Retriever not reachable at {args.url}: {e!r}\nStart it: bash examples/SRD/tools/search/run_retrieval.sh")
        raise SystemExit(1)

    print(f"Benchmarking {args.url}  (requests={args.requests}, topk={args.topk})\n")
    print(f"{'conc':>5} {'qps':>8} {'p50ms':>8} {'p95ms':>8} {'maxms':>9} {'errors':>7}")
    for c in args.concurrency:
        r = _bench(args.url, c, args.requests, args.topk)
        print(f"{r['concurrency']:>5} {r['throughput_qps']:>8} {r['lat_p50_ms']:>8} {r['lat_p95_ms']:>8} {r['lat_max_ms']:>9} {r['errors']:>7}")
    print(
        "\nGuide: rollout issues many concurrent search calls (n_samples_per_prompt x "
        "rollout_batch_size). If qps plateaus / p95 explodes at the concurrency the rollout "
        "actually drives, the retriever is the bottleneck -> give it more GPU, enable request "
        "batching, or use a lighter embedder."
    )


if __name__ == "__main__":
    main()
