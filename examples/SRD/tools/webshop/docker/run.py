"""Container entrypoint that starts N worker.py processes then router.py in the foreground."""

import multiprocessing
import os
import time

import httpx
import uvicorn

N_WORKERS = int(os.environ.get("WEBSHOP_N_WORKERS", "16"))
_WORKER_BASE_PORT = int(os.environ.get("WEBSHOP_WORKER_BASE_PORT", "9101"))


def _run_worker(port: int) -> None:
    uvicorn.run("worker:app", host="127.0.0.1", port=port, log_level="info")


def main() -> None:
    procs = []
    for i in range(N_WORKERS):
        port = _WORKER_BASE_PORT + i
        p = multiprocessing.Process(target=_run_worker, args=(port,), name=f"webshop-worker-{i}", daemon=True)
        p.start()
        procs.append(p)

    # Wait for every worker's own /health before starting the router, so the
    # FIRST real request the router forwards doesn't race a worker that's
    # still importing gym/web_agent_site (a multi-second cold start).
    deadline = time.time() + 120
    for i, p in enumerate(procs):
        port = _WORKER_BASE_PORT + i
        while True:
            if not p.is_alive():
                raise RuntimeError(f"webshop worker {i} (pid target port {port}) died during startup")
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                raise RuntimeError(f"webshop worker {i} on port {port} did not become healthy within 120s")
            time.sleep(0.5)

    uvicorn.run("router:app", host="0.0.0.0", port=8422, log_level="info")


if __name__ == "__main__":
    main()
