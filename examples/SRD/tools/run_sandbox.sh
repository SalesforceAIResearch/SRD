#!/bin/bash
# One-click launcher for the code_interpreter sandbox sidecar.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8420}"
IMAGE_TAG="${IMAGE_TAG:-sdpo-react-sandbox}"
CONTAINER="${CONTAINER:-sdpo-react-sandbox}"
# Each code_interpreter call spawns its own `python3 -c` subprocess (real OS
# process isolation, see sandbox_server.py); at --generate-max-turns 5-20 x
# --n-samples-per-prompt 8 x --rollout-batch-size 32, several hundred of these
# can be in flight per rollout batch. The original --cpus=4 was measured
# pegged at 384% (of its own 4-core budget) DURING an otherwise-idle 96-core
# host (84% idle overall) -- a self-inflicted bottleneck, not a real resource
# constraint. 32 cores / 32g leaves the training/rollout GPU processes' own
# host-CPU needs untouched (they are GPU-bound, not CPU-bound) while giving
# the sandbox real headroom.
# 64 of the node's 96 cores (p5en.48xlarge: CPUTot=96, RealMemory=2T). The
# rollout/training processes on the same host are GPU-bound, so leaving them 32
# cores is plenty, and `--cpus` is a quota, not a reservation. Raised from 32
# together with SANDBOX_MAX_CONCURRENCY: grading is wall-clock-timed, and with
# only 32 cores a fanned-out eval starved every submission (see
# docker/sandbox_server.py::MAX_CONCURRENCY -- 5.6% vs 18.5% solved on the same
# OJBench candidates). Memory likewise: 32g of a 2T host was needlessly tight
# for competitive-programming solutions that legitimately allocate hundreds of MB.
SANDBOX_CPUS="${SANDBOX_CPUS:-64}"
SANDBOX_MEMORY="${SANDBOX_MEMORY:-256g}"
# How many submissions the sidecar RUNS at once (it queues the rest); keep it
# below SANDBOX_CPUS so each one gets real cores.
SANDBOX_MAX_CONCURRENCY="${SANDBOX_MAX_CONCURRENCY:-32}"

# The training job itself (e.g. inside an enroot session, see
# ../enroot-run-sdpo-react.sh) has no `docker` binary and no need for one: the
# sandbox is a HOST-level singleton, started once by the launcher BEFORE the
# training container starts, and reached over the shared network namespace.
# When docker isn't on PATH, this script degrades to a pure health check --
# if the sidecar isn't already up in that case, we can't start it from here,
# so fail loudly with a pointer at the real fix instead of a confusing
# "docker: command not found".
if ! command -v docker >/dev/null 2>&1; then
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "Sandbox sidecar reachable at 127.0.0.1:${PORT} (no local docker -- assumed host-managed)."
        exit 0
    fi
    echo "No 'docker' binary here AND sandbox sidecar not reachable at 127.0.0.1:${PORT}." >&2
    echo "Start it on the HOST first: bash $SCRIPT_DIR/run_sandbox.sh (see ../enroot-run-sdpo-react.sh)." >&2
    exit 1
fi

if docker ps --filter "name=^${CONTAINER}$" --filter "status=running" --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        # Healthy is not enough: a sidecar left over from an EARLIER job on this
        # node runs that job's image and settings, so a grading-behaviour fix
        # (e.g. the concurrency cap) would silently not apply on exactly the
        # nodes that have been busy. /stats reports the running server's own
        # max_concurrency -- if it disagrees with what we want, replace it.
        # `|| true`, and not just for tidiness: under `set -euo pipefail` the
        # OLD server -- the one we are trying to detect -- has no
        # max_concurrency in /stats, so grep exits 1, pipefail propagates it,
        # and the whole script died right here, leaving the stale sidecar up
        # and every caller silently grading under the flooring bug.
        RUNNING_CONC="$(curl -sf "http://127.0.0.1:${PORT}/stats" 2>/dev/null \
            | grep -o '"max_concurrency":[0-9]*' | grep -o '[0-9]*$' || true)"
        if [ "$RUNNING_CONC" = "$SANDBOX_MAX_CONCURRENCY" ]; then
            echo "Sandbox sidecar already running and healthy: ${CONTAINER} (port ${PORT}, max_concurrency=${RUNNING_CONC})"
            exit 0
        fi
        echo "Sidecar ${CONTAINER} is stale (max_concurrency='${RUNNING_CONC}', want ${SANDBOX_MAX_CONCURRENCY}) -- replacing."
        docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    fi
    echo "Container ${CONTAINER} is running but not healthy on port ${PORT} -- restarting."
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
fi

docker build -t "$IMAGE_TAG" "$SCRIPT_DIR/docker"

# Bind to 127.0.0.1 only: this sidecar is reached exclusively by rollout code
# on the same host (miles.utils.http_utils.post -> http://127.0.0.1:$PORT), it
# is never meant to be reachable off-host.
docker run -d --rm \
    --name "$CONTAINER" \
    --network=bridge \
    --memory="${SANDBOX_MEMORY}" \
    --cpus="${SANDBOX_CPUS}" \
    -e SANDBOX_MAX_CONCURRENCY="${SANDBOX_MAX_CONCURRENCY}" \
    -p "127.0.0.1:${PORT}:8420" \
    "$IMAGE_TAG"

for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "Sandbox sidecar up: ${CONTAINER} (port ${PORT})"
        exit 0
    fi
    sleep 1
done

echo "Sandbox sidecar failed to become healthy within 30s" >&2
docker logs "$CONTAINER" >&2 || true
exit 1
