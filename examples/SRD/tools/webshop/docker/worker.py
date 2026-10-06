"""One WebShop sidecar worker process that wraps WebShop's ``WebAgentTextEnv-v0`` gym env in a session-pool HTTP server."""

import logging
import random
import threading
from typing import Optional, Union

import anyio
import gym
from fastapi import FastAPI
from pydantic import BaseModel

# Vendored WebShop's own gym env registration (see Dockerfile: cloned into
# /app/webshop, PYTHONPATH set there) -- importing this module registers
# WebAgentTextEnv-v0 with gym.
import web_agent_site.envs  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("webshop_sidecar_worker")

app = FastAPI()


# Each worker process now only serves the task_ids that hash to it
# (router.py splits the eval task set across N_WORKERS processes), so it
# needs less concurrency than the old single-process server did at 512 -- 64
# is comfortably above one shard's realistic concurrent-session count and
# avoids the GIL/thread-count scheduling overhead that 512-in-one-process
# caused (see router.py's docstring for the full story of why one process
# wasn't enough).
@app.on_event("startup")
async def _raise_threadpool_limit():
    anyio.to_thread.current_default_thread_limiter().total_tokens = 64

# gym.make("WebAgentTextEnv-v0") constructs a brand-new pyserini
# LuceneSearcher per session (see engine.py's init_search_engine, called from
# WebAgentTextEnv.__init__) -- a JVM-backed index open, not a cheap local
# call. A rollout batch's worth of NEW sessions all calling init_session()
# concurrently (each on its own FastAPI threadpool worker) means many
# concurrent first-time JVM/Lucene index opens; also, `random.seed(0)` right
# before gym.make() mutates the process-global `random` module state (see the
# module docstring), so two overlapping init_session() calls can interleave
# their seed-then-sample sequence and land on the wrong goal/price threshold.
# Same "serialize the shared/expensive one-time setup path" fix as
# ../../alfworld/docker/server.py's _ENV_LOCK.
_INIT_LOCK = threading.Lock()


class WebshopPool:
    """One live WebAgentTextEnv per session id."""

    def __init__(self):
        self.sessions: dict[str, object] = {}

    def init_session(self, session_id: str, task_id=None) -> None:
        with _INIT_LOCK:
            # See module docstring: seeds the process-global `random` module
            # BEFORE gym.make() so get_goals()'s unseeded price-threshold
            # sample is deterministic too, not just the goal-list shuffle
            # order.
            random.seed(0)
            env = gym.make("WebAgentTextEnv-v0", observation_mode="text")
            if task_id is not None:
                env.reset(session=task_id)
            else:
                env.reset()
            self.sessions[session_id] = env

    def step(self, session_id: str, action: str) -> tuple[str, bool, bool, float]:
        env = self.sessions[session_id]
        # WebAgentTextEnv.step()'s 4th return value (`info`) is ALWAYS None --
        # confirmed by reading web_agent_site/envs/web_agent_text_env.py:
        # `step()` hardcodes `info = None` and never assigns it. The real
        # cumulative reward (0.0 while shopping, the get_reward() score in
        # [0, 1] once a purchase is made) is the 2nd return value. Reading it
        # from `info.get("task_score")` (the original bug here) always fell
        # through the `isinstance(info, dict)` guard to 0.0, silently
        # discarding every episode's true reward regardless of outcome --
        # confirmed live: a purchase that actually scored 1.0 reported 0.0.
        obs, reward, done, _info = env.step(action)
        task_score = float(reward)
        won = bool(done and task_score >= 1.0)
        return obs, bool(done), won, task_score

    def cleanup(self, session_id: str) -> None:
        env = self.sessions.pop(session_id, None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


_POOL = WebshopPool()
_INITIALIZED_SESSIONS: set[str] = set()


class StepRequest(BaseModel):
    action: str
    # pydantic 1.8.2 (pinned by WebShop's own spacy<3.4 requirement -- see
    # requirements.txt) doesn't support PEP 604 `X | Y` union syntax; use
    # typing.Optional/Union instead. IMPORTANT: pydantic 1.x's Union coercion
    # tries member types IN ORDER and keeps the first that doesn't error --
    # `Union[int, str]` with a JSON string body value like "9" should coerce
    # to int 9, but empirically (confirmed live) it was landing as the STRING
    # "9". WebAgentTextEnv.reset()'s own `isinstance(session, int)` check then
    # silently fails, falling through to a RANDOM goal every time (which,
    # combined with our seeded `random.seed(0)`, deterministically picked the
    # SAME "random" goal regardless of the requested task_id -- the exact bug
    # this comment is here to prevent regressing). Use plain `int` (not a
    # Union) and cast at the call site instead of trusting pydantic to coerce.
    task_id: Optional[int] = None


class StepResponse(BaseModel):
    observation: str
    done: bool
    won: bool
    task_score: float


@app.get("/health")
async def health() -> dict:
    # async (not def): step()/init_session() below are sync defs, so FastAPI
    # dispatches each concurrent /step call to a worker in its shared
    # threadpool -- a rollout batch's worth of sessions all blocking on a
    # slow gym.make()/env.step() (or, for a fresh session, the Lucene/JVM
    # search-engine init inside init_search_engine()) can saturate that pool.
    # A sync def here would then queue behind them for the same workers and
    # look like the sidecar died under load, even though it's still making
    # progress underneath. async runs directly on the event loop, so it
    # always answers immediately regardless of threadpool saturation.
    return {"status": "ok"}


@app.post("/session/{session_id}/step", response_model=StepResponse)
def step(session_id: str, req: StepRequest) -> StepResponse:
    if session_id not in _INITIALIZED_SESSIONS:
        _POOL.init_session(session_id, task_id=req.task_id)
        _INITIALIZED_SESSIONS.add(session_id)

    try:
        observation, done, won, task_score = _POOL.step(session_id, req.action)
        return StepResponse(observation=observation, done=done, won=won, task_score=task_score)
    except Exception as e:  # a tool-usage error must surface as an observation, not a 500
        logger.warning("session=%s action=%r failed: %r", session_id, req.action, e)
        return StepResponse(observation=f"error:\n{e}", done=False, won=False, task_score=0.0)


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    """Explicit cleanup hook (not required for correctness -- see the module
    docstring -- but frees a live env proactively instead of waiting for
    process exit)."""
    _POOL.cleanup(session_id)
    _INITIALIZED_SESSIONS.discard(session_id)
    return {"status": "ok"}
