"""One ALFWorld sidecar worker process that wraps the ``alfworld`` package's TextWorld env in a session-pool HTTP server."""

import logging
import os
import threading

import anyio
import fast_downward as _fd_mod
import yaml
import textworld.envs.pddl.logic as _pddl_logic_mod
import textworld.envs.pddl.textgen as _pddl_textgen_mod
import textworld.logic as _tw_logic_mod
from alfworld.agents.environment import get_environment
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("alfworld_sidecar_worker")

app = FastAPI()

# FastAPI dispatches sync `def` routes (step() below) to anyio's threadpool,
# capped at 40 workers by default. Each worker process now only serves the
# game files that hash to it (router.py splits ~100 unique eval game files
# across N_WORKERS processes), so it needs less concurrency than the old
# single-process server did at 512 -- 64 is comfortably above one shard's
# realistic concurrent-session count and avoids the GIL/thread-count
# scheduling overhead that 512-in-one-process caused (see router.py's
# docstring for the full story of why one process wasn't enough).
@app.on_event("startup")
async def _raise_threadpool_limit():
    anyio.to_thread.current_default_thread_limiter().total_tokens = 64

_CONFIG_PATH = os.environ.get("ALFWORLD_CONFIG", "/app/config_tw.yaml")
with open(_CONFIG_PATH) as f:
    _CONFIG = yaml.safe_load(f)

# AlfredTWEnv.__init__ always calls collect_game_files(), an os.walk over the
# WHOLE dataset dir (8810+ entries, one json parse each) -- expensive, and
# identical for every session of the same split. AlfworldPool.init_session
# used to construct a fresh AlfredTWEnv per session (needed real init_env()
# per session for the game-file-pinning mechanism below, but NOT a fresh
# collect_game_files() scan every time) -- with many concurrent sessions in
# one rollout batch, all of them re-walking the dataset dir simultaneously
# saturated this single-process server's threadpool and starved /health
# along with it, stalling the whole training rollout. Cache the computed
# game_files list per split so the walk runs once per split for the sidecar's
# lifetime; _get_base_env below bypasses __init__ (hence collect_game_files)
# on a cache hit, restoring only the state init_env()/step() actually need.
_GAME_FILES_CACHE: dict[str, list[str]] = {}
_GAME_FILES_CACHE_LOCK = threading.Lock()


def _get_base_env(env_type: str, config: dict, split: str):
    env_cls = get_environment(env_type)
    # Lock held across the (potential) scan itself, not just the cache
    # lookup -- otherwise every session concurrently missing an empty cache
    # would all fall through and redo the expensive collect_game_files() walk
    # in parallel (which is exactly what stalled the sidecar before this
    # fix), rather than the first one computing it and the rest reusing it.
    with _GAME_FILES_CACHE_LOCK:
        cached = _GAME_FILES_CACHE.get(split)
        if cached is None:
            env = env_cls(config, train_eval=split)
            _GAME_FILES_CACHE[split] = list(env.game_files)
            return env
    env = env_cls.__new__(env_cls)
    env.config = config
    env.train_eval = split
    env.game_files = list(cached)
    env.num_games = len(env.game_files)
    return env


# load()/reset()/step() go through FOUR independent pieces of shared,
# non-thread-safe process-global state -- concurrent sessions in one rollout
# batch corrupt each other's calls into any of them:
#   - textworld.envs.pddl.logic._PARSER (PddlLogicParser): parses the game's
#     PDDL grammar text once at load() time, via GameLogic.__init__ ->
#     _parse_and_convert(). tatsu-generated parsers keep mutable state
#     (_rule_stack) on the module-level parser instance, not per-call.
#   - textworld.envs.pddl.textgen._PARSER (CSGParser): same tatsu-parser
#     issue, in a SIBLING module -- parses grammar rule text both at
#     load() time (ContextSensitiveGrammar.parse(), a classmethod) AND on
#     every step()/reset() (ContextSensitiveGrammar.derive(), rendering
#     feedback text), both via that module's OWN _parse_and_convert().
#   - textworld.logic._PARSER (GameLogicParser): same tatsu-parser issue, in
#     a THIRD module -- used by Rule.parse_conjunctive_query(), called from
#     ContextSensitiveGrammar.replace()'s rule-condition check on every
#     conditional grammar rule, i.e. also on the step()/reset() hot path.
#     Easy to miss (unlike the other two, it's not in the pddl/ subpackage)
#     -- confirmed live: patching only the first two still left ~16% of an
#     800-session load test failing with the exact same IndexErrors, all
#     inside this third parser instead.
#     All three: concurrent corruption crashes the request with `IndexError:
#     pop from empty list` / `TypeError: 'TerminalSymbol' object is not
#     iterable` / tatsu FailedParse/FailedLeftRecursion (a corrupted parser
#     mis-parsing otherwise-valid text).
#   - fast_downward.interface.pddl2sas() (PDDL grounding, called twice per
#     PddlState.__init__: once with optimize=False, once optimize=True):
#     mutates the process-global `fast_downward.translate.options` module
#     (`options.filter_unimportant_vars = ...` etc, read back throughout
#     translate.py) AND process-global `sys.argv`, NOT per-call state.
#     Concurrent calls with different `optimize` values race on those globals
#     -- confirmed live: `KeyError: (149, 0)` in build_sas_operator's
#     `implied_facts[fact]` lookup, which only happens when
#     options.add_implied_preconditions flips under a call that didn't
#     expect it. The exception is caught at the FastAPI route boundary (a
#     tool-usage error must surface as an observation, not a 500) so it
#     doesn't take the uvicorn worker down, but it does corrupt the
#     concurrent request's result.
#
# Locking the WHOLE reset()/step() call (an earlier version of this fix) is
# wrong: PDDL grounding is otherwise a plain, thread-safe, per-session
# CPU-bound computation (confirmed: ~1-2s per game file run serially) --
# serializing it too would still be fine for correctness, but throws away
# the parallelism a training rollout's batch of concurrent sessions needs.
#
# A SHARED LOCK around _parse_and_convert (an earlier version of this fix)
# is also wrong, even split into a separate lock per parser: derive() runs
# on EVERY step()/reset() call (all ~800 sessions in one eval batch) and each
# call is tens of milliseconds (confirmed: 20 sequential env.step() calls ==
# ~1s), NOT sub-millisecond -- serializing 800 of those behind one lock,
# under real thread-scheduling/lock-convoy overhead on a 96-core box with a
# 100+-thread threadpool, measured 12 minutes wall-clock with 64 client
# timeouts. A lock is the wrong tool here: it doesn't need to be correct
# under concurrent mutation, it needs to not share the mutable state at all.
#
# Fix: construct a FRESH PARSER INSTANCE for every call, instead of reusing
# the tatsu-generated parser's single process-global instance. Cheap
# (confirmed: 100 fresh CSGParser() constructions take ~5ms total) -- this
# isn't a hot loop needing amortization. A per-THREAD cached instance
# (threading.local, an earlier version of this fix) is NOT equivalent and
# is NOT safe: confirmed live, a thread's second parse() call on its own
# cached instance still hit `IndexError: pop from empty list` -- some tatsu
# internal state (_rule_stack/_statestack) survives a parse (successful or
# not) in a way that corrupts the NEXT parse on that same instance, even
# with no concurrent access at all. Fresh-per-call sidesteps that entirely:
# there's no reuse for leftover state to poison.
def _make_fresh_parse_and_convert(parser_cls, semantics_cls, converter_cls):
    def _parse_and_convert(*args, **kwargs):
        parser = parser_cls(semantics=semantics_cls(), parseinfo=True)
        model = parser.parse(*args, **kwargs)
        return converter_cls().walk(model)

    return _parse_and_convert


_pddl_textgen_mod._parse_and_convert = _make_fresh_parse_and_convert(
    _pddl_textgen_mod.CSGParser, _pddl_textgen_mod.CSGModelBuilderSemantics, _pddl_textgen_mod._Converter
)
_pddl_logic_mod._parse_and_convert = _make_fresh_parse_and_convert(
    _pddl_logic_mod.PddlLogicParser, _pddl_logic_mod.PddlLogicModelBuilderSemantics, _pddl_logic_mod._ModelConverter
)
_tw_logic_mod._parse_and_convert = _make_fresh_parse_and_convert(
    _tw_logic_mod.GameLogicParser, _tw_logic_mod.GameLogicModelBuilderSemantics, _tw_logic_mod._ModelConverter
)

# fast_downward.interface.pddl2sas() mutates the process-global
# `fast_downward.translate.options` module (`options.filter_unimportant_vars
# = ...` etc, read back throughout translate.py) AND process-global
# `sys.argv`, NOT per-call/per-thread state -- unlike the tatsu parsers
# above, this one genuinely needs a lock (it's write-then-read of one shared
# global object, not a self-contained per-thread instance you could swap
# in). Concurrent calls with different `optimize` values race on those
# globals -- confirmed live: `KeyError: (149, 0)` in build_sas_operator's
# `implied_facts[fact]` lookup, which only happens when
# options.add_implied_preconditions flips under a call that didn't expect
# it.
#
# Grounding is also a pure function of (domain, problem, optimize) -- and
# eval calls it with the SAME (domain, problem) up to n_samples_per_eval_prompt
# times (every session for a given eval row replays the SAME game_file --
# see AlfworldPool's game-file-pinning mechanism above). SINGLE-FLIGHT per
# key, not just a check-then-lock-then-store cache: a naive version (check
# cache -> miss -> acquire _GROUNDING_LOCK -> compute -> store) lets ALL
# n_samples_per_eval_prompt sessions for the SAME never-before-seen game
# file pass the cache check as misses BEFORE the first one finishes and
# stores its result, so all of them queue on _GROUNDING_LOCK and each
# REDOES the same multi-second grounding -- confirmed live: this turned "1
# grounding per unique game file" into up to 8 (n_samples_per_eval_prompt)
# per file, an 8x slowdown on exactly the workload (800 sessions = 100
# files x 8 samples) this cache exists to speed up. Fix: a per-key
# threading.Event lets every session for the same key past the FIRST one
# wait for that one's result and reuse it, instead of recomputing;
# different keys don't block each other at all (only _GROUNDING_LOCK does,
# to protect fast_downward's shared globals during the ACTUAL compute).
_GROUNDING_LOCK = threading.Lock()
_PDDL2SAS_CACHE: dict[tuple, tuple] = {}
_PDDL2SAS_INFLIGHT: dict[tuple, threading.Event] = {}
_PDDL2SAS_CACHE_LOCK = threading.Lock()
_original_pddl2sas = _fd_mod.pddl2sas


def _cached_pddl2sas(domain, problem, verbose=False, optimize=False):
    key = (domain, problem, optimize)
    while True:
        with _PDDL2SAS_CACHE_LOCK:
            cached = _PDDL2SAS_CACHE.get(key)
            if cached is not None:
                return cached
            event = _PDDL2SAS_INFLIGHT.get(key)
            if event is None:
                # We're the first for this key -- claim it and compute.
                event = threading.Event()
                _PDDL2SAS_INFLIGHT[key] = event
                break
        # Someone else is already computing this key -- wait for them to
        # finish, then loop back to the cache check (now a guaranteed hit)
        # instead of racing them into _GROUNDING_LOCK ourselves.
        event.wait()

    try:
        with _GROUNDING_LOCK:
            result = _original_pddl2sas(domain, problem, verbose=verbose, optimize=optimize)
        with _PDDL2SAS_CACHE_LOCK:
            _PDDL2SAS_CACHE[key] = result
        return result
    finally:
        with _PDDL2SAS_CACHE_LOCK:
            del _PDDL2SAS_INFLIGHT[key]
        event.set()


# textworld.envs.pddl.logic's PddlState.__init__ calls the top-level
# `fast_downward.pddl2sas(...)` attribute (not fast_downward.interface's),
# so that's the binding that must be patched for its caller to see this.
_fd_mod.pddl2sas = _cached_pddl2sas


class AlfworldPool:
    """One live TextWorld env per session id. See module docstring for the
    game-file-pinning mechanism (the only non-obvious part of this class)."""

    def __init__(self, config: dict):
        self._config = config
        self.sessions: dict[str, object] = {}

    def init_session(self, session_id: str, game_file: str | None = None, split: str = "train") -> None:
        # split must be one of "train" | "eval_in_distribution" |
        # "eval_out_of_distribution" (AlfredTWEnv.collect_game_files reads a
        # DIFFERENT dataset.*_data_path config key per split -- confirmed by
        # reading the installed package's source). The data builder stamps
        # which split a game_file came from onto metadata["alfworld_split"];
        # the sidecar must load that SAME split's directory listing or the
        # requested game_file won't be found in game_files to filter down to.
        env_type = self._config["env"]["type"]
        base_env = _get_base_env(env_type, self._config, split)
        if game_file:
            # See module docstring: game_files is a plain mutable list;
            # filtering it to one entry before init_env is the only way to
            # pin ALFWorld's shuffle-cycle to a specific file.
            base_env.game_files = [game_file]
        env = base_env.init_env(batch_size=1)
        env.reset()  # REQUIRED before the first step() -- initializes TextWorld's
        # internal per-env state (self.last); the model's first tool call is
        # executed as an action AFTER this reset, same as data/build_alfworld_data.py
        # captures the reset's own initial observation for the row's prompt.
        # (derive()'s own shared-parser access is serialized by the
        # ContextSensitiveGrammar.derive patch above -- no lock needed here.)
        self.sessions[session_id] = env

    def step(self, session_id: str, action: str) -> tuple[str, bool, bool]:
        env = self.sessions[session_id]
        obs, _scores, dones, infos = env.step([action])
        observation = obs[0]
        done = bool(dones[0])
        won = bool(infos.get("won", [False])[0]) if done else False
        admissible = infos.get("admissible_commands", [[]])[0]
        if admissible:
            observation = f"{observation}\nAdmissible actions: {', '.join(admissible)}"
        return observation, done, won

    def cleanup(self, session_id: str) -> None:
        env = self.sessions.pop(session_id, None)
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


_POOL = AlfworldPool(_CONFIG)
_INITIALIZED_SESSIONS: set[str] = set()


class StepRequest(BaseModel):
    action: str
    game_file: str | None = None
    split: str = "train"


class StepResponse(BaseModel):
    observation: str
    done: bool
    won: bool


@app.get("/health")
async def health() -> dict:
    # async (not def): step()/init_session() are sync defs dispatched to
    # FastAPI's shared threadpool, and a rollout batch's worth of concurrent
    # sessions can occupy every worker with slow-but-legitimate PDDL grounding
    # (see _GROUNDING_LOCK above). A sync def here would then queue behind
    # them for the same workers and look like the sidecar died under load,
    # even though it's still making progress underneath. async runs directly
    # on the event loop, so it always answers immediately regardless of
    # threadpool saturation.
    return {"status": "ok"}


@app.post("/session/{session_id}/step", response_model=StepResponse)
def step(session_id: str, req: StepRequest) -> StepResponse:
    if session_id not in _INITIALIZED_SESSIONS:
        _POOL.init_session(session_id, game_file=req.game_file, split=req.split)
        _INITIALIZED_SESSIONS.add(session_id)

    try:
        observation, done, won = _POOL.step(session_id, req.action)
        return StepResponse(observation=observation, done=done, won=won)
    except Exception as e:  # a tool-usage error must surface as an observation, not a 500
        logger.warning("session=%s action=%r failed: %r", session_id, req.action, e)
        return StepResponse(observation=f"error:\n{e}", done=False, won=False)


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    """Explicit cleanup hook (not required for correctness -- see the module
    docstring -- but frees a live TextWorld env proactively instead of
    waiting for process exit)."""
    _POOL.cleanup(session_id)
    _INITIALIZED_SESSIONS.discard(session_id)
    return {"status": "ok"}
