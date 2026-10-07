"""Integration test for the search/open/find tool set (examples/SRD/
tools/search/), run against the REAL search sidecar + wiki-18 retriever --
not a mock. Exercises the exact path a real rollout takes:

    (mock LLM tool call) -> miles.rollout.generate_utils.tool_call_utils.
    execute_tool_calls (miles core, unmodified) -> examples.SRD.tools.
    registry.execute_tool -> tools.search.client.call_search_tool -> the
    search sidecar container (tools/search/docker/, wrapping i-DeepSearch's
    BrowserTool/BrowserPool verbatim) -> the wiki-18 retriever (tools/search/
    retrieval_server.py or the BM25 backend, tools/search/run_retrieval.sh).

Session-id plumbing: miles/rollout/generate_hub/multi_turn.py sets a fresh
per-trajectory session id (a contextvars.ContextVar) once per `generate()`
call, BEFORE the turn loop, so every tool call within one trajectory shares
it -- this is what lets `open`/`find` resolve against the SAME trajectory's
prior `search`. This test sets that var directly (bypassing the full
`generate()` turn loop, which needs a live SGLang engine) to isolate the
piece actually under test: the tool dispatch chain + the real sidecar.

Requires (skipped entirely if not up -- see the module-level skip below):
  - the search sidecar on port 8421 (tools/search/run_search_sidecar.sh)
  - a retriever backing it on port 8000 (tools/search/run_retrieval.sh, either
    backend -- BM25/CPU or the torch-GPU dense retriever)
  - the code sandbox on port 8420 (tools/run_sandbox.sh), for the
    code_interpreter cross-check at the bottom of this file

Not registered via register_cpu_ci: unlike this directory's other tests (pure
CPU, no Docker, no network), this one needs real running sidecar containers,
so it is not part of the no-infra CPU suite. Run manually:
    pytest tests/fast/examples/test_search_tool.py -v
"""

import json
import uuid

import httpx
import pytest
from openai.types.chat import ChatCompletionMessageToolCall

from examples.SRD.tools import registry
from miles.rollout.generate_hub import multi_turn
from miles.rollout.generate_utils.tool_call_utils import execute_tool_calls
from miles.utils.http_utils import post

SEARCH_SIDECAR_URL = "http://127.0.0.1:8421"
SANDBOX_URL = "http://127.0.0.1:8420"


def _sidecar_up(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=3.0).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _sidecar_up(SEARCH_SIDECAR_URL),
    reason=f"search sidecar not reachable at {SEARCH_SIDECAR_URL} -- start it: "
    "bash examples/SRD/tools/search/run_search_sidecar.sh "
    "(needs a retriever up first: bash examples/SRD/tools/search/run_retrieval.sh)",
)


@pytest.fixture(autouse=True)
def _http_client():
    """miles.utils.http_utils.post needs its shared client initialized --
    normally done once at rollout-worker startup."""
    import miles.utils.http_utils as http_utils

    if http_utils._http_client is None:
        http_utils._http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    yield


def _tool_call(name: str, arguments: dict, idx: int = 0) -> ChatCompletionMessageToolCall:
    """The exact object shape SGLang's FunctionCallParser.parse_non_stream
    hands multi_turn.generate -- see tool_call_utils._execute_tool_call's
    isinstance checks. Using the real openai type (not a duck-typed stand-in)
    means this test exercises the REAL dispatch branch, not a shortcut."""
    return ChatCompletionMessageToolCall(
        id=f"call_{idx}",
        type="function",
        function={"name": name, "arguments": json.dumps(arguments)},
    )


async def _run_tool(name: str, arguments: dict) -> str:
    """One simulated model turn: build a real tool-call object, dispatch it
    through the REAL, unmodified execute_tool_calls -> registry.execute_tool
    chain, return the observation text."""
    tool_call = _tool_call(name, arguments)
    tool_messages = await execute_tool_calls([tool_call], registry.execute_tool)
    return tool_messages[0]["content"]


class TestSearchOpenFindAgainstRealSidecar:
    """search -> open -> find within ONE trajectory (session id), verified
    against the actual running sidecar + retriever, no mocking."""

    @pytest.mark.asyncio
    async def test_search_then_open_then_find_share_state_within_one_trajectory(self):
        # Mirrors multi_turn.generate: one fresh session id set ONCE before
        # the turn loop (see multi_turn.py's _TRAJECTORY_SESSION_ID docstring
        # for why it must be set here, in this coroutine's own frame, and not
        # inside a child Task).
        multi_turn._TRAJECTORY_SESSION_ID.set(str(uuid.uuid4()))

        search_obs = await _run_tool("search", {"query": "who wrote hamlet", "topn": 3})
        assert "Search Results" in search_obs
        assert "【" in search_obs, "search results should carry gpt-oss's 【id†url】 citation markers"

        open_obs = await _run_tool("open", {"id": 0})
        # opening result 0 must NOT re-error -- it must resolve against THIS
        # trajectory's own just-completed search, not raise "No pages".
        assert "No pages to access" not in open_obs
        assert "error" not in open_obs.lower()

        find_obs = await _run_tool("find", {"pattern": "the"})
        # find must operate on the page `open` just navigated to, not error
        # out for lack of a page, and not re-run search.
        assert "No pages to access" not in find_obs
        assert "Cannot run `find` on search results page" not in find_obs

    @pytest.mark.asyncio
    async def test_fresh_trajectory_cannot_see_a_different_trajectorys_search(self):
        """Session isolation: a trajectory that never called `search` must
        NOT be able to `open` a result -- proves state doesn't leak across
        the per-trajectory session id."""
        # Trajectory A searches and gets real results.
        multi_turn._TRAJECTORY_SESSION_ID.set(str(uuid.uuid4()))
        search_obs = await _run_tool("search", {"query": "who wrote hamlet", "topn": 2})
        assert "Search Results" in search_obs

        # Trajectory B (fresh session id) tries to open id=0 with NO prior
        # search of its own.
        multi_turn._TRAJECTORY_SESSION_ID.set(str(uuid.uuid4()))
        open_obs = await _run_tool("open", {"id": 0})
        assert "No pages to access" in open_obs or "error" in open_obs.lower()

    @pytest.mark.asyncio
    async def test_open_by_id_resolves_a_different_result_than_id_zero(self):
        """Sanity: open(id=1) and open(id=0) from the SAME search must
        resolve to genuinely different pages, not silently collapse to the
        same one (would indicate the sidecar's id->url mapping is broken)."""
        multi_turn._TRAJECTORY_SESSION_ID.set(str(uuid.uuid4()))
        await _run_tool("search", {"query": "who wrote hamlet", "topn": 3})

        open_0 = await _run_tool("open", {"id": 0})
        open_1 = await _run_tool("open", {"id": 1})
        assert open_0 != open_1, "open(id=0) and open(id=1) should show different pages"

    @pytest.mark.asyncio
    async def test_unknown_search_tool_name_is_a_clean_observation_not_a_crash(self):
        """registry.execute_tool must surface an unknown-tool error as
        observation text (so the rollout keeps going), never raise."""
        multi_turn._TRAJECTORY_SESSION_ID.set(str(uuid.uuid4()))
        obs = await _run_tool("browse_the_web_please", {})
        assert "error" in obs.lower()
        assert "browse_the_web_please" in obs


class TestCrossToolDispatchStillWorks:
    """The registry dispatches search/open/find AND code_interpreter through
    the same execute_tool -- confirm adding the stateful search tools didn't
    regress the stateless code_interpreter path."""

    @pytest.mark.skipif(
        not _sidecar_up(SANDBOX_URL),
        reason=f"code sandbox not reachable at {SANDBOX_URL} -- start it: "
        "bash examples/SRD/tools/run_sandbox.sh",
    )
    @pytest.mark.asyncio
    async def test_code_interpreter_still_dispatches_through_registry(self):
        multi_turn._TRAJECTORY_SESSION_ID.set("unused-by-stateless-tools")
        obs = await _run_tool("code_interpreter", {"code": "print(21 * 2)"})
        assert obs.strip() == "42"


class TestSearchClientDirectly:
    """Lower-level checks against examples/SRD/tools/search/client.py
    itself (skipping the tool_call_utils/registry dispatch layer), to isolate
    failures: if THESE pass but the dispatch-layer tests above fail, the bug
    is in dispatch, not in the sidecar/client."""

    @pytest.mark.asyncio
    async def test_call_search_tool_returns_real_results(self):
        from examples.SRD.tools.search.client import call_search_tool

        multi_turn._TRAJECTORY_SESSION_ID.set(str(uuid.uuid4()))
        obs = await call_search_tool("search", {"query": "who wrote hamlet", "topn": 2})
        assert "Search Results" in obs

    @pytest.mark.asyncio
    async def test_sidecar_session_delete_is_idempotent(self):
        """DELETE /session/{id} (tools/search/docker/server.py's explicit
        cleanup hook) must succeed even for a session id that was never
        created."""
        resp = await post(
            f"{SEARCH_SIDECAR_URL}/session/{uuid.uuid4()}",
            {},
            max_retries=1,
            action="delete",
        )
        assert resp.get("status") == "ok"
