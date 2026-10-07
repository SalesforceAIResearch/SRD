"""Reframe a multi-turn native tool-calling trace's ChatML turn boundaries for use as an SDPO teacher prefix.

Debug/inspect from the repo root:
    python -m examples.SRD.tools.reframe_trace --demo
    python -m examples.SRD.tools.reframe_trace --dump <sdpo_prompts/*.jsonl>
"""

import json
import re

# Per user's spec: ONLY reframe the ChatML <|im_start|>/<|im_end|> turn
# boundaries into short NLP markers; keep <think>/<tool_call>/<tool_response>
# content verbatim. This mirrors examples/SRD/sdpo.py::_reframe_multiturn_trace
# (the training-side copy) -- keep the two in sync.
_IM_SPLIT_RE = re.compile(r"<\|im_end\|>\s*<\|im_start\|>\s*(assistant|user|system)\b[ \t]*\n?")
_IM_ANY_RE = re.compile(r"<\|im_(?:start|end)\|>[ \t]*(?:assistant|user|system)?[ \t]*\n?")


def reframe_trace(text: str) -> str:
    """Replace ChatML turn boundaries with per-round NLP markers, keeping
    <think>/<tool_call>/<tool_response> verbatim. No-op if no ChatML boundaries."""
    if not text or "<|im_" not in text:
        return text

    segments: list[tuple[str, str]] = []
    last_end = 0
    role_for_next = "assistant"  # trace begins inside the assistant's turn
    for m in _IM_SPLIT_RE.finditer(text):
        segments.append((role_for_next, text[last_end : m.start()]))
        role_for_next = m.group(1)
        last_end = m.end()
    segments.append((role_for_next, text[last_end:]))

    def _clean(seg: str) -> str:
        return _IM_ANY_RE.sub("", seg).strip()

    parts: list[str] = []
    round_no = 0
    for role, seg in segments:
        seg = _clean(seg)
        if not seg:
            continue
        if role == "assistant":
            round_no += 1
            parts.append(f"Round {round_no} reasoning and tool call:\n{seg}")
        else:
            parts.append(f"Observation:\n{seg}")
    return "\n\n".join(parts) if parts else _clean(text)


_DEMO = (
    '<think>\nLet me verify with sympy.\n</think>\n\n'
    '<tool_call>\n{"name": "code_interpreter", "arguments": {"code": "print(2+2)"}}\n</tool_call>'
    "<|im_end|>\n<|im_start|>user\n<tool_response>\n4\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    "The sum is 4. <answer>4</answer><|im_end|>"
)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="reframe the built-in demo trace")
    ap.add_argument("--dump", help="an sdpo_prompts/*.jsonl file; reframe the first has_prefix teacher trace")
    args = ap.parse_args()

    if args.demo:
        print("=== RAW ===\n" + _DEMO)
        print("\n=== REFRAMED ===\n" + reframe_trace(_DEMO))
        return
    if args.dump:
        for line in open(args.dump):
            r = json.loads(line)
            if r.get("has_prefix"):
                tp = r["teacher_prompt_text"]
                i = tp.find("Correct solution:")
                j = tp.find("Correctly solve the original question")
                raw = tp[i + len("Correct solution:") : j].strip() if i >= 0 else tp
                print("=== RAW PREFIX ===\n" + raw[:1500])
                print("\n=== REFRAMED ===\n" + reframe_trace(raw)[:1500])
                return
        print("no has_prefix row found")


if __name__ == "__main__":
    main()
