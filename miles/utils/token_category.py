"""Cheap, regex-based token-category classification for response tokens.

Used by EPO's training-time diagnostics (MegatronTrainRayActor._compute_epo_credit)
to split the per-token credit_t signal by category — epistemic / strategy /
compute / format — matching the simple keyword rules from the PMI-credit
diagnosis plan (arXiv:2603.24472's epistemic-suppression mechanism):

  epistemic: wait, hmm, reconsider, let me, actually, hold on, alternatively, ...
  strategy:  apply, use, by, inequality, theorem, lemma, substitut, let X = ...
  compute:   arithmetic patterns (`3 + 4`, `= 12`, ...)
  format:    <think>, </think>, special/chat tokens

This is intentionally a coarse, decode-and-regex classifier — good enough to
watch a trend (e.g. "does epistemic-token credit collapse first?"), not a
precise linguistic parser.

Cost note: a rollout step can carry `rollout_batch_size * n_samples_per_prompt`
samples (e.g. 256) at up to `--rollout-max-response-len` tokens each (e.g.
8192) — calling `tok.decode()` PER TOKEN would be ~2M tokenizer calls per step.
Instead this classifies from `tok.convert_ids_to_tokens()` (one fast, batched,
dict-lookup call per sample) and does the sliding-window text reconstruction
with plain string joins — no further tokenizer calls. No torch dependency, so
it is unit-testable and importable without the training stack.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

CATEGORIES = ("epistemic", "strategy", "compute", "format", "other")

_EPISTEMIC_RE = re.compile(
    r"\b(wait|hmm+|reconsider|let me|actually|hold on|alternatively|"
    r"double.?check|re-?check|rethink|on second thought|i (?:think|realize|see)|"
    r"不对|重新|等等|让我|再想想|不,)\b",
    re.IGNORECASE,
)
_STRATEGY_RE = re.compile(
    r"\b(apply|applying|using|use the|by the|inequality|theorem|lemma|"
    r"substitut\w*|let\s+\w+\s*=|corollary|property of)\b",
    re.IGNORECASE,
)
_COMPUTE_RE = re.compile(
    r"[0-9]+\s*[+\-*/]\s*[0-9]+|=\s*[0-9]+(?:\.[0-9]+)?"
)
_FORMAT_RE = re.compile(
    r"<think>|</think>|<answer>|</answer>|<\|[^>]*\|>"
)

# Byte-level-BPE / SentencePiece leading-space markers, stripped so sliding-
# window text reconstruction reads like normal text for regex matching.
_SUBWORD_SPACE_MARKERS = ("Ġ", "▁")  # 'Ġ' (GPT2/BPE), '▁' (SentencePiece)


def classify_text_span(text: str) -> str:
    """Classify a short decoded text span into one of CATEGORIES. Checked in a
    fixed priority order (format > epistemic > strategy > compute > other) since
    a span can match more than one pattern (e.g. a format tag inside an
    otherwise-epistemic sentence) and format/epistemic are the highest-signal,
    least-ambiguous categories for the diagnosis this feeds."""
    if not text:
        return "other"
    if _FORMAT_RE.search(text):
        return "format"
    if _EPISTEMIC_RE.search(text):
        return "epistemic"
    if _STRATEGY_RE.search(text):
        return "strategy"
    if _COMPUTE_RE.search(text):
        return "compute"
    return "other"


def _normalize_piece(piece: str) -> str:
    for marker in _SUBWORD_SPACE_MARKERS:
        piece = piece.replace(marker, " ")
    return piece


def classify_response_tokens(tok, token_ids: Sequence[int], window: int = 6) -> list[str]:
    """Classify each response token by regex-matching a small LOCAL window of
    reconstructed text around it (window sub-word pieces before + the token
    itself) rather than the single token in isolation — keyword phrases like
    "let me" or "hold on" span multiple sub-word tokens, and a lone piece
    (" me", " on") carries no signal by itself.

    Cheap by construction: ONE batched `tok.convert_ids_to_tokens()` call (a
    fast vocab lookup, not detokenization) for the whole sequence, then plain
    string joins over a bounded window — no further tokenizer calls, so cost
    is linear in response length with a small constant, not one tokenizer call
    per token. Text reconstruction is best-effort (byte-level-BPE/SentencePiece
    space markers stripped; some tokenizers may still render imperfectly), but
    is only used for coarse keyword matching, not exact detokenization.

    Returns a list of category strings (see CATEGORIES), same length as
    token_ids, one per response token.
    """
    n = len(token_ids)
    if n == 0:
        return []
    pieces = tok.convert_ids_to_tokens(list(token_ids))
    norm_pieces = [_normalize_piece(p) for p in pieces]

    out = []
    for i in range(n):
        start = max(0, i - window + 1)
        window_text = "".join(norm_pieces[start : i + 1])
        cat = classify_text_span(window_text)
        if cat == "other":
            # Format tags (e.g. <think>, <|im_end|>) are typically a single
            # whole vocab entry; check the RAW (unnormalized) piece directly in
            # case space-marker stripping obscured it in the joined window.
            if _FORMAT_RE.search(pieces[i]):
                cat = "format"
        out.append(cat)
    return out
