"""Unit tests for the cheap regex-based token-category classifier used by
EPO's training-time credit_t diagnostics
(MegatronTrainRayActor._compute_epo_credit_category_metrics).
"""

from miles.utils.token_category import CATEGORIES, classify_response_tokens, classify_text_span

# This module intentionally has no explicit CI registration call: modules under
# tests/fast are implicitly assigned to the stage-a-cpu suite by the CI collector
# (an explicit default-form call would be rejected by the AC-9 meta-test).


class _FakeTokenizer:
    """Minimal stand-in for a HF tokenizer: only convert_ids_to_tokens is used
    by classify_response_tokens."""

    def __init__(self, vocab: dict[int, str]):
        self.vocab = vocab

    def convert_ids_to_tokens(self, ids):
        return [self.vocab[i] for i in ids]


def test_classify_text_span_all_categories():
    assert classify_text_span("Wait, let me reconsider this.") == "epistemic"
    assert classify_text_span("Apply the theorem here.") == "strategy"
    assert classify_text_span("3 + 4 = 7") == "compute"
    assert classify_text_span("<think>") == "format"
    assert classify_text_span("The cat sat on the mat.") == "other"
    assert classify_text_span("") == "other"


def test_classify_text_span_priority_format_over_epistemic():
    # A span matching both format and epistemic patterns should resolve as
    # format (highest priority) per the documented fixed order.
    assert classify_text_span("<think>wait, let me reconsider</think>") == "format"


_VOCAB = {
    0: "Wait",
    1: "Ġ,",
    2: "Ġlet",
    3: "Ġme",
    4: "Ġreconsider",
    5: "Ġ.",
    6: "Ġapply",
    7: "Ġthe",
    8: "Ġtheorem",
    9: "Ġ3",
    10: "Ġ+",
    11: "Ġ4",
    12: "Ġ=",
    13: "Ġ7",
    14: "<think>",
    15: "Ġfoo",
}


def test_classify_response_tokens_epistemic_phrase_spans_multiple_tokens():
    ids = [0, 1, 2, 3, 4, 5]  # "Wait , let me reconsider ."
    cats = classify_response_tokens(_FakeTokenizer(_VOCAB), ids)
    assert cats == ["epistemic"] * len(ids)


def test_classify_response_tokens_strategy():
    ids = [6, 7, 8]  # "apply the theorem"
    cats = classify_response_tokens(_FakeTokenizer(_VOCAB), ids)
    assert cats == ["strategy"] * len(ids)


def test_classify_response_tokens_compute_needs_window():
    ids = [9, 10, 11, 12, 13]  # "3 + 4 = 7"
    cats = classify_response_tokens(_FakeTokenizer(_VOCAB), ids)
    # The first two tokens alone ("3", "+") don't yet complete an arithmetic
    # pattern within their own window; later tokens do once enough of the
    # expression has accumulated in the sliding window.
    assert cats[-1] == "compute"
    assert cats[-2] == "compute"


def test_classify_response_tokens_format_single_token():
    ids = [14]  # "<think>"
    cats = classify_response_tokens(_FakeTokenizer(_VOCAB), ids)
    assert cats == ["format"]


def test_classify_response_tokens_empty():
    assert classify_response_tokens(_FakeTokenizer(_VOCAB), []) == []


def test_classify_response_tokens_other_fallback():
    ids = [15, 15]  # "foo foo" -- no category matches
    cats = classify_response_tokens(_FakeTokenizer(_VOCAB), ids)
    assert cats == ["other", "other"]


def test_all_categories_covered_by_constant():
    # Sanity: CATEGORIES is the exact set classify_text_span can return.
    seen = {
        classify_text_span(s)
        for s in ["wait", "apply the theorem", "1 + 1", "<think>", "unrelated text"]
    }
    assert seen <= set(CATEGORIES)
