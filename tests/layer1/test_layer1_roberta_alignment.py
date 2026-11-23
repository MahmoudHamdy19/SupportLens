"""RoBERTa baseline: BIO alignment, span decoding and the span-to-tuple rules.

The alignment logic is pure and is tested with hand-made token offsets, so no tokenizer is
downloaded. Tests that need the real `roberta-base` tokenizer are marked `network`.
"""

import json

import pytest

from src.layer1_ai_extraction import roberta_baseline as rb
from src.layer1_ai_extraction import train_roberta_baseline
from src.layer1_ai_extraction.errors import ModelUnavailableError
from src.layer1_ai_extraction.roberta_baseline import (
    IGNORE_INDEX,
    LABEL_TO_ID,
    RoBERTaCommitmentExtractor,
    char_spans_to_token_labels,
    decode_entities,
    entities_to_commitment,
    find_token_spans,
    has_trained_weights,
    merge_window_predictions,
    record_to_spans,
)

# Synthetic fixture text. Token offsets below are those of a plain word/punctuation split,
# with (0, 0) for the special tokens a tokenizer adds at both ends.
TEXT = "We support 2.x and 3.x."
OFFSETS = [(0, 0), (0, 2), (3, 10), (11, 12), (12, 13), (13, 14), (15, 18), (19, 20), (20, 21), (21, 22), (22, 23), (0, 0)]
SPANS = [(0, 23, "COMMITMENT"), (11, 14, "VERSION"), (19, 22, "VERSION")]


def names(label_ids):
    return [rb.ID_TO_LABEL[i] if i != IGNORE_INDEX else "-" for i in label_ids]


def test_label_set():
    assert rb.LABEL_LIST == ["O", "B-COMMITMENT", "I-COMMITMENT", "B-VERSION", "I-VERSION", "B-TEMPORAL", "I-TEMPORAL"]
    assert not any("STATUS" in label for label in rb.LABEL_LIST)  # tau is not a span


def test_alignment_with_nested_version_spans():
    labels = char_spans_to_token_labels(OFFSETS, SPANS)
    assert names(labels) == [
        "-", "B-COMMITMENT", "I-COMMITMENT",
        "B-VERSION", "I-VERSION", "I-VERSION",
        "I-COMMITMENT",
        "B-VERSION", "I-VERSION", "I-VERSION",
        "I-COMMITMENT", "-",
    ]


def test_alignment_marks_tokens_outside_spans_as_o():
    labels = char_spans_to_token_labels(OFFSETS, [(3, 10, "COMMITMENT")])
    assert names(labels) == ["-", "O", "B-COMMITMENT", "O", "O", "O", "O", "O", "O", "O", "O", "-"]


def test_alignment_with_a_temporal_span_and_priority():
    labels = char_spans_to_token_labels(OFFSETS, [(0, 23, "COMMITMENT"), (3, 10, "TEMPORAL")])
    assert names(labels)[1:4] == ["B-COMMITMENT", "B-TEMPORAL", "I-COMMITMENT"]


def test_decoding_is_the_inverse_of_alignment():
    entities = decode_entities([(s, e) for s, e in OFFSETS], char_spans_to_token_labels(OFFSETS, SPANS))
    assert entities["commitment"] == (0, 23)
    assert entities["version"] == [(11, 14), (19, 22)]
    assert entities["temporal"] == []


def test_a_version_at_the_edge_of_the_commitment_is_kept_inside_it():
    offsets = [(0, 3), (4, 11), (12, 15)]  # "Use support 2.x"
    labels = [LABEL_TO_ID["B-COMMITMENT"], LABEL_TO_ID["I-COMMITMENT"], LABEL_TO_ID["B-VERSION"]]
    entities = decode_entities(offsets, labels)
    assert entities["commitment"] == (0, 15)
    assert entities["version"] == [(12, 15)]


def test_decoding_picks_the_run_with_most_commitment_tokens():
    o = LABEL_TO_ID["O"]
    offsets = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    labels = [LABEL_TO_ID["B-COMMITMENT"], o, LABEL_TO_ID["B-COMMITMENT"], LABEL_TO_ID["I-COMMITMENT"], LABEL_TO_ID["I-COMMITMENT"], o]
    assert decode_entities(offsets, labels)["commitment"] == (4, 9)


def test_decoding_without_commitment_tokens_returns_nothing():
    labels = [LABEL_TO_ID["B-VERSION"], LABEL_TO_ID["O"]]
    entities = decode_entities([(0, 3), (4, 7)], labels)
    assert entities == {"commitment": None, "version": [], "temporal": []}


def test_two_adjacent_version_spans_stay_separate():
    labels = [LABEL_TO_ID["B-COMMITMENT"], LABEL_TO_ID["B-VERSION"], LABEL_TO_ID["B-VERSION"], LABEL_TO_ID["I-VERSION"]]
    entities = decode_entities([(0, 2), (3, 6), (7, 10), (10, 12)], labels)
    assert entities["version"] == [(3, 6), (7, 12)]


def test_merging_windows_keeps_the_more_confident_prediction():
    first = [(0, 2, 1, 0.9), (3, 5, 2, 0.4), (0, 0, 0, 1.0)]
    second = [(3, 5, 0, 0.8), (6, 8, 1, 0.7)]
    offsets, labels = merge_window_predictions([first, second])
    assert offsets == [(0, 2), (3, 5), (6, 8)]
    assert labels == [1, 0, 1]


def test_spans_to_tuple_rules():
    text = "Supported: 2.x and >= 3.0 only."
    entities = {"commitment": (0, 30), "version": [(11, 14), (19, 25)], "temporal": []}
    result = entities_to_commitment(text, entities)
    assert result["commitment_text"] == text[:30]
    assert result["version_expression"] == ["2.x", ">=3.0"]
    assert result["support_status"] == "explicit_range"
    assert result["temporal_reference"] == "vulnerability_publication_time"


def test_tau_is_derived_not_predicted():
    text = "Only the latest release is supported."
    latest = entities_to_commitment(text, {"commitment": (0, len(text)), "version": [(9, 15)], "temporal": []})
    assert latest["version_expression"] == ["latest"] and latest["support_status"] == "latest_only"
    all_versions = entities_to_commitment("All versions are supported.", {"commitment": (0, 27), "version": [], "temporal": []})
    assert all_versions["support_status"] == "all_versions"
    vague = entities_to_commitment("We help where we can.", {"commitment": (0, 21), "version": [], "temporal": []})
    assert vague["support_status"] == "ambiguous"


def test_temporal_reference_comes_from_the_predicted_span():
    text = "Each version is supported for a year from release."
    entities = {"commitment": (0, len(text)), "version": [], "temporal": [(text.index("release"), len(text) - 1)]}
    assert entities_to_commitment(text, entities)["temporal_reference"] == "release_time"


def test_no_commitment_gives_the_empty_ambiguous_tuple():
    result = entities_to_commitment("anything", {"commitment": None, "version": [], "temporal": []})
    assert result == {
        "commitment_text": "",
        "version_expression": [],
        "support_status": "ambiguous",
        "temporal_reference": "vulnerability_publication_time",
    }


# -- span targets from ground-truth records -----------------------------------------------------


def record(**commitment):
    base = {
        "commitment_text": "We support 2.x and 3.x.",
        "version_expression": ["2.x", "3.x"],
        "support_status": "explicit_range",
        "temporal_reference": "release_time",
    }
    base.update(commitment)
    return {"policy_text": "# Policy\n\nWe support 2.x and 3.x.\n\nSee also 2.x docs and version 12.x.", "commitment": base}


def test_find_token_spans_respects_word_and_version_boundaries():
    text = "2.x 12.x 2.x.1 (2.x) 3.1 3.1.2"
    assert find_token_spans(text, "2.x") == [(0, 3), (16, 19)]
    assert find_token_spans(text, "3.1") == [(21, 24)]


def test_record_to_spans_labels_versions_inside_the_commitment_only():
    rec = record()
    spans, stats = record_to_spans(rec)
    text = rec["policy_text"]
    start = text.index("We support")
    assert (start, start + 23, "COMMITMENT") in spans
    versions = sorted((a, b) for a, b, kind in spans if kind == "VERSION")
    assert [text[a:b] for a, b in versions] == ["2.x", "3.x"]  # "2.x docs" later in the text is not labelled
    assert stats["commitment"] == 1 and stats["version"] == 2 and stats["version_missing"] == 0


def test_record_to_spans_counts_what_it_cannot_locate():
    spans, stats = record_to_spans(record(commitment_text="Not in the text.", version_expression=["9.9.9"]))
    assert spans == []
    assert stats["commitment_missing"] == 1 and stats["version_missing"] == 1


def test_temporal_span_is_trained_only_when_the_record_provides_one():
    plain, _ = record_to_spans(record())
    assert not any(kind == "TEMPORAL" for _, _, kind in plain)
    with_span, stats = record_to_spans(record(commitment_text="We support 2.x and 3.x at release.", temporal_expression="at release"))
    assert stats["temporal"] == 0  # the commitment text is not in this policy, so no region to search
    rec = record(temporal_expression="and")
    spans, stats = record_to_spans(rec)
    assert stats["temporal"] == 1 and any(kind == "TEMPORAL" for _, _, kind in spans)


# -- trained weights and strict mode ------------------------------------------------------------


def test_trained_weights_detection(tmp_path):
    assert not has_trained_weights(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"id2label": {"0": "LABEL_0"}}), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    assert not has_trained_weights(tmp_path)  # wrong label set
    id2label = {str(i): label for i, label in rb.ID_TO_LABEL.items()}
    (tmp_path / "config.json").write_text(json.dumps({"id2label": id2label}), encoding="utf-8")
    assert has_trained_weights(tmp_path)


def test_strict_extract_without_weights_raises(tmp_path):
    with pytest.raises(ModelUnavailableError, match="no trained RoBERTa weights"):
        RoBERTaCommitmentExtractor(model_dir=tmp_path).extract("We support 2.x.")


def test_non_strict_extract_without_weights_is_degraded(tmp_path):
    out = RoBERTaCommitmentExtractor(model_dir=tmp_path, strict=False).extract("We support version 2.x with security fixes.")
    assert out.degraded and out.actual_model_used == "none"
    assert "fallback" in out.extractor_engine and "RoBERTa" not in out.extractor_engine.split("(fallback")[0]


def test_training_hyperparameters_follow_the_thesis():
    config = rb.RobertaTrainingConfig()
    assert (config.model_name, config.learning_rate, config.batch_size, config.num_epochs, config.seed) == (
        "roberta-base", 2e-5, 16, 5, 42,
    )


def test_training_refuses_without_the_annotated_splits(tmp_path):
    with pytest.raises(FileNotFoundError, match="Training data not found"):
        rb.train_roberta(train_path=tmp_path / "train.jsonl", val_path=tmp_path / "val.jsonl", output_dir=tmp_path / "out")
    assert train_roberta_baseline.main(["--train", str(tmp_path / "train.jsonl"), "--val", str(tmp_path / "val.jsonl")]) == 2


# -- windows with a real fast tokenizer (offline, byte-level BPE built in the fixture) -----------


def test_overlapping_windows_reproduce_the_spans(tiny_tokenizer):
    text = "# Policy\n\n" + "Intro sentence about reporting. " * 3 + "We support 2.x and 3.x only.\n\n" + "Closing words. " * 3
    start = text.index("We support")
    commitment = "We support 2.x and 3.x only."
    spans = [(start, start + len(commitment), "COMMITMENT"), (start + 11, start + 14, "VERSION"), (start + 19, start + 22, "VERSION")]
    encoded = tiny_tokenizer(
        text, truncation=True, max_length=40, stride=12, return_overflowing_tokens=True, return_offsets_mapping=True
    )
    assert len(encoded["input_ids"]) > 2
    windows = []
    for offsets in encoded["offset_mapping"]:
        labels = char_spans_to_token_labels(offsets, spans)
        windows.append([(s, e, label, 1.0) for (s, e), label in zip(offsets, labels) if label != IGNORE_INDEX])
    merged_offsets, merged_labels = merge_window_predictions(windows)
    entities = decode_entities(merged_offsets, merged_labels)
    result = entities_to_commitment(text, entities)
    assert result["commitment_text"].strip() == commitment
    assert result["version_expression"] == ["2.x", "3.x"]
    assert result["support_status"] == "explicit_range"


@pytest.mark.network
def test_alignment_with_the_real_roberta_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    except OSError:
        pytest.skip("roberta-base tokenizer is not available offline")
    rec = record()
    spans, _ = record_to_spans(rec)
    encoded = tokenizer(rec["policy_text"], return_offsets_mapping=True)
    labels = char_spans_to_token_labels(encoded["offset_mapping"], spans)
    assert labels[0] == IGNORE_INDEX and labels[-1] == IGNORE_INDEX  # <s> and </s>
    entities = decode_entities(encoded["offset_mapping"], labels)
    assert entities_to_commitment(rec["policy_text"], entities)["version_expression"] == ["2.x", "3.x"]


def test_class_weights_raise_rare_labels():
    from src.layer1_ai_extraction.roberta_baseline import IGNORE_INDEX, label_class_weights

    o, b_version = LABEL_TO_ID["O"], LABEL_TO_ID["B-VERSION"]
    windows = [{"input_ids": [], "labels": [IGNORE_INDEX] + [o] * 99 + [b_version]}]
    weights = label_class_weights(windows)
    assert weights[o] == 1.0
    assert abs(weights[b_version] - 99 ** 0.5) < 1e-9
    assert weights[LABEL_TO_ID["B-TEMPORAL"]] == 1.0  # no training tokens
    assert label_class_weights(windows, "none") == [1.0] * len(weights)
    with pytest.raises(ValueError):
        label_class_weights(windows, "squared")


def test_token_metrics_expose_a_label_that_is_never_predicted():
    from src.layer1_ai_extraction.roberta_baseline import IGNORE_INDEX, token_metrics

    o, b_version = LABEL_TO_ID["O"], LABEL_TO_ID["B-VERSION"]
    gold = [IGNORE_INDEX, o, b_version, b_version]
    scores = token_metrics(gold, [o, o, o, b_version])
    assert scores["B-VERSION"]["gold_tokens"] == 2 and scores["B-VERSION"]["predicted_tokens"] == 1
    assert scores["B-VERSION"]["recall"] == 0.5 and scores["B-VERSION"]["precision"] == 1.0
    assert scores["I-VERSION"]["f1"] == 0.0
