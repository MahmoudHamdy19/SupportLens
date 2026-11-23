"""Strict and non-strict behaviour of the Qwen extractor (no GPU, no 7B weights)."""

import json

import pytest

from src.layer1_ai_extraction import qwen_extractor
from src.layer1_ai_extraction.errors import ModelUnavailableError
from src.layer1_ai_extraction.qwen_extractor import QwenCommitmentExtractor
from src.layer1_ai_extraction.schema import COMMITMENT_KEYS, OUTPUT_KEYS, SUPPORT_STATUSES, TEMPORAL_REFERENCES

# Synthetic unit-test fixture, not data.
POLICY = "# Security\n\nOnly the latest release is supported with security updates.\n"


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(qwen_extractor, "cuda_available", lambda: False)


def test_strict_raises_when_the_adapter_is_missing(tmp_path):
    extractor = QwenCommitmentExtractor(lora_dir=tmp_path / "absent", strict=True)
    with pytest.raises(ModelUnavailableError, match="adapter"):
        extractor.extract(POLICY)


def test_strict_raises_for_an_incomplete_adapter_directory(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")  # configuration but no weights
    with pytest.raises(ModelUnavailableError, match="adapter"):
        QwenCommitmentExtractor(lora_dir=tmp_path, strict=True).extract(POLICY)


def test_strict_raises_without_cuda_even_for_the_zero_shot_baseline(no_cuda):
    extractor = QwenCommitmentExtractor(use_adapter=False, strict=True)
    with pytest.raises(ModelUnavailableError, match="CUDA"):
        extractor.extract(POLICY)


def test_strict_never_falls_back(no_cuda, tmp_path):
    extractor = QwenCommitmentExtractor(lora_dir=tmp_path, strict=True)
    for text in (POLICY, ""):
        with pytest.raises(ModelUnavailableError):
            extractor.extract(text)


def test_non_strict_fallback_is_marked_degraded_and_honest(tmp_path):
    out = QwenCommitmentExtractor(lora_dir=tmp_path / "absent", strict=False).extract(POLICY)
    data = out.to_dict()
    assert tuple(data) == OUTPUT_KEYS
    assert data["degraded"] is True
    assert data["lora_applied"] is False
    assert data["actual_model_used"] == "none"
    assert "fallback" in data["extractor_engine"] and "RuleBasedRegexBaseline" in data["extractor_engine"]
    assert "Qwen" not in data["extractor_engine"]
    assert data["execution_device"] == "cpu"
    # the fallback is the regex baseline's tuple
    assert data["support_status"] == "latest_only" and data["version_expression"] == ["latest"]


def test_non_strict_zero_shot_without_cuda_falls_back(no_cuda):
    out = QwenCommitmentExtractor(use_adapter=False, strict=False).extract(POLICY)
    assert out.degraded and not out.lora_applied
    assert "CUDA" in out.extractor_engine


def test_reasons_are_reported_without_loading_anything(no_cuda, tmp_path):
    problems = QwenCommitmentExtractor(lora_dir=tmp_path, strict=True).availability_problems()
    assert any("adapter" in p for p in problems) and any("CUDA" in p for p in problems)


def test_four_bit_loading_needs_cuda(tmp_path):
    problems = QwenCommitmentExtractor(use_adapter=False, device="cpu", load_in_4bit=True).availability_problems()
    assert any("4-bit" in p for p in problems)


def test_adapter_files_are_detected(tmp_path):
    assert not qwen_extractor.adapter_files_present(tmp_path)
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "some/base"}), encoding="utf-8")
    assert not qwen_extractor.adapter_files_present(tmp_path)
    (tmp_path / "adapter_model.safetensors").write_bytes(b"x")
    assert qwen_extractor.adapter_files_present(tmp_path)
    assert qwen_extractor.adapter_base_model(tmp_path) == "some/base"


def test_the_zero_shot_baseline_ignores_the_recorded_adapter_base(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "other/base"}), encoding="utf-8")
    assert QwenCommitmentExtractor(lora_dir=tmp_path, use_adapter=False).base_model_name == qwen_extractor.BASE_MODEL_NAME
    assert QwenCommitmentExtractor(lora_dir=tmp_path, use_adapter=True).base_model_name == "other/base"


# -- the neural path on a tiny random model (CPU, offline) ---------------------------------------


def test_zero_shot_neural_path_on_a_tiny_random_model(tiny_qwen_dir):
    pytest.importorskip("torch")
    extractor = QwenCommitmentExtractor(
        use_adapter=False,
        strict=True,
        device="cpu",
        base_model_name=str(tiny_qwen_dir),
        load_in_4bit=False,
        max_new_tokens=300,
    )
    assert extractor.availability_problems() == []
    out = extractor.extract(POLICY)
    data = out.to_dict()
    assert tuple(data) == OUTPUT_KEYS
    assert data["support_status"] in SUPPORT_STATUSES and data["temporal_reference"] in TEMPORAL_REFERENCES
    assert all(isinstance(data[key], (str, list)) for key in COMMITMENT_KEYS)
    assert data["degraded"] is False
    assert data["lora_applied"] is False
    assert "zero-shot" in data["extractor_engine"] and "LoRA" not in data["extractor_engine"]
    assert data["execution_device"] == "cpu"
    assert data["actual_model_used"] == str(tiny_qwen_dir)
    extractor.release()


def test_strict_empty_text_with_a_loaded_model_returns_the_empty_tuple(tiny_qwen_dir):
    pytest.importorskip("torch")
    extractor = QwenCommitmentExtractor(
        use_adapter=False, strict=True, device="cpu", base_model_name=str(tiny_qwen_dir), load_in_4bit=False
    )
    out = extractor.extract("   ")
    assert out.support_status == "ambiguous" and out.commitment_text == "" and not out.degraded
