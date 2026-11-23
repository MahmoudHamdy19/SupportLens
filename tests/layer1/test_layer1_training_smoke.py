"""End-to-end training smoke tests on tiny random models (CPU, seconds).

They check that the training loops, the saved artefacts, the run metadata and the strict extractors
work together. Random weights learn nothing useful; no quality claim is made. The records are
synthetic unit-test fixtures, not data.
"""

import json

import pytest

from src.layer1_ai_extraction import roberta_baseline as rb
from src.layer1_ai_extraction import train_qwen_lora as tq
from src.layer1_ai_extraction.qwen_extractor import QwenCommitmentExtractor
from src.layer1_ai_extraction.roberta_baseline import RoBERTaCommitmentExtractor


def synthetic_record(i):
    version = f"{i + 1}.x"
    sentence = f"We support {version} with security fixes."
    return {
        "package_name": f"pkg{i}",
        "policy_text": f"# Security\n\n{sentence}\n\nReport by email.\n",
        "commitment": {
            "commitment_text": sentence,
            "version_expression": [version],
            "support_status": "explicit_range",
            "temporal_reference": "vulnerability_publication_time",
        },
    }


@pytest.fixture
def splits(tmp_path):
    for name, count in (("train", 8), ("val", 3)):
        lines = [json.dumps(synthetic_record(i)) for i in range(count)]
        (tmp_path / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path / "train.jsonl", tmp_path / "val.jsonl"


def test_lora_training_then_strict_inference_with_the_adapter(tiny_qwen_dir, splits, tmp_path):
    pytest.importorskip("peft")
    train_path, val_path = splits
    output_dir = tmp_path / "adapter"
    config = tq.LoraTrainingConfig(
        base_model=str(tiny_qwen_dir),
        num_epochs=2,
        per_device_batch_size=2,
        gradient_accumulation_steps=2,
        load_in_4bit=False,
        gradient_checkpointing=False,
        device="cpu",
        learning_rate=1e-3,
        early_stopping_patience=5,
    )
    log = tq.train(config, train_path, val_path, output_dir)

    assert (output_dir / "adapter_config.json").exists()
    assert (output_dir / "adapter_model.safetensors").exists()
    assert len(log["epochs"]) == 2 and log["best_epoch"] in (1, 2)
    assert all(entry["val_loss"] == entry["val_loss"] and entry["train_loss"] > 0 for entry in log["epochs"])

    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["hyperparameters"]["lora_r"] == 128 and metadata["hyperparameters"]["lora_alpha"] == 256
    assert metadata["trainable_percent_thesis_range"] == [1.0, 2.0]
    assert metadata["trainable_percent_within_thesis_range"] == (1.0 <= metadata["trainable_percent"] <= 2.0)
    assert metadata["hyperparameters"]["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert metadata["seed"] == 42
    assert 0 < metadata["trainable_parameters"] < metadata["total_parameters"]
    assert metadata["trainable_percent"] == pytest.approx(
        100 * metadata["trainable_parameters"] / metadata["total_parameters"]
    )
    assert metadata["data"]["train"]["used"] == 8 and metadata["data"]["val"]["used"] == 3
    assert len(metadata["data"]["train"]["sha256"]) == 64
    assert metadata["hyperparameters"]["device"] == "cpu"  # this run was on the CPU
    if not metadata["environment"]["hardware"]["cuda_available"]:
        assert metadata["environment"]["hardware"]["gpu_name"] is None
    assert metadata["environment"]["libraries"]["peft"] is not None
    assert json.loads((output_dir / "training_log.json").read_text(encoding="utf-8")) == log

    extractor = QwenCommitmentExtractor(
        lora_dir=output_dir, use_adapter=True, strict=True, device="cpu", load_in_4bit=False, max_new_tokens=300
    )
    assert extractor.base_model_name == str(tiny_qwen_dir)  # recorded in the adapter configuration
    out = extractor.extract(synthetic_record(0)["policy_text"])
    assert out.lora_applied is True and out.degraded is False
    assert "+ LoRA" in out.extractor_engine and "zero-shot" not in out.extractor_engine
    extractor.release()


def test_lora_training_selects_the_best_epoch_by_validation_loss(tiny_qwen_dir, splits, tmp_path):
    pytest.importorskip("peft")
    config = tq.LoraTrainingConfig(
        base_model=str(tiny_qwen_dir),
        num_epochs=3,
        per_device_batch_size=2,
        gradient_accumulation_steps=1,
        load_in_4bit=False,
        gradient_checkpointing=False,
        device="cpu",
        learning_rate=5e-2,  # large enough that the validation loss need not fall monotonically
        early_stopping_patience=1,
    )
    log = tq.train(config, *splits, tmp_path / "adapter")
    losses = [entry["val_loss"] for entry in log["epochs"]]
    assert log["best_val_loss"] == min(losses)
    assert log["best_epoch"] == losses.index(min(losses)) + 1
    assert log["stopped_early"] == (len(losses) < 3)


@pytest.fixture
def tiny_roberta_dir(tmp_path_factory):
    """Randomly initialised tiny RoBERTa token classifier with the real roberta-base tokenizer."""
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base")
    except OSError:
        pytest.skip("roberta-base tokenizer is not available offline")
    config = transformers.RobertaConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=514,
        num_labels=len(rb.LABEL_LIST),
        id2label=rb.ID_TO_LABEL,
        label2id=rb.LABEL_TO_ID,
    )
    directory = tmp_path_factory.mktemp("tiny_roberta")
    transformers.RobertaForTokenClassification(config).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory


@pytest.mark.network
def test_roberta_training_then_strict_extraction(tiny_roberta_dir, splits, tmp_path):
    train_path, val_path = splits
    output_dir = tmp_path / "roberta"
    config = rb.RobertaTrainingConfig(model_name=str(tiny_roberta_dir), num_epochs=2, batch_size=4, learning_rate=1e-3)
    log = rb.train_roberta(config, train_path, val_path, output_dir, device="cpu")
    assert [entry["epoch"] for entry in log["epochs"]] == [1, 2]
    assert rb.has_trained_weights(output_dir)

    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["hyperparameters"]["learning_rate"] == 1e-3 and metadata["seed"] == 42
    assert metadata["data"]["train"]["span_counts"]["commitment"] == 8
    assert metadata["data"]["train"]["span_counts"]["version"] == 8

    out = RoBERTaCommitmentExtractor(model_dir=output_dir, strict=True, device="cpu").extract("We support 2.x with security fixes.")
    data = out.to_dict()
    assert data["extractor_engine"].startswith("RoBERTa-base") and data["degraded"] is False
    assert data["lora_applied"] is False and data["execution_device"] == "cpu"
    assert data["support_status"] in ("latest_only", "explicit_range", "all_versions", "ambiguous")
    assert out.commitment_text == "" or out.commitment_text in "We support 2.x with security fixes."
