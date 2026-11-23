"""LoRA fine-tuning of Qwen2.5-7B-Instruct for commitment extraction (thesis 3.2.2).

Setup
    LoRA on the attention projections q_proj, k_proj, v_proj, o_proj; r=128, alpha=256, dropout 0.05
    learning rate 2e-4, 3 epochs, effective batch size 8 (per-device batch 1 x 8 accumulation steps)
    seed 42; loss only on the answer tokens; the best epoch by validation loss is kept

LoRA rank
    Thesis 3.2.2 injects low-rank matrices into each layer and 3.3.2 places them in the attention
    layers, training about 1-2 % of the parameters. The thesis gives no rank, so the rank is the one
    that lands in that range for Qwen2.5-7B-Instruct (28 layers, hidden size 3584, grouped-query
    attention with 4 key/value heads of size 128, so k_proj and v_proj output 512 features). A LoRA
    pair on a d_in x d_out projection adds r * (d_in + d_out) parameters:

        q_proj  r * (3584 + 3584) = 7168 r
        k_proj  r * (3584 +  512) = 4096 r
        v_proj  r * (3584 +  512) = 4096 r
        o_proj  r * (3584 + 3584) = 7168 r
        per layer 22528 r; 28 layers 630784 r

    r=128 gives 80,740,352 trainable parameters, about 1.06 % of the 7.62 B base parameters
    (r=16 would give 0.13 %, r=256 2.1 %). alpha = 2 r keeps the usual scaling alpha / r = 2.
    The training run records the measured trainable percentage in run_metadata.json and logs a
    warning when it falls outside 1-2 % (as it does for the tiny test models).

4-bit base model
    The 7B base is loaded 4-bit (NF4, double quantisation) by default so that training fits a
    15 GB Colab T4 (QLoRA). The adapter is applied to the same quantised base at inference.
    `--no-4bit` loads the base in 16-bit instead.

Data
    Reads data/ground_truth/train.jsonl and val.jsonl (INTERFACES.md section 5) and refuses to run if
    either is missing. The script never creates or edits them.

Outputs (default models/qwen_lora_adapters/)
    the PEFT adapter and tokenizer, training_log.json (per-epoch losses) and run_metadata.json
    (hardware, library versions, hyper-parameters, seed, parameter counts, data hashes). All values
    are read from the run that produced them.

Run from the project root:  python -m src.layer1_ai_extraction.train_qwen_lora
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .constrained_decoding import CommitmentGrammar
from .errors import ModelUnavailableError
from .prompting import IM_END, SYSTEM_PROMPT, build_chatml_prompt, prepare_policy_text, serialize_commitment
from .qwen_extractor import (
    BASE_MODEL_NAME,
    DEFAULT_ADAPTER_DIR,
    compute_dtype,
    cuda_available,
    load_base_model,
    missing_modules,
)
from .run_metadata import environment_info, file_sha256, utc_now, write_json
from .seeding import DEFAULT_SEED, set_global_seed
from .training_data import TRAIN_PATH, VAL_PATH, load_jsonl, require_training_files

IGNORE_INDEX = -100

logger = logging.getLogger(__name__)

# Thesis 3.3.2: LoRA trains about 1-2 % of the parameters.
THESIS_TRAINABLE_PERCENT_RANGE = (1.0, 2.0)

# Qwen2.5-7B-Instruct architecture used for the rank calculation in the module docstring.
QWEN25_7B_LAYERS = 28
QWEN25_7B_HIDDEN = 3584
QWEN25_7B_KV_DIM = 512
QWEN25_7B_TOTAL_PARAMETERS = 7_615_616_512


def expected_lora_parameters(
    rank: int, layers: int = QWEN25_7B_LAYERS, hidden: int = QWEN25_7B_HIDDEN, kv_dim: int = QWEN25_7B_KV_DIM
) -> int:
    """Trainable LoRA parameters on q/k/v/o for a Qwen2-style decoder (bias-free adapters)."""
    per_layer = rank * ((hidden + hidden) + 2 * (hidden + kv_dim) + (hidden + hidden))
    return per_layer * layers


def check_trainable_percent(trainable: int, total: int, log: Optional[logging.Logger] = None) -> Tuple[float, bool]:
    """Percentage of trainable parameters and whether it lies in the thesis range of 1-2 %.

    A value outside the range is logged as a warning; training continues.
    """
    percent = 100.0 * trainable / total if total else 0.0
    low, high = THESIS_TRAINABLE_PERCENT_RANGE
    within = low <= percent <= high
    if not within:
        (log or logger).warning(
            "trainable parameters are %.4f %% of the model, outside the thesis range of %.0f-%.0f %% (Section 3.3.2)",
            percent,
            low,
            high,
        )
    return percent, within


@dataclass(frozen=True)
class LoraTrainingConfig:
    base_model: str = BASE_MODEL_NAME
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_seq_len: int = 4096
    early_stopping_patience: int = 1
    seed: int = DEFAULT_SEED
    load_in_4bit: bool = True
    gradient_checkpointing: bool = True
    device: str = "cuda"

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps


# -- data -----------------------------------------------------------------------------------


def build_example(record: Dict[str, Any], tokenizer: Any, max_seq_len: int) -> Tuple[Optional[Dict[str, List[int]]], Optional[str]]:
    """Token ids and loss labels for one record, or (None, reason) when it cannot be used.

    `tokenizer(text, add_special_tokens=False)["input_ids"]` must return the token ids. The prompt
    tokens are masked out of the loss; the answer and the end-of-turn token are trained.
    """
    commitment = record["commitment"]
    target = serialize_commitment(commitment)  # raises on an invalid tuple: gold data must be valid
    policy = prepare_policy_text(record["policy_text"])
    text = commitment["commitment_text"]
    if text and text not in policy:
        return None, "commitment_text is not a verbatim substring of the policy text seen by the model"
    if not CommitmentGrammar().accepts(target):
        return None, "target exceeds the length limits of the constrained decoder"
    prompt_ids = list(tokenizer(build_chatml_prompt(policy), add_special_tokens=False)["input_ids"])
    target_ids = list(tokenizer(target + IM_END, add_special_tokens=False)["input_ids"])
    if len(prompt_ids) + len(target_ids) > max_seq_len:
        return None, f"sequence longer than {max_seq_len} tokens"
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [IGNORE_INDEX] * len(prompt_ids) + target_ids,
    }, None


def build_examples(
    records: Sequence[Dict[str, Any]], tokenizer: Any, max_seq_len: int
) -> Tuple[List[Dict[str, List[int]]], List[Dict[str, str]]]:
    examples, skipped = [], []
    for record in records:
        example, reason = build_example(record, tokenizer, max_seq_len)
        if example is None:
            skipped.append({"package_name": str(record.get("package_name", "?")), "reason": reason or ""})
        else:
            examples.append(example)
    return examples, skipped


def collate(batch: Sequence[Dict[str, List[int]]], pad_id: int) -> Dict[str, Any]:
    """Right-pad a list of examples into tensors."""
    import torch

    width = max(len(example["input_ids"]) for example in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for row, example in enumerate(batch):
        n = len(example["input_ids"])
        input_ids[row, :n] = torch.tensor(example["input_ids"], dtype=torch.long)
        labels[row, :n] = torch.tensor(example["labels"], dtype=torch.long)
        attention[row, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def target_token_count(labels: Any) -> int:
    """Number of tokens that contribute to the shifted language-model loss."""
    return int((labels[:, 1:] != IGNORE_INDEX).sum())


# -- model ----------------------------------------------------------------------------------


def check_training_environment(config: LoraTrainingConfig) -> None:
    required = ["torch", "transformers", "peft"]
    if config.load_in_4bit:
        required += ["bitsandbytes", "accelerate"]
    missing = missing_modules(required)
    if missing:
        raise ModelUnavailableError("missing libraries: " + ", ".join(missing) + " (see requirements-gpu.txt)")
    if config.device.startswith("cuda") and not cuda_available():
        raise ModelUnavailableError("CUDA is not available; training the 7B model needs a GPU (e.g. Colab T4)")
    if config.load_in_4bit and not config.device.startswith("cuda"):
        raise ModelUnavailableError("4-bit loading requires a CUDA device")


def apply_lora(model: Any, config: LoraTrainingConfig) -> Any:
    """Prepare the (possibly 4-bit) base model and attach trainable LoRA adapters."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    checkpointing = {"use_reentrant": False}
    if config.load_in_4bit:
        try:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=config.gradient_checkpointing,
                gradient_checkpointing_kwargs=checkpointing,
            )
        except TypeError:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)
    elif config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpointing)
        model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora)


def count_parameters(model: Any) -> Tuple[int, int]:
    """(trainable, total) parameters; uses PEFT's counter, which accounts for 4-bit weights."""
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
        return int(trainable), int(total)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# -- training loop --------------------------------------------------------------------------


def _autocast(device: str):
    import contextlib

    import torch

    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=compute_dtype())
    return contextlib.nullcontext()


def evaluate_loss(model: Any, examples: Sequence[Dict[str, List[int]]], pad_id: int, device: str, batch_size: int) -> float:
    """Token-weighted mean loss over the examples."""
    import torch

    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = {k: v.to(model.device) for k, v in collate(examples[start : start + batch_size], pad_id).items()}
            tokens = target_token_count(batch["labels"])
            with _autocast(device):
                loss = model(**batch).loss
            total_loss += float(loss) * tokens
            total_tokens += tokens
    return total_loss / max(total_tokens, 1)


def run_training(
    model: Any,
    pad_id: int,
    train_examples: Sequence[Dict[str, List[int]]],
    val_examples: Sequence[Dict[str, List[int]]],
    config: LoraTrainingConfig,
    output_dir: Path,
    tokenizer: Any = None,
    log: Any = print,
) -> Dict[str, Any]:
    """Train the PEFT model, keep the best epoch by validation loss, and return the training log."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    if not train_examples or not val_examples:
        raise ValueError("training and validation sets must both contain usable examples")
    device = config.device
    use_scaler = device.startswith("cuda") and compute_dtype() == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if device.startswith("cuda") else None

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    micro = config.per_device_batch_size
    accumulation = config.gradient_accumulation_steps
    updates_per_epoch = math.ceil(math.ceil(len(train_examples) / micro) / accumulation)
    total_updates = updates_per_epoch * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(config.warmup_ratio * total_updates), total_updates)

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs: List[Dict[str, Any]] = []
    best_loss, best_epoch, stale, stopped_early = math.inf, 0, 0, False
    for epoch in range(1, config.num_epochs + 1):
        started = time.time()
        model.train()
        order = torch.randperm(len(train_examples), generator=torch.Generator().manual_seed(config.seed + epoch)).tolist()
        batches = [order[i : i + micro] for i in range(0, len(order), micro)]
        epoch_loss, epoch_tokens = 0.0, 0
        for group_start in range(0, len(batches), accumulation):
            group = batches[group_start : group_start + accumulation]
            for indices in group:
                batch = {k: v.to(model.device) for k, v in collate([train_examples[i] for i in indices], pad_id).items()}
                tokens = target_token_count(batch["labels"])
                with _autocast(device):
                    loss = model(**batch).loss
                epoch_loss += float(loss.detach()) * tokens
                epoch_tokens += tokens
                scaled = loss / len(group)
                (scaler.scale(scaled) if use_scaler else scaled).backward()
            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        val_loss = evaluate_loss(model, val_examples, pad_id, device, micro)
        entry = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(epoch_tokens, 1),
            "val_loss": val_loss,
            "seconds": round(time.time() - started, 1),
        }
        epochs.append(entry)
        log(f"epoch {epoch}/{config.num_epochs}: train_loss={entry['train_loss']:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            model.save_pretrained(str(output_dir))
            if tokenizer is not None:
                tokenizer.save_pretrained(str(output_dir))
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                stopped_early = epoch < config.num_epochs
                break
    if best_epoch == 0:
        raise RuntimeError("no epoch produced a finite validation loss; no adapter was saved")
    return {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "stopped_early": stopped_early,
        "optimizer_updates_per_epoch": updates_per_epoch,
    }


# -- entry point ----------------------------------------------------------------------------


def train(
    config: LoraTrainingConfig = LoraTrainingConfig(),
    train_path: Path = TRAIN_PATH,
    val_path: Path = VAL_PATH,
    output_dir: Path = DEFAULT_ADAPTER_DIR,
) -> Dict[str, Any]:
    """Fine-tune the adapter and write the adapter, training_log.json and run_metadata.json."""
    train_path, val_path, output_dir = Path(train_path), Path(val_path), Path(output_dir)
    require_training_files(train_path, val_path)
    check_training_environment(config)
    from transformers import AutoTokenizer

    set_global_seed(config.seed)
    started_at = utc_now()
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    train_records, val_records = load_jsonl(train_path), load_jsonl(val_path)
    train_examples, train_skipped = build_examples(train_records, tokenizer, config.max_seq_len)
    val_examples, val_skipped = build_examples(val_records, tokenizer, config.max_seq_len)

    model = apply_lora(load_base_model(config.base_model, config.device, config.load_in_4bit), config)
    trainable, total = count_parameters(model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    trainable_percent, within_range = check_trainable_percent(trainable, total)
    print(f"trainable parameters: {trainable:,} of {total:,} ({trainable_percent:.4f} %)")
    training_log = run_training(model, pad_id, train_examples, val_examples, config, output_dir, tokenizer)

    write_json(output_dir / "training_log.json", training_log)
    write_json(
        output_dir / "run_metadata.json",
        {
            "started_at": started_at,
            "finished_at": utc_now(),
            "base_model": config.base_model,
            "quantization": "nf4 4-bit with double quantisation" if config.load_in_4bit else "none",
            "hyperparameters": asdict(config),
            "effective_batch_size": config.effective_batch_size,
            "seed": config.seed,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_percent": trainable_percent,
            "trainable_percent_thesis_range": list(THESIS_TRAINABLE_PERCENT_RANGE),
            "trainable_percent_within_thesis_range": within_range,
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "data": {
                "train": {
                    "path": str(train_path),
                    "sha256": file_sha256(train_path),
                    "records": len(train_records),
                    "used": len(train_examples),
                    "skipped": train_skipped,
                },
                "val": {
                    "path": str(val_path),
                    "sha256": file_sha256(val_path),
                    "records": len(val_records),
                    "used": len(val_examples),
                    "skipped": val_skipped,
                },
            },
            "environment": environment_info(),
        },
    )
    return training_log


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of Qwen2.5-7B-Instruct for commitment extraction.")
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--val", type=Path, default=VAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--base-model", default=BASE_MODEL_NAME)
    parser.add_argument("--no-4bit", action="store_true", help="load the base model in 16-bit instead of 4-bit NF4")
    parser.add_argument("--device", default="cuda", help="cuda (default) or cpu; cpu only makes sense for a tiny base model")
    parser.add_argument("--epochs", type=int, default=LoraTrainingConfig.num_epochs)
    parser.add_argument("--batch-size", type=int, default=LoraTrainingConfig.per_device_batch_size)
    parser.add_argument("--grad-accum", type=int, default=LoraTrainingConfig.gradient_accumulation_steps)
    parser.add_argument("--max-seq-len", type=int, default=LoraTrainingConfig.max_seq_len)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    config = LoraTrainingConfig(
        base_model=args.base_model,
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        load_in_4bit=not args.no_4bit and args.device.startswith("cuda"),
        device=args.device,
    )
    try:
        train(config, args.train, args.val, args.output_dir)
    except (FileNotFoundError, ModelUnavailableError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
