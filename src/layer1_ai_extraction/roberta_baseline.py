"""Baseline 3: fine-tuned RoBERTa-base token classifier (thesis 3.3.2, item 3).

Token classification (BIO) over three span types:

    COMMITMENT  the commitment text t
    VERSION     one version token of v (each token of the gold list is one span)
    TEMPORAL    the temporal reference T, when the policy text states one

Fine-tuning follows the thesis: all weights updated, cross-entropy loss, AdamW, learning rate 2e-5,
batch size 16, 5 epochs, seed 42. The learning rate is constant (no schedule) and the weight
decay is the AdamW default of PyTorch (0.01); the last epoch is kept. The cross-entropy is
class-weighted by default (`class_weighting="inverse_sqrt"`): VERSION tokens are a tiny fraction of
all tokens, and with uniform weights the model learned never to predict them (v F1 0.07 in the
first run). Each label weight is sqrt(count(O) / count(label)) over the training windows; labels
without training tokens keep weight 1. `class_weighting="none"` restores the uniform loss. After
training, token-level precision, recall and F1 per label on the validation windows are written to
run_metadata.json (`val_token_metrics`), so a model that never predicts a label is visible at once. Long policies are split into
overlapping 512-token windows.

Span targets come from the ground-truth records (INTERFACES.md section 5): `commitment_text` and
`version_expression` are verbatim substrings of `policy_text`. Version tokens are labelled inside
the commitment span (all occurrences); where a version span lies inside the commitment span the
token carries the VERSION label, which takes priority. Records carry no temporal span text, so a
TEMPORAL span is trained only for records that provide the optional verbatim string
`commitment.temporal_expression`.

tau and T are not spans, and are derived deterministically from the predicted spans:

    t    the contiguous run of predicted non-O tokens that contains most COMMITMENT tokens
    v    the VERSION spans inside that run, whitespace-normalised, without duplicates
    tau  `commitment_rules.derive_support_status(v, t)`
    T    `commitment_rules.temporal_from_span_texts(TEMPORAL spans in the run)`; with no TEMPORAL
         span the default vulnerability_publication_time

If no COMMITMENT token is predicted the result is the empty ambiguous tuple.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .baselines import RuleBasedRegexBaseline
from .commitment_rules import derive_support_status, normalise_version_token, temporal_from_span_texts
from .errors import ModelUnavailableError
from .run_metadata import environment_info, file_sha256, utc_now, write_json
from .schema import CommitmentExtractionOutput
from .seeding import DEFAULT_SEED, set_global_seed
from .training_data import PROJECT_ROOT, TRAIN_PATH, VAL_PATH, load_jsonl, require_training_files

ROBERTA_MODEL_NAME = "roberta-base"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "roberta_baseline"
IGNORE_INDEX = -100

SPAN_TYPES = ("COMMITMENT", "VERSION", "TEMPORAL")
LABEL_LIST = ["O"] + [f"{prefix}-{span}" for span in SPAN_TYPES for prefix in ("B", "I")]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
# When spans overlap, the token takes the label of the first type in this order.
SPAN_PRIORITY = ("VERSION", "TEMPORAL", "COMMITMENT")

Span = Tuple[int, int, str]  # (start char, end char, span type)


# -- span targets from a ground-truth record (pure) -------------------------------------------


def find_token_spans(text: str, token: str, start: int = 0, end: Optional[int] = None) -> List[Tuple[int, int]]:
    """Occurrences of `token` in text[start:end] that are not part of a longer word or version number."""
    end = len(text) if end is None else end
    pattern = r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])(?!\.\d)"
    return [(m.start(), m.end()) for m in re.finditer(pattern, text[:end]) if m.start() >= start]


def record_to_spans(record: Dict[str, Any]) -> Tuple[List[Span], Dict[str, int]]:
    """Character spans for a ground-truth record and counts of what could not be located."""
    text = record["policy_text"]
    commitment = record["commitment"]
    spans: List[Span] = []
    stats = {"commitment": 0, "commitment_missing": 0, "version": 0, "version_missing": 0, "temporal": 0}
    region: Optional[Tuple[int, int]] = None
    commitment_text = commitment.get("commitment_text", "")
    if commitment_text:
        index = text.find(commitment_text)
        if index >= 0:
            region = (index, index + len(commitment_text))
            spans.append((region[0], region[1], "COMMITMENT"))
            stats["commitment"] += 1
        else:
            stats["commitment_missing"] += 1
    for token in commitment.get("version_expression", []):
        found = find_token_spans(text, token, *region) if region else []
        if not found:
            found = find_token_spans(text, token)[:1]
        if found:
            spans.extend((a, b, "VERSION") for a, b in found)
            stats["version"] += 1
        else:
            stats["version_missing"] += 1
    temporal = commitment.get("temporal_expression")
    if temporal:
        found = find_token_spans(text, temporal, *region)[:1] if region else []
        if found:
            spans.append((found[0][0], found[0][1], "TEMPORAL"))
            stats["temporal"] += 1
    return spans, stats


# -- BIO alignment and decoding (pure, tokenizer independent) ---------------------------------


def char_spans_to_token_labels(offsets: Sequence[Tuple[int, int]], spans: Sequence[Span]) -> List[int]:
    """BIO label ids for tokens given their character offsets.

    A token overlapping a span belongs to it; special and padding tokens (empty offsets) get
    IGNORE_INDEX. Where spans overlap, SPAN_PRIORITY decides. The first token of each span
    instance is `B-`, later tokens are `I-`, also after an interrupting nested span.
    """
    labels: List[int] = []
    seen = set()
    for start, end in offsets:
        if start == end:
            labels.append(IGNORE_INDEX)
            continue
        chosen: Optional[int] = None
        for span_type in SPAN_PRIORITY:
            for index, (s, e, kind) in enumerate(spans):
                if kind == span_type and start < e and end > s:
                    chosen = index
                    break
            if chosen is not None:
                break
        if chosen is None:
            labels.append(LABEL_TO_ID["O"])
            continue
        prefix = "I" if chosen in seen else "B"
        seen.add(chosen)
        labels.append(LABEL_TO_ID[f"{prefix}-{spans[chosen][2]}"])
    return labels


def decode_entities(offsets: Sequence[Tuple[int, int]], label_ids: Sequence[int]) -> Dict[str, Any]:
    """Character spans of the predicted commitment and of the VERSION and TEMPORAL spans inside it."""
    tokens = [
        (start, end, ID_TO_LABEL.get(int(label), "O"))
        for (start, end), label in zip(offsets, label_ids)
        if start != end and int(label) != IGNORE_INDEX
    ]
    runs: List[List[Tuple[int, int, str]]] = []
    current: List[Tuple[int, int, str]] = []
    for token in tokens:
        if token[2] == "O":
            if current:
                runs.append(current)
            current = []
        else:
            current.append(token)
    if current:
        runs.append(current)

    best: Optional[List[Tuple[int, int, str]]] = None
    best_count = 0
    for run in runs:
        count = sum(1 for _, _, label in run if label.endswith("COMMITMENT"))
        if count > best_count:
            best, best_count = run, count
    if best is None:
        return {"commitment": None, "version": [], "temporal": []}

    def entities(kind: str) -> List[Tuple[int, int]]:
        found: List[Tuple[int, int]] = []
        open_span: Optional[List[int]] = None
        for start, end, label in best:
            if label.endswith(kind):
                if label.startswith("I-") and open_span is not None:
                    open_span[1] = end
                    continue
                if open_span is not None:
                    found.append((open_span[0], open_span[1]))
                open_span = [start, end]
            elif open_span is not None:
                found.append((open_span[0], open_span[1]))
                open_span = None
        if open_span is not None:
            found.append((open_span[0], open_span[1]))
        return found

    return {
        "commitment": (best[0][0], best[-1][1]),
        "version": entities("VERSION"),
        "temporal": entities("TEMPORAL"),
    }


def merge_window_predictions(windows: Sequence[Sequence[Tuple[int, int, int, float]]]) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Merge per-window token predictions (start, end, label id, probability) into one sequence.

    Overlapping windows predict the same token twice; the more confident prediction is kept.
    """
    best: Dict[Tuple[int, int], Tuple[int, float]] = {}
    for window in windows:
        for start, end, label, probability in window:
            if start == end:
                continue
            key = (start, end)
            if key not in best or probability > best[key][1]:
                best[key] = (label, probability)
    ordered = sorted(best)
    return ordered, [best[key][0] for key in ordered]


def entities_to_commitment(text: str, entities: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the deterministic rules that turn predicted spans into the tuple c = <t, tau, v, T>."""
    if entities["commitment"] is None:
        return CommitmentExtractionOutput.empty().commitment()
    start, end = entities["commitment"]
    commitment_text = text[start:end]
    tokens: List[str] = []
    for a, b in entities["version"]:
        token = normalise_version_token(text[a:b])
        if token and token not in tokens:
            tokens.append(token)
    return {
        "commitment_text": commitment_text,
        "version_expression": tokens,
        "support_status": derive_support_status(tokens, commitment_text),
        "temporal_reference": temporal_from_span_texts(text[a:b] for a, b in entities["temporal"]),
    }


# -- trained-model location -------------------------------------------------------------------


def has_trained_weights(model_dir: Path) -> bool:
    """True if `model_dir` holds a saved token-classification model with this label set."""
    config_path = model_dir / "config.json"
    weights = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
    if not (config_path.exists() and weights):
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            id2label = json.load(handle).get("id2label", {})
    except (OSError, ValueError):
        return False
    return sorted(id2label.values()) == sorted(LABEL_LIST)


# -- training ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class RobertaTrainingConfig:
    model_name: str = ROBERTA_MODEL_NAME
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 5
    weight_decay: float = 0.01
    max_length: int = 512
    stride: int = 128
    seed: int = DEFAULT_SEED
    class_weighting: str = "inverse_sqrt"


CLASS_WEIGHTINGS = ("inverse_sqrt", "none")


def label_class_weights(windows: Sequence[Dict[str, List[int]]], scheme: str = "inverse_sqrt") -> List[float]:
    """Cross-entropy weight per label id: sqrt(count(O) / count(label)), 1.0 for unseen labels."""
    if scheme not in CLASS_WEIGHTINGS:
        raise ValueError(f"class_weighting must be one of {CLASS_WEIGHTINGS}, got {scheme!r}")
    counts = [0] * len(LABEL_LIST)
    for window in windows:
        for label in window["labels"]:
            if label != IGNORE_INDEX:
                counts[label] += 1
    if scheme == "none" or counts[LABEL_TO_ID["O"]] == 0:
        return [1.0] * len(LABEL_LIST)
    reference = counts[LABEL_TO_ID["O"]]
    return [math.sqrt(reference / count) if count else 1.0 for count in counts]


def token_metrics(gold: Sequence[int], predicted: Sequence[int]) -> Dict[str, Dict[str, float]]:
    """Token-level precision, recall, F1 and counts per non-O label (IGNORE_INDEX positions skipped)."""
    result: Dict[str, Dict[str, float]] = {}
    pairs = [(g, p) for g, p in zip(gold, predicted) if g != IGNORE_INDEX]
    for label_id, label in ID_TO_LABEL.items():
        if label == "O":
            continue
        tp = sum(1 for g, p in pairs if g == label_id and p == label_id)
        n_gold = sum(1 for g, _ in pairs if g == label_id)
        n_pred = sum(1 for _, p in pairs if p == label_id)
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gold if n_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[label] = {"precision": precision, "recall": recall, "f1": f1, "gold_tokens": n_gold, "predicted_tokens": n_pred}
    return result


def encode_record(record: Dict[str, Any], tokenizer: Any, config: RobertaTrainingConfig) -> Tuple[List[Dict[str, List[int]]], Dict[str, int]]:
    """Training windows (input ids and BIO labels) for one record."""
    spans, stats = record_to_spans(record)
    encoded = tokenizer(
        record["policy_text"],
        truncation=True,
        max_length=config.max_length,
        stride=config.stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    windows = []
    for ids, offsets in zip(encoded["input_ids"], encoded["offset_mapping"]):
        windows.append({"input_ids": list(ids), "labels": char_spans_to_token_labels(offsets, spans)})
    return windows, stats


def _collate(batch: Sequence[Dict[str, List[int]]], pad_id: int) -> Dict[str, Any]:
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


def _mean_loss(model: Any, windows: Sequence[Dict[str, List[int]]], pad_id: int, batch_size: int, device: str) -> float:
    import torch

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = {k: v.to(device) for k, v in _collate(windows[start : start + batch_size], pad_id).items()}
            n = int((batch["labels"] != IGNORE_INDEX).sum())
            total += float(model(**batch).loss) * n
            count += n
    return total / max(count, 1)


def train_roberta(
    config: RobertaTrainingConfig = RobertaTrainingConfig(),
    train_path: Path = TRAIN_PATH,
    val_path: Path = VAL_PATH,
    output_dir: Path = DEFAULT_MODEL_DIR,
    device: Optional[str] = None,
    log: Any = print,
) -> Dict[str, Any]:
    """Fine-tune RoBERTa-base; write the model, training_log.json and run_metadata.json."""
    train_path, val_path, output_dir = Path(train_path), Path(val_path), Path(output_dir)
    require_training_files(train_path, val_path)
    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise ModelUnavailableError(f"torch and transformers are required to train RoBERTa: {exc}") from exc

    set_global_seed(config.seed)
    started_at = utc_now()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_records, val_records = load_jsonl(train_path), load_jsonl(val_path)
    totals = {"commitment": 0, "commitment_missing": 0, "version": 0, "version_missing": 0, "temporal": 0}
    split_windows: Dict[str, List[Dict[str, List[int]]]] = {}
    for name, records in (("train", train_records), ("val", val_records)):
        windows: List[Dict[str, List[int]]] = []
        for record in records:
            record_windows, stats = encode_record(record, tokenizer, config)
            windows.extend(record_windows)
            if name == "train":
                for key, value in stats.items():
                    totals[key] += value
        split_windows[name] = windows
    if not split_windows["train"] or not split_windows["val"]:
        raise ValueError("training and validation sets must both contain records")

    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name, num_labels=len(LABEL_LIST), id2label=ID_TO_LABEL, label2id=LABEL_TO_ID
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    pad_id = tokenizer.pad_token_id
    epochs: List[Dict[str, Any]] = []
    train = split_windows["train"]
    weights = label_class_weights(train, config.class_weighting)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device), ignore_index=IGNORE_INDEX)
    log("label weights: " + ", ".join(f"{label}={w:.2f}" for label, w in zip(LABEL_LIST, weights)))
    for epoch in range(1, config.num_epochs + 1):
        started = time.time()
        model.train()
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(config.seed + epoch)).tolist()
        total, count = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            batch = {
                k: v.to(device)
                for k, v in _collate([train[i] for i in order[start : start + config.batch_size]], pad_id).items()
            }
            n = int((batch["labels"] != IGNORE_INDEX).sum())
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), batch["labels"].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * n
            count += n
        entry = {
            "epoch": epoch,
            "train_loss": total / max(count, 1),
            "val_loss": _mean_loss(model, split_windows["val"], pad_id, config.batch_size, device),
            "seconds": round(time.time() - started, 1),
        }
        epochs.append(entry)
        log(f"epoch {epoch}/{config.num_epochs}: train_loss={entry['train_loss']:.4f} val_loss={entry['val_loss']:.4f}")

    gold_ids: List[int] = []
    predicted_ids: List[int] = []
    model.eval()
    with torch.no_grad():
        val_windows = split_windows["val"]
        for start in range(0, len(val_windows), config.batch_size):
            batch = {k: v.to(device) for k, v in _collate(val_windows[start : start + config.batch_size], pad_id).items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            gold_ids.extend(batch["labels"].reshape(-1).tolist())
            predicted_ids.extend(logits.argmax(-1).reshape(-1).tolist())
    val_token_metrics = token_metrics(gold_ids, predicted_ids)
    for label, scores in val_token_metrics.items():
        log(f"val {label}: f1={scores['f1']:.3f} gold={scores['gold_tokens']} predicted={scores['predicted_tokens']}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    training_log = {"epochs": epochs, "kept_epoch": config.num_epochs}
    write_json(output_dir / "training_log.json", training_log)
    write_json(
        output_dir / "run_metadata.json",
        {
            "started_at": started_at,
            "finished_at": utc_now(),
            "hyperparameters": asdict(config),
            "device": device,
            "seed": config.seed,
            "parameters": sum(p.numel() for p in model.parameters()),
            "labels": LABEL_LIST,
            "label_weights": dict(zip(LABEL_LIST, weights)),
            "val_token_metrics": val_token_metrics,
            "data": {
                "train": {"path": str(train_path), "sha256": file_sha256(train_path), "records": len(train_records),
                          "windows": len(train), "span_counts": totals},
                "val": {"path": str(val_path), "sha256": file_sha256(val_path), "records": len(val_records),
                        "windows": len(split_windows["val"])},
            },
            "environment": environment_info(),
        },
    )
    return training_log


# -- inference --------------------------------------------------------------------------------


class RoBERTaCommitmentExtractor:
    """Fine-tuned RoBERTa span extractor.

    `strict=True` (default, as for every evaluated baseline) raises `ModelUnavailableError` when no
    trained weights exist in `model_dir`. `strict=False` falls back to the regex baseline with
    `degraded=True`.
    """

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        strict: bool = True,
        device: Optional[str] = None,
        max_length: int = 512,
        stride: int = 128,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        self.strict = strict
        self.device = device
        self.max_length = max_length
        self.stride = stride
        self._model: Any = None
        self._tokenizer: Any = None

    def train(self, train_path: Path = TRAIN_PATH, val_path: Path = VAL_PATH, config: Optional[RobertaTrainingConfig] = None) -> Dict[str, Any]:
        """Fine-tune into `model_dir` and drop any previously loaded weights."""
        self._model = self._tokenizer = None
        return train_roberta(config or RobertaTrainingConfig(), train_path, val_path, self.model_dir, self.device)

    def availability_problems(self) -> List[str]:
        problems = []
        if not has_trained_weights(self.model_dir):
            problems.append(f"no trained RoBERTa weights in {self.model_dir}")
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            problems.append(f"missing library: {exc.name}")
        return problems

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self._model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir)).to(self.device)
        except (OSError, ValueError) as exc:
            raise ModelUnavailableError(f"cannot load RoBERTa from {self.model_dir}: {exc}") from exc
        self._model.eval()

    def predict_commitment(self, text: str) -> Dict[str, Any]:
        """Run the token classifier over all windows and apply the span-to-tuple rules."""
        import torch

        self._load()
        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        windows = []
        for ids, offsets in zip(encoded["input_ids"], encoded["offset_mapping"]):
            with torch.no_grad():
                logits = self._model(input_ids=torch.tensor([ids], device=self.device)).logits[0]
            probabilities, labels = torch.softmax(logits, dim=-1).max(dim=-1)
            windows.append([(s, e, int(l), float(p)) for (s, e), l, p in zip(offsets, labels.tolist(), probabilities.tolist())])
        merged_offsets, merged_labels = merge_window_predictions(windows)
        return entities_to_commitment(text, decode_entities(merged_offsets, merged_labels))

    def extract(self, text: str) -> CommitmentExtractionOutput:
        problems = self.availability_problems()
        if problems:
            reason = "; ".join(problems)
            if self.strict:
                raise ModelUnavailableError(reason)
            baseline = RuleBasedRegexBaseline().extract(text)
            return CommitmentExtractionOutput(
                **baseline.commitment(),
                extractor_engine=f"{baseline.extractor_engine} (fallback: {reason})",
                actual_model_used="none",
                lora_applied=False,
                execution_device="cpu",
                degraded=True,
            )
        provenance = dict(
            extractor_engine="RoBERTa-base token classification (fine-tuned)",
            actual_model_used=ROBERTA_MODEL_NAME + " (fine-tuned)",
            lora_applied=False,
        )
        if not text or not text.strip():
            return CommitmentExtractionOutput.empty(execution_device="none", **provenance)
        commitment = self.predict_commitment(text)
        return CommitmentExtractionOutput(**commitment, execution_device=str(self.device), **provenance)
