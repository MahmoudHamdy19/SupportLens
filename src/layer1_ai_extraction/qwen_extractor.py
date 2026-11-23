"""Qwen2.5-7B-Instruct commitment extractor (thesis 3.2.2).

Greedy decoding of one JSON object, strictly constrained to the commitment schema by
`constrained_decoding`. With `use_adapter=True` the LoRA adapter trained by `train_qwen_lora`
is applied; with `use_adapter=False` the same base model, prompt, schema and constraint run
without the adapter (zero-shot baseline, thesis 3.3.2 item 2).

    strict=True   Never falls back. A missing adapter, library, CUDA device or model raises
                  `ModelUnavailableError`. Used by every evaluation script.
    strict=False  A missing prerequisite or a failed run falls back to the regex baseline. The
                  result then has `degraded=True` and an `extractor_engine` that names the fallback.

`lora_applied` is True only if a PEFT adapter is actually attached to the model that ran.
`device=None` means CUDA; pass `device="cpu"` explicitly to run a small model without a GPU.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .baselines import RuleBasedRegexBaseline
from .constrained_decoding import ConstraintEngine, TokenVocabulary
from .errors import ExtractionFormatError, ModelUnavailableError
from .prompting import build_chatml_prompt, prepare_policy_text
from .schema import CommitmentExtractionOutput

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "models" / "qwen_lora_adapters"
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 1024


# -- environment helpers (shared with train_qwen_lora) ---------------------------------------


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def missing_modules(names: List[str]) -> List[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def adapter_files_present(adapter_dir: Path) -> bool:
    """A PEFT adapter directory holds `adapter_config.json` and the adapter weights."""
    weights = (adapter_dir / "adapter_model.safetensors").exists() or (adapter_dir / "adapter_model.bin").exists()
    return (adapter_dir / "adapter_config.json").exists() and weights


def adapter_base_model(adapter_dir: Path) -> Optional[str]:
    """Base model recorded in the adapter configuration, if any."""
    try:
        with open(adapter_dir / "adapter_config.json", "r", encoding="utf-8") as handle:
            return json.load(handle).get("base_model_name_or_path")
    except (OSError, ValueError):
        return None


def device_index(device: str) -> int:
    return int(device.split(":", 1)[1]) if ":" in device else 0


def compute_dtype() -> Any:
    """bfloat16 on Ampere or newer GPUs, float16 otherwise (a T4 has no native bfloat16)."""
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def dtype_kwarg(dtype: Any) -> Dict[str, Any]:
    """`from_pretrained` renamed `torch_dtype` to `dtype` in transformers 4.56."""
    import transformers

    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def build_quantization_config() -> Any:
    """NF4 4-bit quantisation with double quantisation; the same for training and inference."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype(),
    )


def load_base_model(base_model_name: str, device: str, load_in_4bit: bool) -> Any:
    """Load the causal LM, 4-bit quantised on the GPU or in full precision on the requested device."""
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: Dict[str, Any] = dtype_kwarg(compute_dtype() if device.startswith("cuda") else torch.float32)
    if load_in_4bit:
        kwargs["quantization_config"] = build_quantization_config()
        kwargs["device_map"] = {"": device_index(device)}
    model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
    if not load_in_4bit:
        model.to(device)
    return model


# -- extractor ----------------------------------------------------------------------------------


class QwenCommitmentExtractor:
    """Schema-constrained Qwen2.5-7B-Instruct extractor, with or without the LoRA adapter."""

    def __init__(
        self,
        lora_dir: Optional[Path] = None,
        use_adapter: bool = True,
        strict: bool = False,
        device: Optional[str] = None,
        base_model_name: Optional[str] = None,
        load_in_4bit: Optional[bool] = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self.lora_dir = Path(lora_dir) if lora_dir is not None else DEFAULT_ADAPTER_DIR
        self.use_adapter = use_adapter
        self.strict = strict
        self.device = device or "cuda"
        self.load_in_4bit = self.device.startswith("cuda") if load_in_4bit is None else load_in_4bit
        self.max_new_tokens = max_new_tokens
        recorded_base = adapter_base_model(self.lora_dir) if use_adapter else None
        self.base_model_name = base_model_name or recorded_base or BASE_MODEL_NAME
        self._model: Any = None
        self._tokenizer: Any = None
        self._engine: Optional[ConstraintEngine] = None
        self._lora_applied = False

    # -- availability -------------------------------------------------------------------------

    def availability_problems(self) -> List[str]:
        """Reasons the neural extractor cannot run now (empty list means it can). Loads nothing."""
        if self._model is not None:
            return []
        problems: List[str] = []
        if self.use_adapter and not adapter_files_present(self.lora_dir):
            problems.append(f"LoRA adapter not found in {self.lora_dir}")
        required = ["torch", "transformers"]
        if self.use_adapter:
            required.append("peft")
        if self.load_in_4bit:
            required += ["bitsandbytes", "accelerate"]
        missing = missing_modules(required)
        if missing:
            problems.append("missing libraries: " + ", ".join(missing))
        if self.device.startswith("cuda") and not cuda_available():
            problems.append("CUDA is not available")
        if self.load_in_4bit and not self.device.startswith("cuda"):
            problems.append("4-bit loading requires a CUDA device")
        return problems

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer

            tokenizer_source = (
                str(self.lora_dir)
                if self.use_adapter and (self.lora_dir / "tokenizer_config.json").exists()
                else self.base_model_name
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
            model = load_base_model(self.base_model_name, self.device, self.load_in_4bit)
            if self.use_adapter:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(self.lora_dir))
            model.eval()
            engine = ConstraintEngine(TokenVocabulary.from_tokenizer(tokenizer))
        except (OSError, ImportError, ValueError, RuntimeError) as exc:
            raise ModelUnavailableError(f"cannot load {self.base_model_name}: {type(exc).__name__}: {exc}") from exc
        self._tokenizer, self._model, self._engine = tokenizer, model, engine
        self._lora_applied = bool(self.use_adapter and getattr(model, "peft_config", None))

    def release(self) -> None:
        """Free the model and GPU memory."""
        self._model = self._tokenizer = self._engine = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # -- provenance ---------------------------------------------------------------------------

    def _provenance(self) -> Dict[str, Any]:
        short = self.base_model_name.split("/")[-1]
        quantisation = " (4-bit NF4)" if self.load_in_4bit else ""
        suffix = "+ LoRA" if self._lora_applied else "zero-shot"
        return dict(
            extractor_engine=f"{short} {suffix} (schema-constrained)",
            actual_model_used=self.base_model_name + quantisation,
            lora_applied=self._lora_applied,
            execution_device=str(self._model.device),
            degraded=False,
        )

    # -- inference ----------------------------------------------------------------------------

    def generate_json(self, policy_text: str) -> str:
        """Run the constrained greedy decoder and return the generated JSON text."""
        import torch
        from transformers import LogitsProcessorList

        self._ensure_loaded()
        prompt = build_chatml_prompt(prepare_policy_text(policy_text))
        encoded = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self._model.device)
        prompt_length = int(encoded["input_ids"].shape[1])
        processor = self._engine.new_processor(prompt_length, self.max_new_tokens)
        eos_id = self._engine.vocab.eos_id
        generate_kwargs = {k: v for k, v in encoded.items() if k != "token_type_ids"}
        with torch.no_grad():
            output = self._model.generate(
                **generate_kwargs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                logits_processor=LogitsProcessorList([processor]),
                eos_token_id=eos_id,
                pad_token_id=self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else eos_id,
            )
        return self._tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)

    def _parse(self, generated: str) -> CommitmentExtractionOutput:
        try:
            data = json.loads(generated)
            tokens: List[str] = []
            for token in data["version_expression"]:
                token = token.strip()
                if token and token not in tokens:
                    tokens.append(token)
            return CommitmentExtractionOutput(
                commitment_text=data["commitment_text"].strip(),
                version_expression=tokens,
                support_status=data["support_status"],
                temporal_reference=data["temporal_reference"],
                **self._provenance(),
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ExtractionFormatError(f"constrained output is not a valid commitment: {generated!r}") from exc

    def _fallback(self, text: str, reason: str) -> CommitmentExtractionOutput:
        log.warning("Qwen extractor unavailable (%s); using the regex baseline.", reason)
        baseline = RuleBasedRegexBaseline().extract(text)
        return CommitmentExtractionOutput(
            **baseline.commitment(),
            extractor_engine=f"{baseline.extractor_engine} (fallback: {reason})",
            actual_model_used="none",
            lora_applied=False,
            execution_device="cpu",
            degraded=True,
        )

    def extract(self, text: str) -> CommitmentExtractionOutput:
        """Extract the commitment tuple from the policy text."""
        problems = self.availability_problems()
        if problems:
            reason = "; ".join(problems)
            if self.strict:
                raise ModelUnavailableError(reason)
            return self._fallback(text, reason)
        if not text or not text.strip():
            return CommitmentExtractionOutput.empty(
                extractor_engine="none (empty policy text)",
                actual_model_used="none",
                lora_applied=False,
                execution_device="none",
            )
        try:
            return self._parse(self.generate_json(text))
        except Exception as exc:
            if self.strict:
                raise
            return self._fallback(text, f"{type(exc).__name__}: {exc}")
