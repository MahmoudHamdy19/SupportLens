"""Commitment tuple c = <t, tau, v, T> and the Layer 1 output record.

Thesis 3.1.3.2 defines the tuple; INTERFACES.md section 1 fixes the key names:

    commitment_text     t    verbatim sentence(s) from SECURITY.md
    version_expression  v    raw version tokens, e.g. ["5.2.x", ">=3.0"] or ["latest"]
    support_status      tau  latest_only | explicit_range | all_versions | ambiguous
    temporal_reference  T    release_time | vulnerability_publication_time

`is_measurable` is derived from the four elements and cannot be set by the caller.
The remaining keys record which engine produced the tuple.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Mapping, Sequence, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

SupportStatus = Literal["latest_only", "explicit_range", "all_versions", "ambiguous"]
TemporalReference = Literal["release_time", "vulnerability_publication_time"]

SUPPORT_STATUSES: tuple = get_args(SupportStatus)
TEMPORAL_REFERENCES: tuple = get_args(TemporalReference)

# Order of the keys in the generated JSON (thesis 3.2.2) and in `to_dict()`.
COMMITMENT_KEYS = ("commitment_text", "version_expression", "support_status", "temporal_reference")
PROVENANCE_KEYS = ("extractor_engine", "actual_model_used", "lora_applied", "execution_device", "degraded")
OUTPUT_KEYS = COMMITMENT_KEYS + ("is_measurable",) + PROVENANCE_KEYS

DEFAULT_TEMPORAL_REFERENCE: str = "vulnerability_publication_time"


def compute_is_measurable(support_status: str, version_expression: Sequence[str]) -> bool:
    """A commitment is unmeasurable if it is ambiguous, or an explicit range without any version token."""
    if support_status == "ambiguous":
        return False
    if support_status == "explicit_range" and len(version_expression) == 0:
        return False
    return True


class CommitmentExtractionOutput(BaseModel):
    """Layer 1 output: the commitment tuple plus provenance."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    commitment_text: str
    version_expression: List[str] = Field(default_factory=list)
    support_status: SupportStatus
    temporal_reference: TemporalReference

    extractor_engine: str = "unspecified"
    actual_model_used: str = "none"
    lora_applied: bool = False
    execution_device: str = "none"
    degraded: bool = False

    @field_validator("version_expression")
    @classmethod
    def _tokens_are_clean(cls, tokens: List[str]) -> List[str]:
        cleaned = [token.strip() for token in tokens]
        if any(not token for token in cleaned):
            raise ValueError("version_expression must not contain blank tokens")
        return cleaned

    @property
    def is_measurable(self) -> bool:
        return compute_is_measurable(self.support_status, self.version_expression)

    @classmethod
    def empty(cls, **provenance: Any) -> "CommitmentExtractionOutput":
        """Tuple for a policy without a recognisable commitment (ambiguous, no versions)."""
        return cls(
            commitment_text="",
            version_expression=[],
            support_status="ambiguous",
            temporal_reference=DEFAULT_TEMPORAL_REFERENCE,
            **provenance,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CommitmentExtractionOutput":
        """Rebuild an output from `to_dict()`. A stored `is_measurable` must agree with the derived value."""
        payload = dict(data)
        stored = payload.pop("is_measurable", None)
        output = cls(**payload)
        if stored is not None and bool(stored) != output.is_measurable:
            raise ValueError("is_measurable is derived from the tuple and does not match the stored value")
        return output

    def commitment(self) -> Dict[str, Any]:
        """The four thesis elements only."""
        return {
            "commitment_text": self.commitment_text,
            "version_expression": list(self.version_expression),
            "support_status": self.support_status,
            "temporal_reference": self.temporal_reference,
        }

    def to_dict(self) -> Dict[str, Any]:
        result = self.commitment()
        result["is_measurable"] = self.is_measurable
        result["extractor_engine"] = self.extractor_engine
        result["actual_model_used"] = self.actual_model_used
        result["lora_applied"] = self.lora_applied
        result["execution_device"] = self.execution_device
        result["degraded"] = self.degraded
        return result

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)
