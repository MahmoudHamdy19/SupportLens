"""
SupportLens web interface (Streamlit).

Pages:
  1. Audit a PyPI package: runs Layers 0 to 4 live and shows every layer's output.
  2. Evaluation results: shows the result files written by scripts/run_evaluation.py (nothing is computed here).
  3. Dataset status: counts of collected, annotated and split records.
Run: streamlit run app.py
"""

import json
import os
from pathlib import Path

import streamlit as st

from src.supportlens_pipeline import SupportLensPipeline

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "data" / "evaluation"
VERDICT_COLORS = {
    "Consistent": "#2E7D5B",
    "Partially Consistent": "#C77D0A",
    "Potential Inconsistency": "#B83227",
    "Insufficient Evidence": "#52667A",
    "Unmeasurable Commitment": "#6B4FA0",
}


def load_env(path: Path = ROOT / ".env") -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.split("#", 1)[0].strip())


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def page_audit() -> None:
    st.header("Audit a PyPI package")
    st.caption("Layer 0 acquisition, Layer 1 extraction, Layer 2 evidence, Layer 3 Algorithm 1, Layer 4 explanation.")
    package = st.text_input("PyPI package name", value="urllib3").strip()
    strict = st.checkbox("Require the fine-tuned Qwen model (no regex fallback)", value=False)
    if not st.button("Run audit") or not package:
        return
    with st.spinner(f"Auditing {package} ..."):
        try:
            result = SupportLensPipeline(github_token=os.environ.get("GITHUB_TOKEN"), strict=strict).audit_package(package)
        except Exception as exc:
            st.error(f"Audit failed: {exc}")
            return

    if result.get("status") == "EXCLUDED":
        st.warning(f"Excluded: {result.get('exclusion_reason')}")
        return

    verdict = result["layer3"]["verdict"]
    color = VERDICT_COLORS.get(verdict, "#333")
    st.markdown(f"<h3 style='color:{color}'>Verdict: {verdict}</h3>", unsafe_allow_html=True)
    l1 = result["layer1"]
    if l1.get("degraded"):
        st.warning(f"Layer 1 ran in degraded mode: {l1.get('extractor_engine')}. Results are not those of the fine-tuned model.")
    st.markdown(result.get("explanation") or "")

    with st.expander("Layer 0: acquisition"):
        l0 = result["layer0"]
        st.write({k: l0.get(k) for k in ("github_repo", "security_policy_url", "repo_verification", "repo_metadata", "policy_word_count")})
        st.code(l0.get("raw_policy_text") or "", language="markdown")
    with st.expander("Layer 1: commitment tuple c = <t, tau, v, T>"):
        st.json(l1)
    with st.expander("Layer 2: evidence"):
        st.json({k: v for k, v in result["layer2"].items() if k != "vulnerabilities"})
        st.dataframe([{
            "id": v.get("id"), "published_at": v.get("published_at"),
            "fixed_versions": ", ".join(v.get("fixed_versions", [])), "sources": v.get("database_source"),
        } for v in result["layer2"]["vulnerabilities"]])
    with st.expander("Layer 3: rules and evidence trail"):
        st.write(result["layer3"].get("rules_triggered"))
        for line in result["layer3"].get("evidence_trail", []):
            st.text(line)
    st.download_button("Download JSON record", json.dumps(result, indent=2, ensure_ascii=False, default=str),
                       file_name=f"supportlens_{package}.json", mime="application/json")


def page_results() -> None:
    st.header("Evaluation results")
    files = {
        "Extraction (RQ1)": EVAL_DIR / "extraction_metrics.json",
        "Verdicts (RQ2)": EVAL_DIR / "verdict_metrics.json",
        "Statistical tests": EVAL_DIR / "statistical_tests.json",
        "Expert rubric": EVAL_DIR / "expert_evaluation.json",
        "Run manifest": EVAL_DIR / "run_manifest.json",
    }
    if not any(p.exists() for p in files.values()):
        st.info("No evaluation has been run yet. Results appear here after `python scripts/run_evaluation.py` "
                "has been run on the adjudicated ground truth with the trained models.")
        return
    for title, path in files.items():
        data = read_json(path)
        st.subheader(title)
        if data is None:
            st.caption(f"{path.name} not produced yet.")
        else:
            st.json(data)


def page_dataset() -> None:
    st.header("Dataset status")
    rows = {
        "Collected PyPI policies (data/processed/pypi_dataset.jsonl)": count_lines(ROOT / "data/processed/pypi_dataset.jsonl"),
        "Exclusions logged (data/audit/exclusion_log.jsonl)": count_lines(ROOT / "data/audit/exclusion_log.jsonl"),
        "Annotation sample (data/annotations/sample_300.jsonl)": count_lines(ROOT / "data/annotations/sample_300.jsonl"),
        "Adjudicated ground truth (data/ground_truth/ground_truth.jsonl)": count_lines(ROOT / "data/ground_truth/ground_truth.jsonl"),
        "Train / validation / test": " / ".join(str(count_lines(ROOT / f"data/ground_truth/{s}.jsonl")) for s in ("train", "val", "test")),
    }
    st.table([{"item": k, "count": v} for k, v in rows.items()])
    agreement = read_json(ROOT / "data/annotations/agreement_report.json")
    if agreement:
        st.subheader("Inter-annotator agreement")
        st.json(agreement)


def main() -> None:
    st.set_page_config(page_title="SupportLens", layout="wide")
    load_env()
    st.sidebar.title("SupportLens")
    page = st.sidebar.radio("Page", ["Audit a PyPI package", "Evaluation results", "Dataset status"])
    {"Audit a PyPI package": page_audit, "Evaluation results": page_results, "Dataset status": page_dataset}[page]()


if __name__ == "__main__":
    main()
