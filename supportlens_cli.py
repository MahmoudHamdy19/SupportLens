"""
SupportLens command-line interface.

    python supportlens_cli.py --pypi requests            # full five-layer audit of a PyPI package
    python supportlens_cli.py --file path/SECURITY.md     # Layer 1 extraction of a local policy draft
    python supportlens_cli.py --pypi requests --strict    # fail instead of using the regex fallback
"""

import argparse
import json
import os
import sys

from src.supportlens_pipeline import SupportLensPipeline
from src.layer0_data_acquisition.policy_preprocessing import PolicyPreprocessor
from src.layer1_ai_extraction import QwenCommitmentExtractor


def _load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.split("#", 1)[0].strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="SupportLens: assess version-support commitments against evidence")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pypi", metavar="PACKAGE", help="PyPI package name to audit")
    group.add_argument("--file", metavar="SECURITY_MD", help="local SECURITY.md to extract the commitment from")
    parser.add_argument("--strict", action="store_true", help="require the fine-tuned Qwen model (no fallback)")
    parser.add_argument("--json", action="store_true", help="print the full JSON record")
    args = parser.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _load_env_file()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = PolicyPreprocessor.clean_text(f.read())
        commitment = QwenCommitmentExtractor(strict=args.strict).extract(text).to_dict()
        print(json.dumps(commitment, indent=2, ensure_ascii=False))
        return 0

    result = SupportLensPipeline(github_token=os.environ.get("GITHUB_TOKEN"), strict=args.strict).audit_package(args.pypi)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    if result.get("status") == "EXCLUDED":
        print(f"{args.pypi}: excluded. {result.get('exclusion_reason')}")
        return 1
    l1 = result.get("layer1", {})
    print(f"Package:    {args.pypi}")
    print(f"Verdict:    {result.get('layer3', {}).get('verdict')}")
    print(f"Extractor:  {l1.get('extractor_engine')}{' (degraded fallback)' if l1.get('degraded') else ''}")
    print(f"Commitment: {l1.get('commitment_text')}")
    print(f"Report id:  {result.get('report_id', '-')}")
    print()
    print(result.get("explanation") or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
