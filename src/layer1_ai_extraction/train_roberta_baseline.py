"""Command line for fine-tuning the RoBERTa-base baseline.

Run from the project root:  python -m src.layer1_ai_extraction.train_roberta_baseline

Hyper-parameters follow thesis 3.3.2 (AdamW, learning rate 2e-5, batch size 16, 5 epochs, seed 42);
the cross-entropy is class-weighted by default (--class-weighting none for the uniform loss); see
`roberta_baseline` for the method. The command refuses to run without
data/ground_truth/train.jsonl and val.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .errors import ModelUnavailableError
from .roberta_baseline import CLASS_WEIGHTINGS, DEFAULT_MODEL_DIR, RobertaTrainingConfig, train_roberta
from .seeding import DEFAULT_SEED
from .training_data import TRAIN_PATH, VAL_PATH


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune the RoBERTa-base token-classification baseline.")
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--val", type=Path, default=VAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=RobertaTrainingConfig.num_epochs)
    parser.add_argument("--batch-size", type=int, default=RobertaTrainingConfig.batch_size)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--class-weighting", choices=CLASS_WEIGHTINGS, default=RobertaTrainingConfig.class_weighting)
    args = parser.parse_args(argv)
    config = RobertaTrainingConfig(
        num_epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, class_weighting=args.class_weighting
    )
    try:
        train_roberta(config, args.train, args.val, args.output_dir)
    except (FileNotFoundError, ModelUnavailableError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
