"""XTransfer (HHAR release) — single-run entry.

Reproduces one cell of the cross-modality FSL experiment:
source = miniImageNet ResNet18, target = HHAR, method = SRR (repair) + LWS.

The method config is `configs/hhar_single.yaml`; only per-run knobs are CLI:

    uv run python run.py --shot 5 --fold 1
"""
import argparse
import os
import sys

# make the repo root importable so `import xtransfer...` resolves when run from anywhere
_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from xtransfer.train import main, DEFAULT_CONFIG  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="XTransfer single run on HHAR (Our-Single).")
    ap.add_argument("--shot", type=int, default=5, help="few-shot support size (3/5/10)")
    ap.add_argument("--fold", type=int, default=1, help="LOOCV fold / episode id")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="method config yaml")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(("miniImageNet",), dataset="HHAR", epo_id=args.fold,
         n_shot=args.shot, config_file=args.config)
