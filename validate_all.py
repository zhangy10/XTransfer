"""XTransfer (HHAR) — Leave-One-Out cross-validation sweep.

Runs source=miniImageNet ResNet18 -> target=HHAR across shots x folds and
reports the mean accuracy per shot (cf. paper Table 4, "Our-Single").

    uv run python validate_all.py                 # shots 3,5,10 x folds 1,3,6,10,12
    uv run python validate_all.py --shots 5 --folds 1 3 6 10 12
"""
import argparse
import os
import pickle
import sys

_REPO = os.path.dirname(os.path.abspath(__file__))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from xtransfer.train import main  # noqa: E402

SOURCE = ("miniImageNet",)


def final_metrics(log_path):
    """Extract per-run final accuracies from a run's log_dict.pkl."""
    with open(log_path, "rb") as f:           # trusted: produced by this repo
        d = pickle.load(f)

    def last(key):
        v = d.get(key)
        return float(v[-1]) if isinstance(v, (list, tuple)) and v else None

    def best(key):
        v = d.get(key)
        return float(max(v)) if isinstance(v, (list, tuple)) and v else None

    return {
        "lws_test_acc": last("test_acc"),            # recombined model (KNN)
        "prune_test_acc": last("prune_test_acc"),    # after channel removal
        "finetune_last": last("finetune_knn_test_acc"),
        "finetune_best": best("finetune_knn_test_acc"),
    }


def parse_args():
    ap = argparse.ArgumentParser(description="XTransfer LOOCV sweep.")
    ap.add_argument("--dataset", default="HHAR", choices=["HHAR", "WESAD"], help="target sensing dataset")
    ap.add_argument("--shots", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 3, 6, 10, 12])
    return ap.parse_args()


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


if __name__ == "__main__":
    args = parse_args()
    rows = []
    for shot in args.shots:
        per_shot = []
        for fold in args.folds:
            out_dir = main(SOURCE, dataset=args.dataset, epo_id=fold, n_shot=shot)
            m = final_metrics(os.path.join(out_dir, "log_dict.pkl"))
            per_shot.append(m)
            print(f"[shot={shot} fold={fold}] {m}")
        rows.append((shot, {k: mean([p[k] for p in per_shot]) for k in per_shot[0]}))

    print("\n==== mean accuracy over folds ====")
    for shot, agg in rows:
        summary = "  ".join(f"{k}={v:.4f}" for k, v in agg.items())
        print(f"shot={shot}:  {summary}")
