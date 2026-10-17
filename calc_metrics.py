import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder import CTCTextEncoder


def main(gt_dir, pred_dir):
    """
    Compute mean WER and CER over utterances.
    Files are matched by ID: gt_dir/<ID>.txt <-> pred_dir/<ID>.txt.
    Utterances without a prediction (e.g. unreadable audio) are skipped.
    """
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    wers, cers, missing = [], [], []

    for gt_path in sorted(gt_dir.glob("*.txt")):
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            missing.append(gt_path.stem)
            continue
        target = CTCTextEncoder.normalize_text(gt_path.read_text(encoding="utf-8"))
        pred = CTCTextEncoder.normalize_text(pred_path.read_text(encoding="utf-8"))
        wers.append(calc_wer(target.strip(), pred.strip()))
        cers.append(calc_cer(target.strip(), pred.strip()))

    if missing:
        print(f"No prediction for {len(missing)} utterances: {missing[:10]}")
    assert wers, f"No matching files in {gt_dir} and {pred_dir}"

    print(f"Utterances: {len(wers)}")
    print(f"WER: {sum(wers) / len(wers):.4f}")
    print(f"CER: {sum(cers) / len(cers):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WER/CER between two dirs")
    parser.add_argument(
        "--gt_dir", required=True, help="dir with <ID>.txt ground truth"
    )
    parser.add_argument(
        "--pred_dir", required=True, help="dir with <ID>.txt predictions"
    )
    args = parser.parse_args()
    main(args.gt_dir, args.pred_dir)
