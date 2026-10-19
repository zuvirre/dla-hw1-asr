import gzip
import urllib.request
from pathlib import Path

import gdown

# final checkpoint (ds2_final2_960h, epoch 93), Google Drive public link
CHECKPOINT_ID = "1Dcj4hXZz-0NiChn-eKe7CAt5Y1zWxuk3"
CHECKPOINT_PATH = Path("saved/ds2_final2_960h/model_best.pth")

# LibriSpeech 3-gram LM (openslr resource 11) converted to lowercase,
# Google Drive copy; fallback - build it from the openslr original
LM_ID = "1sSy3JerqW4p6IxETmY5_SzX3Wv15n0rr"
LM_URL = "https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"
LM_PATH = Path("data/lm/3-gram.pruned.1e-7.lower.arpa")


def build_lm_from_openslr():
    """
    Download the original uppercase LM from openslr and convert it
    to lowercase (our vocabulary is lowercase).
    """
    gz_path = LM_PATH.parent / "3-gram.pruned.1e-7.arpa.gz"
    print(f"Downloading {LM_URL}")
    urllib.request.urlretrieve(LM_URL, gz_path)
    with gzip.open(gz_path, "rt", encoding="utf-8") as fin, open(
        LM_PATH, "w", encoding="utf-8", newline="\n"
    ) as fout:
        for line in fin:
            fout.write(line.lower())
    gz_path.unlink()


def main():
    """
    Download the final checkpoint and the LM to the paths used in
    src/configs/inference.yaml. Run from the repository root.
    """
    if not CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=CHECKPOINT_ID, output=str(CHECKPOINT_PATH))

    if not LM_PATH.exists():
        LM_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            gdown.download(id=LM_ID, output=str(LM_PATH))
        except Exception as e:
            print(f"Google Drive download failed ({e}), using openslr")
            LM_PATH.unlink(missing_ok=True)
            build_lm_from_openslr()

    print(f"Checkpoint: {CHECKPOINT_PATH}\nLM: {LM_PATH}")


if __name__ == "__main__":
    main()
