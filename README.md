# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#results">Results</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#training">Training</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains a solution of the ASR homework of the [HSE DLA course](https://github.com/markovka17/dla). The assignment is [here](https://github.com/markovka17/dla/tree/2026/hw1ASR).

The model is [DeepSpeech2](https://arxiv.org/abs/1512.02595) (14.45M parameters): 2 convolutional layers over a 128-bin log-mel spectrogram, 3 bidirectional GRU layers (512 hidden units), CTC loss over 28 characters. It is trained on the full LibriSpeech (960 h) with augmentations (speed perturbation, low-pass filter, gain, colored noise, SpecAugment).

Decoding: argmax, our own CTC beam search, and beam search with a LibriSpeech 3-gram KenLM language model ([pyctcdecode](https://github.com/kensho-technologies/pyctcdecode)).

- Demo notebook: [`demo.ipynb`](demo.ipynb) (open in Google Colab).
- Report and training logs: [Comet ML report](https://www.comet.com/oleg-musatov/dla-hw1-asr/reports/sbuTRzLnMw4GtWfEpVQ4UvOtf).

## Results

Final model (`saved/ds2_final2_960h/model_best.pth`), LibriSpeech test, beam size 10, WER/CER in %:

| decoder | test-clean WER | test-clean CER | test-other WER | test-other CER |
|---|---|---|---|---|
| argmax | 17.05 | 5.13 | 36.51 | 13.87 |
| beam search | 16.45 | 4.96 | 35.90 | 13.56 |
| **beam search + 3-gram LM** (α 0.5, β 1.0) | **10.83** | **4.00** | **26.77** | **12.40** |

## Installation

**Python 3.12 is required** (tested on 3.12.3). Python 3.13+ is not supported: there are no `torch==2.2.0` / `numpy==1.26.4` wheels for it. All commands below are run from the repository root.

KenLM (needed for beam search with the language model) is built from source during the installation. On Windows this needs a C++ compiler, see [Language model](#language-model).

Clone the repository:

```bash
git clone https://github.com/zuvirre/dla-hw1-asr.git
cd dla-hw1-asr
```

### Option A: `uv` (recommended, any OS)

[`uv`](https://docs.astral.sh/uv/) does not need Python or `pip`: it downloads Python 3.12 itself, so this works with any system Python or without it.

1. Install `uv` ([official instructions](https://docs.astral.sh/uv/getting-started/installation/)), then open a new terminal:

   ```bash
   # Windows
   winget install --id=astral-sh.uv -e
   # Linux / macOS
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create and activate the environment:

   ```bash
   uv venv --python 3.12 .venv

   # Linux / macOS
   source .venv/bin/activate
   # Windows (PowerShell); the first line allows activation scripts in this terminal only
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .venv\Scripts\activate
   ```

3. (Windows with an NVIDIA GPU only) Install the CUDA build of torch (the default Windows wheel is CPU-only; on Linux it already includes CUDA 12.1):

   ```bash
   uv pip install torch==2.2.0 torchaudio==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
   ```

4. Install all required packages:

   ```bash
   uv pip install -r requirements.txt
   ```

### Option B: `venv` + `pip` (Python 3.12 is already installed)

1. Create and activate the environment:

   ```bash
   # Linux / macOS
   python3.12 -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell); the second line allows activation scripts in this terminal only
   py -3.12 -m venv .venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .venv\Scripts\activate
   ```

2. (Windows with an NVIDIA GPU only) Install the CUDA build of torch (the default Windows wheel is CPU-only; on Linux it already includes CUDA 12.1):

   ```bash
   python -m pip install torch==2.2.0 torchaudio==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
   ```

3. Install all required packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

### Download the checkpoint and the language model

```bash
python download_weights.py
```

The checkpoint goes to `saved/ds2_final2_960h/model_best.pth`, the LM to `data/lm/3-gram.pruned.1e-7.lower.arpa` (the paths used in `src/configs/inference.yaml`). Both are downloaded from Google Drive; if Google Drive is unavailable, the LM is built from the [openslr](https://www.openslr.org/11/) original.

In Google Colab the whole installation takes about 2 minutes, see [`demo.ipynb`](demo.ipynb).

### (For development) pre-commit

```bash
pre-commit install
```

### Language model

Beam search with the LM uses [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode) and [KenLM](https://github.com/kpu/kenlm).

- **LM file.** LibriSpeech 3-gram pruned LM `3-gram.pruned.1e-7.arpa.gz` from [openslr resource 11](https://www.openslr.org/11/), converted to lowercase (our vocabulary is lowercase). `download_weights.py` does everything; the conversion is in `build_lm_from_openslr()`.
- **KenLM** has no wheels on PyPI, so `requirements.txt` installs it from source (`https://github.com/kpu/kenlm/archive/master.zip`). This needs a C++ compiler and CMake:
  - **Linux / Colab:** nothing to do, the compiler is already there (the build takes about a minute).
  - **Windows:** install [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the **"Desktop development with C++"** workload (MSVC v143 + Windows SDK, ~7 GB), e.g.:

    ```bash
    winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    ```

    If Build Tools are already installed without this workload, add it via Visual Studio Installer → "Modify". Without a compiler the KenLM build fails with `No CMAKE_C_COMPILER could be found`, and then **`pip install -r requirements.txt` fails entirely** (nothing is installed).
- **Without the LM.** If KenLM can not be installed, remove the KenLM line from `requirements.txt`, install the rest and run inference with our beam search only:

  ```bash
  python inference.py inferencer.decoder=beam metrics=ds2
  ```

## How To Use

### Inference

Evaluate the final model on LibriSpeech test-clean and test-other (the parts are downloaded automatically on the first run, ~17.5 min on RTX 4070).
LibriSpeech is downloaded from the openslr mirror `openslr.magicdatatech.com`; if it is slow, change the base URL in `URL_LINKS` in `src/datasets/librispeech_dataset.py` to `https://www.openslr.org/resources/12/`.

```bash
python inference.py
```

It prints WER/CER for three decoders (argmax, beam search, beam search + LM) and saves predictions to `data/saved/ds2_final2_960h/<part>/<UtteranceID>.txt`.

Run inference on a custom dataset:

```bash
python inference.py datasets=custom_dir datasets.custom.data_dir=PATH_TO_DIR inferencer.save_path=SAVE_PATH
```

where `PATH_TO_DIR` has the following structure:

```
NameOfTheDirectoryWithUtterances
├── audio
│   ├── UtteranceID1.wav # may be flac or mp3
│   ├── UtteranceID2.wav
│   .
│   .
│   .
│   └── UtteranceIDn.wav
└── transcriptions # ground truth, may not exist
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    .
    .
    .
    └── UtteranceIDn.txt
```

- Predictions are saved as `SAVE_PATH/UtteranceID.txt`. `SAVE_PATH` is either a directory name inside `data/saved/` or an absolute path.
- Audio of any sample rate is resampled to 16 kHz; unreadable files are skipped with a warning.
- If `transcriptions/` exists, WER/CER are printed as well (texts are lowercased, punctuation is removed).
- Decoder used for the saved predictions: `inferencer.decoder=lm` (default), `beam` or `argmax`.

### Metrics

Calculate WER and CER given the ground truth and predicted transcriptions (files are matched by `UtteranceID.txt`):

```bash
python calc_metrics.py --gt_dir PATH_TO_DIR/transcriptions --pred_dir SAVE_PATH
```

## Training

To train a model, run the following command:

```bash
python train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

Logging is done with [Comet ML](https://www.comet.com/). The configs `ds2_local`, `ds2_final`, `ds2_final2` log in offline mode (archives in `.cometml-runs/`, upload with `comet upload <archive>.zip`). All experiments in the report were logged this way: offline during training, the archives were uploaded to Comet ML afterwards. For online logging set `writer.mode=online` and provide an API key (`comet login` or the `COMET_API_KEY` environment variable).

LibriSpeech is downloaded automatically to `data/datasets/librispeech/` on the first run.

### Reproducing the final model

The final model was trained in 3 stages on one RTX 4070 (12 GB), ~23 h in total. Each stage starts from the best checkpoint of the previous one, so `writer.run_name` must be exactly as below (it defines the checkpoint directory `saved/<run_name>/`).

1. LibriSpeech train-clean-100, augmentations, 40k steps (~3 h):

   ```bash
   python train.py -cn=ds2_local transforms=ds2_augs writer.run_name=ds2_augs_100h
   ```

2. Fine-tuning on the full LibriSpeech 960 h, 110k steps (~10 h):

   ```bash
   python train.py -cn=ds2_final writer.run_name=ds2_final_960h
   ```

3. Second fine-tuning cycle on 960 h with a lower peak LR, 100k steps (~10 h); the final checkpoint is `saved/ds2_final2_960h/model_best.pth`:

   ```bash
   python train.py -cn=ds2_final2 writer.run_name=ds2_final2_960h
   ```
Examples of all train augmentations (audio and spectrograms, each augmentation separately and the whole chain) are logged with:

   ```bash
   python log_augmentations.py transforms=ds2_augs writer.run_name=ds2_augs_check
   ```
Details (configs, logs, what was tried) are in the [report](https://www.comet.com/oleg-musatov/dla-hw1-asr/reports/sbuTRzLnMw4GtWfEpVQ4UvOtf).

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

The demo example is from [LibriSpeech](https://www.openslr.org/12/) (CC BY 4.0), the language model is from [openslr resource 11](https://www.openslr.org/11/) (public domain).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
