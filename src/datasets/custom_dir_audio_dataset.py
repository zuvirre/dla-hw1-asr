import logging
from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        data_dir = Path(data_dir)
        audio_dir = data_dir / "audio"
        transcription_dir = data_dir / "transcriptions"
        has_transcriptions = transcription_dir.exists()

        index = []
        for path in sorted(audio_dir.iterdir()):
            if path.suffix.lower() not in (".wav", ".flac", ".mp3"):
                continue
            try:
                audio, sr = torchaudio.load(str(path))
            except Exception as e:
                logger.warning(f"Skipping unreadable audio {path.name}: {e}")
                continue
            text = ""
            if has_transcriptions:
                transcription_path = transcription_dir / f"{path.stem}.txt"
                if transcription_path.exists():
                    text = transcription_path.read_text(encoding="utf-8").strip()
                else:
                    logger.warning(f"No transcription for {path.name}")
            index.append(
                {"path": str(path), "text": text, "audio_len": audio.shape[-1] / sr}
            )

        logger.info(
            f"Custom dir: {len(index)} audio files, "
            f"transcriptions: {'yes' if has_transcriptions else 'no'}"
        )
        super().__init__(index, *args, **kwargs)
