import torchaudio
from torch import Tensor, nn


class TorchAudiomentation(nn.Module):
    def __init__(self, aug):
        super().__init__()
        self._aug = aug

    def forward(self, data: Tensor):
        x = data.unsqueeze(0)  # (1, 1, L)
        return self._aug(x).squeeze(1)  # (1, L)


class RandomSpeed(nn.Module):
    """
    Speed perturbation: resample audio by a random factor from 'factors'.
    Changes tempo and pitch together (as in Ko et al., 2015).
    """

    def __init__(self, sample_rate, factors):
        """
        Args:
            sample_rate (int): audio sample rate.
            factors (list[float]): speed factors to choose from uniformly.
                Include 1.0 to keep part of the audio unchanged.
        """
        super().__init__()
        self.speed = torchaudio.transforms.SpeedPerturbation(sample_rate, factors)

    def forward(self, audio: Tensor) -> Tensor:
        audio, _ = self.speed(audio)  # (1, L / factor); lengths are not needed
        return audio
