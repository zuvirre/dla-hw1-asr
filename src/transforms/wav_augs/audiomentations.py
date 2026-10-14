import torchaudio
from torch import Tensor, nn


class Torch(nn.Module):
    def __init__(self, aug):
        super().__init__()
        self._aug = aug

    def __call__(self, data: Tensor):
        x = data.unsqueeze(0)  # (1, 1, L)
        return self._aug(x).squeeze(1)  # (1, L)
