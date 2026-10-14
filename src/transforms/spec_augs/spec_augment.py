from torch import Tensor, nn
from torchaudio.functional import mask_along_axis


class SpecAugment(nn.Module):
    """
    Masked regions are filled with the spectrogram mean: our log-mel
    is in dB and not normalized, so a constant 0 would look like
    a loud flat band instead of missing information.
    """

    def __init__(
        self,
        freq_mask_param,
        time_mask_param,
        n_freq_masks=2,
        n_time_masks=2,
        max_time_ratio=1.0,
    ):
        """
        Args:
            freq_mask_param (int): max width of one frequency mask
                (in mel bins).
            time_mask_param (int): max width of one time mask (in frames).
            n_freq_masks (int): number of frequency masks.
            n_time_masks (int): number of time masks.
            max_time_ratio (float): max width of one time mask as a
                fraction of the utterance length (protects short records).
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.max_time_ratio = max_time_ratio

    def forward(self, spectrogram: Tensor) -> Tensor:
        # spectrogram: (1, n_mels, T)
        mask_value = spectrogram.mean().item()
        for _ in range(self.n_freq_masks):
            spectrogram = mask_along_axis(
                spectrogram, self.freq_mask_param, mask_value, axis=1
            )  # (1, n_mels, T)
        for _ in range(self.n_time_masks):
            spectrogram = mask_along_axis(
                spectrogram,
                self.time_mask_param,
                mask_value,
                axis=2,
                p=self.max_time_ratio,
            )  # (1, n_mels, T)
        return spectrogram
