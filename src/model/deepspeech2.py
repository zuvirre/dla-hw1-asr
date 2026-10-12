from torch import Tensor, nn
from torch.nn import Sequential


def conv_out_length(length, kernel: int, stride: int, padding: int):
    return (length + 2 * padding - kernel) // stride + 1


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        n_feats: int,
        n_tokens: int,
        conv_params: list[dict],
        conv_channels: int = 32,
        rnn_hidden: int = 512,
        rnn_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        blocks = []
        in_channels = 1
        n_freq = n_feats
        for params in conv_params:
            kernel = tuple(params["kernel"])
            stride = tuple(params["stride"])
            padding = tuple(params["padding"])

            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        conv_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm2d(conv_channels),
                    nn.Hardtanh(0, 20),
                )
            )

            n_freq = conv_out_length(
                n_freq, kernel[0], stride[0], padding[0]
            )  # частотная ось после слоя
            in_channels = conv_channels

        self.conv_blocks = nn.ModuleList(blocks)
        self.rnn_input_size = conv_channels * n_freq

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        x = spectrogram.unsqueeze(1)
        for block in self.conv_blocks:
            x = block(x)
        return x  # временная заглушка до GRU

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths  # we don't reduce time dimension here

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
