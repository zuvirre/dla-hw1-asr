from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def conv_out_length(length, kernel: int, stride: int, padding: int):
    return (length + 2 * padding - kernel) // stride + 1


class ConvBlock(nn.Module):
    """
    One DeepSpeech2 conv layer: Conv2d -> BatchNorm2d -> HardTanh.
    Stores its kernel/stride/padding to recompute output sizes.
    """

    def __init__(self, in_channels, out_channels, kernel, stride, padding):
        super().__init__()
        self.kernel = tuple(kernel)
        self.stride = tuple(stride)
        self.padding = tuple(padding)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.Hardtanh(0, 20)

    def forward(self, x):
        # (B, C_in, F, T) -> (B, C_out, F', T')
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

    def output_freq(self, n_freq):
        """Frequency size after this block."""
        return conv_out_length(n_freq, self.kernel[0], self.stride[0], self.padding[0])

    def output_length(self, lengths):
        """Time lengths after this block."""
        return conv_out_length(lengths, self.kernel[1], self.stride[1], self.padding[1])


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
        """
        Args:
            n_feats (int): number of input frequency features.
            n_tokens (int): number of tokens in the vocabulary.
            conv_params (list[dict]): kernel/stride/padding for each conv layer,
                pairs (freq, time).
            conv_channels (int): number of channels in conv layers.
            rnn_hidden (int): hidden size of each GRU direction.
            rnn_layers (int): number of stacked GRU layers.
            dropout (float): dropout between GRU layers.
        """

        super().__init__()

        self.conv_blocks = nn.ModuleList()
        in_channels = 1
        n_freq = n_feats

        for params in conv_params:
            block = ConvBlock(in_channels, conv_channels, **params)
            self.conv_blocks.append(block)
            n_freq = block.output_freq(n_freq)
            in_channels = conv_channels

        self.rnn_input_size = conv_channels * n_freq

        self.rnn = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.fc = nn.Linear(2 * rnn_hidden, n_tokens)

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
        x = spectrogram.unsqueeze(1)  # (B, 1, F, T)
        for block in self.conv_blocks:
            x = block(x)  # (B, C, F', T')
        lengths = self.transform_input_lengths(spectrogram_length)

        x = x.permute(0, 3, 2, 1)  # (B, T', F', C)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, T', F' * C)
        t_out = x.shape[1]
        x = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(
            x, batch_first=True, total_length=t_out
        )  # (B, T', 2 * rnn_hidden)

        logits = self.fc(x)  # (B, T', n_tokens)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return {"log_probs": log_probs, "log_probs_length": lengths}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths (B,).
        Returns:
            output_lengths (Tensor): new temporal lengths (B,).
        """
        lengths = input_lengths
        for block in self.conv_blocks:
            lengths = block.output_length(lengths)
        return lengths

    @staticmethod
    def _count_params(module):
        """Return (all, trainable) parameter counts of a module."""
        all_parameters = sum(p.numel() for p in module.parameters())
        trainable_parameters = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
        return all_parameters, trainable_parameters

    def __str__(self):
        """
        Model prints with the number of parameters, total and per block.
        """
        result_info = super().__str__()

        parts = {
            "conv": self.conv_blocks,
            "rnn": self.rnn,
            "fc": self.fc,
        }
        for name, module in parts.items():
            all_parameters, trainable_parameters = self._count_params(module)
            result_info += f"\n{name} parameters: {all_parameters} (trainable: {trainable_parameters})"

        all_parameters, trainable_parameters = self._count_params(self)
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
