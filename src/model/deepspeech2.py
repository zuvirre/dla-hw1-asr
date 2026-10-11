import torch
from torch import Tensor, nn


class DeepSpeech2(nn.Module):
    def __init__(
        self,
        n_feats: int,
        n_tokens: int,
        conv_channels: int = 32,
        rnn_hidden: int = 512,
        rnn_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_feats (int): number of input frequency features (n_mels).
            n_tokens (int): number of tokens in the vocabulary.
            conv_channels (int): number of channels in conv layers.
            rnn_hidden (int): hidden size of each GRU direction.
            rnn_layers (int): number of stacked GRU layers.
            dropout (float): dropout between GRU layers.
        """
        super().__init__()
        pass
