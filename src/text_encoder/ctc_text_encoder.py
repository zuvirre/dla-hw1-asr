import re
from collections import defaultdict
from string import ascii_lowercase

import numpy as np
import torch

# TODO add BPE, LM, support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded_text = []
        previous_ind = self.char2ind[self.EMPTY_TOK]
        for ind in inds:
            ind = int(ind)
            if ind != previous_ind and ind != self.char2ind[self.EMPTY_TOK]:
                decoded_text.append(self.ind2char[ind])
            previous_ind = ind
        return "".join(decoded_text)

    def ctc_beam_search(self, log_probs, length, beam_size=10) -> str:
        """
        CTC prefix beam search.

        Args:
            log_probs (Tensor | np.ndarray): (T, V) log-probs of one utterance.
            length (int): number of valid frames.
            beam_size (int): number of prefixes kept after each frame.
        Returns:
            text (str): the most probable text.
        """
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu().numpy()

        log_probs = log_probs[:length]
        probs = np.exp(log_probs)
        beam = {"": [1.0, 0.0]}  # prefix -> [p_blank, p_nonblank]

        for prob_t in probs:
            new_beam = defaultdict(lambda: [0.0, 0.0])
            for prefix, (p_b, p_nb) in beam.items():
                for ind, p in enumerate(prob_t):
                    if ind == self.char2ind[self.EMPTY_TOK]:
                        new_beam[prefix][0] += (p_b + p_nb) * p
                        continue
                    char = self.ind2char[ind]
                    if len(prefix) > 0 and prefix[-1] == char:
                        new_beam[prefix][1] += p_nb * p
                        new_beam[prefix + char][1] += p_b * p
                    else:
                        new_beam[prefix + char][1] += (p_b + p_nb) * p

            beam = dict(sorted(new_beam.items(), key=lambda x: -sum(x[1]))[:beam_size])

            total = sum(p_b + p_nb for p_b, p_nb in beam.values())
            beam = {
                prefix: [p_b / total, p_nb / total]
                for prefix, (p_b, p_nb) in beam.items()
            }
        return next(iter(beam))

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
