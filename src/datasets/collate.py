import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    texts = [item["text"] for item in dataset_items]  # for metrics
    audio_paths = [item["audio_path"] for item in dataset_items]  # for metrics

    spectrograms = [item["spectrogram"].squeeze(0) for item in dataset_items]
    spectrogram_length = torch.tensor([spec.shape[-1] for spec in spectrograms])

    spectrograms_t = [spec.T for spec in spectrograms]
    spectrogram = pad_sequence(spectrograms_t, batch_first=True)
    spectrogram = spectrogram.transpose(1, 2)

    texts_encoded = [item["text_encoded"].squeeze(0) for item in dataset_items]
    text_encoded_length = torch.tensor([txt.shape[0] for txt in texts_encoded])
    text_encoded = pad_sequence(texts_encoded, batch_first=True).long()

    return {
        "spectrogram": spectrogram,
        "spectrogram_length": spectrogram_length,
        "text_encoded": text_encoded,
        "text_encoded_length": text_encoded_length,
        "text": texts,
        "audio_path": audio_paths,
    }
