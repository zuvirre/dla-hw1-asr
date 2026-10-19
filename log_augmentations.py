import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets import LibrispeechDataset
from src.logger.utils import plot_spectrogram
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)

# dev-clean utterances to show (index is sorted by length)
UTTERANCES = [500, 1500, 2500]


@hydra.main(version_base=None, config_path="src/configs", config_name="ds2_local")
def main(config):
    """
    Log original and augmented audio and spectrograms to the experiment
    tracker: each train augmentation separately (always applied) and the
    whole train chain as used in training. Shows that the augmentations
    work and are not too severe.

    Run: python log_augmentations.py transforms=ds2_augs writer.run_name=ds2_augs_check

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    train_transforms = config.transforms.instance_transforms.train
    sample_rate = train_transforms.get_spectrogram.transforms[0].sample_rate
    get_spectrogram = instantiate(train_transforms.get_spectrogram)
    wave_chain = instantiate(train_transforms.audio)
    spec_augment = instantiate(train_transforms.spectrogram)

    # each wave augmentation from the config separately, with p = 1
    wave_augs = {}
    for aug_config in OmegaConf.to_container(train_transforms.audio.transforms):
        if "aug" in aug_config:  # torch_audiomentations wrapper
            aug_config["aug"]["p"] = 1.0
            name = aug_config["aug"]["_target_"].split(".")[-1]
        else:
            name = aug_config["_target_"].split(".")[-1]
        if name == "Gain":
            aug_config["p"] = 1.0
        if "factors" in aug_config:  # factor 1.0 would leave audio unchanged
            aug_config["factors"] = [0.9, 1.1]
        wave_augs[name] = instantiate(aug_config)

    dataset = LibrispeechDataset(part="dev-clean")

    for step, ind in enumerate(UTTERANCES):
        writer.set_step(step, mode="augs")
        audio = dataset.load_audio(dataset._index[ind]["path"])  # (1, L)
        spectrogram = get_spectrogram(audio)  # (1, n_mels, T)

        examples = {"original": (audio, spectrogram)}
        for name, aug in wave_augs.items():
            aug_audio = aug(audio)
            examples[name] = (aug_audio, get_spectrogram(aug_audio))
        examples["SpecAugment"] = (None, spec_augment(spectrogram.clone()))
        chain_audio = wave_chain(audio)
        examples["full_train_chain"] = (
            chain_audio,
            spec_augment(get_spectrogram(chain_audio)),
        )

        for name, (aug_audio, aug_spectrogram) in examples.items():
            if aug_audio is not None:
                writer.add_audio(f"audio_{name}", aug_audio, sample_rate=sample_rate)
            writer.add_image(
                f"spectrogram_{name}", plot_spectrogram(aug_spectrogram[0], name)
            )
        logger.info(f"Logged utterance {ind}: {', '.join(examples)}")


if __name__ == "__main__":
    main()
