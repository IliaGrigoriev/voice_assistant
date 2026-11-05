import json

import torch

from data.dataloader import Dataloader
from trainer import VOCABULARY


class ExperimentEvaluator:
    """Evaluate one model on both domains and persist comparable results."""

    def __init__(self, config, device):
        self.config = config
        self.device = device

    @property
    def loader_options(self):
        return {
            "batch_size": self.config.BATCH_SIZE,
            "num_workers": self.config.NUM_WORKERS,
        }

    def _test_loaders(self, include_librispeech=True):
        medical = Dataloader(
            VOCABULARY,
            Dataloader.MEDDIALOG_AUDIO,
            voice_manifest_path=self.config.VOICE_MANIFEST,
            require_voice_manifest=self.config.REQUIRE_VOICE_MANIFEST,
        )
        loaders = {
            "medical": medical.get_test_batches(**self.loader_options),
        }
        if include_librispeech:
            librispeech = Dataloader(VOCABULARY, Dataloader.LIBRISPEECH)
            loaders["librispeech"] = librispeech.get_test_batches(
                **self.loader_options
            )
        return loaders

    def evaluate_domains(self, trainer, include_librispeech=True):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        metrics = {}
        for domain, loader in self._test_loaders(include_librispeech).items():
            domain_metrics = trainer.evaluate(loader)
            metrics.update({
                f"{domain}_{name}": value
                for name, value in domain_metrics.items()
            })
        metrics.update(trainer.parameter_counts)
        metrics["model_size_bytes"] = trainer.model_size_bytes
        metrics["evaluation_gpu_peak_memory_bytes"] = (
            torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda" else 0
        )
        return metrics

    def write_results(self, experiment_name, purpose, metrics):
        result = {
            "experiment": experiment_name,
            "purpose": purpose,
            "device": str(self.device),
            **metrics,
        }
        folder = self.config.OUTPUT_DIR / experiment_name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return result
