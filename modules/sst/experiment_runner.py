import json
import signal

import torch

from experiments import Experiments


class ExperimentRunner:
    """Select, stop, skip, and resume the experiment suite."""

    VALID_SELECTIONS = ("original", "frozen", "full", "conformer", "all")

    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if config.EXPERIMENT not in self.VALID_SELECTIONS:
            raise ValueError(
                f"Unknown experiment {config.EXPERIMENT!r}; choose from "
                f"{self.VALID_SELECTIONS}"
            )
        self.experiments = Experiments(config, self.device)
        signal.signal(signal.SIGINT, self._stop_handler)
        signal.signal(signal.SIGTERM, self._stop_handler)

    @staticmethod
    def _stop_handler(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    def _selected(self):
        registry = self.experiments.registry
        if self.config.EXPERIMENT == "all":
            return registry.values()
        return (registry[self.config.EXPERIMENT],)

    def _write_suite_state(self, status, results):
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (self.config.OUTPUT_DIR / "suite_state.json").write_text(
            json.dumps({
                "status": status,
                "completed": [result["experiment"] for result in results],
            }, indent=2),
            encoding="utf-8",
        )

    def run(self):
        selected = tuple(self._selected())
        if len(selected) > 1 and self.config.RESUME_FROM:
            raise ValueError("RESUME_FROM can only be used with one experiment")
        results = []
        try:
            for experiment_name, method in selected:
                result_path = (
                    self.config.OUTPUT_DIR / experiment_name / "results.json"
                )
                if (self.config.SKIP_COMPLETED_EXPERIMENTS
                        and result_path.is_file()):
                    print(f"Skipping completed experiment {experiment_name}")
                    self.experiments.publish_best_model(
                        experiment_name,
                        self.config.OUTPUT_DIR / experiment_name,
                    )
                    results.append(json.loads(
                        result_path.read_text(encoding="utf-8")
                    ))
                else:
                    results.append(method())
        except KeyboardInterrupt:
            self._write_suite_state("interrupted", results)
            print("Experiment suite stopped safely; run main.py to resume")
            return results

        self._write_suite_state("complete", results)
        (self.config.OUTPUT_DIR / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        return results
