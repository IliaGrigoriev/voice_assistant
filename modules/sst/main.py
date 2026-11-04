from const import ExperimentConfig
from experiment_runner import ExperimentRunner


class Main:
    """STT experiment entry point."""

    def run(self):
        return ExperimentRunner(ExperimentConfig).run()


if __name__ == "__main__":
    Main().run()
