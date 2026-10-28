import amrlib
import penman
from pathlib import Path


class PredicateGraphs:

    MODEL_URL = (
        "https://github.com/bjascob/amrlib-models/releases/download/"
        "parse_xfm_bart_base-v0_1_0/"
        "model_parse_xfm_bart_base-v0_1_0.tar.gz"
    )

    def __init__(self):
        self.download_model()
        self._parser = amrlib.load_stog_model()

    @classmethod
    def download_model(cls):
        model_path = Path(amrlib.__file__).resolve().parent / "data" / "model_stog"
        if not model_path.exists():
            amrlib.download("model_stog", cls.MODEL_URL)

    def sent_to_graph(self, sentence: str) -> penman.Graph:
        # Convert sentence to AMR (PENMAN notation)
        amr = self._parser.parse_sents([sentence])[0]

        # Convert AMR string into a graph object
        return penman.decode(amr)
