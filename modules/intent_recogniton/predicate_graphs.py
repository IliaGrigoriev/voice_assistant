import amrlib
import penman


class PredicateGraphs:

    def __init__(self):
        self._parser = amrlib.load_stog_model()

    def sent_to_graph(self, sentence: str) -> penman.Graph:
        # Convert sentence to AMR (PENMAN notation)
        amr = self._parser.parse_sents([sentence])[0]

        # Convert AMR string into a graph object
        return penman.decode(amr)