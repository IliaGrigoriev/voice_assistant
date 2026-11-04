import spacy


class SpacyActionGraph:
    """Analyze sentences and expose spaCy dependency relationships."""

    # Initialize the analyzer with a spaCy model or an existing pipeline
    # ---------------------------------------------------------------------
    def __init__(self, model: str = "en_core_web_sm", nlp=None):
        self.model = model
        self._nlp = nlp

    # Load and return the configured spaCy language pipeline
    # ---------------------------------------------------------------------
    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = spacy.load(self.model)
        return self._nlp

    # Analyze a sentence and return normalized token relationships
    # ---------------------------------------------------------------------
    def analyze(self, sentence: str) -> list[dict]:
        doc = self.nlp(sentence)
        return [
            {
                "id": token.i,
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "dependency": token.dep_,
                "head_id": token.head.i,
                "head": token.head.text,
                "ner_type": token.ent_type_ or None,
            }
            for token in doc
        ]

    # Print the key linguistic attributes for every token in a sentence
    # ---------------------------------------------------------------------
    def print_analysis(self, sentence: str) -> None:
        for token in self.analyze(sentence):
            print(
                token["text"],
                token["lemma"],
                token["pos"],
                token["dependency"],
                token["head"],
            )
