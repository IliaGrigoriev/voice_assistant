"""Shared text normalization, CTC decoding, and error-rate metrics."""

import re


class TextProcessor:
    """One source of truth for target text and greedy CTC predictions."""

    def __init__(self, vocabulary, blank_id=0):
        self.vocabulary = vocabulary
        self.blank_id = blank_id
        self.allowed = set(vocabulary) - {"<blank>"}

    def normalize(self, text):
        text = re.sub(r"\s+", " ", text.lower()).strip()
        return "".join(character for character in text if character in self.allowed)

    def decode(self, token_ids):
        decoded = []
        previous = None
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id != self.blank_id and token_id != previous:
                decoded.append(self.vocabulary[token_id])
            previous = token_id
        return "".join(decoded).strip()

    @staticmethod
    def edit_distance(reference, hypothesis):
        previous = list(range(len(hypothesis) + 1))
        for row, reference_item in enumerate(reference, start=1):
            current = [row]
            for column, hypothesis_item in enumerate(hypothesis, start=1):
                current.append(min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                ))
            previous = current
        return previous[-1]

    def error_counts(self, references, hypotheses):
        counts = {"word_errors": 0, "words": 0, "char_errors": 0, "chars": 0}
        for reference, hypothesis in zip(references, hypotheses):
            counts["word_errors"] += self.edit_distance(
                reference.split(), hypothesis.split()
            )
            counts["words"] += len(reference.split())
            counts["char_errors"] += self.edit_distance(reference, hypothesis)
            counts["chars"] += len(reference)
        return counts

    @staticmethod
    def rates(counts):
        return {
            "wer": counts["word_errors"] / max(1, counts["words"]),
            "cer": counts["char_errors"] / max(1, counts["chars"]),
        }
