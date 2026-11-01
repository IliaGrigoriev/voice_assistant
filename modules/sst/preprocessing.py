import torch

class Preprocessing:

    def __init__(self):
        pass

    # Encode transcripts
    # ---------------------------------------------------------------
    def encode_transcripts(self, VOCABULARY, transcripts):
        CHAR_TO_ID = {
            character: index
            for index, character in enumerate(VOCABULARY)
        }
        encoded = [
            torch.tensor(
                [CHAR_TO_ID[character] for character in transcript],
                dtype=torch.long,
            )
            for transcript in transcripts
        ]

        encoded_transcripts = torch.cat(encoded)

        transcript_lengths = torch.tensor(
            [len(item) for item in encoded],
            dtype=torch.long,
        )

        return [encoded_transcripts, transcript_lengths]