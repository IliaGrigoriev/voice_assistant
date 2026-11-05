"""Read transcriptions from MedDialog metadata."""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Transcription:
    sentence_id: str
    text: str


class MetadataReader:
    def __init__(self, metadata_path: Path) -> None:
        self.metadata_path = metadata_path

    def read(self) -> list[Transcription]:
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with self.metadata_path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            required = {"filename", "transcription"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Metadata is missing columns {sorted(missing)}; "
                    f"found {reader.fieldnames}"
                )
            rows = [
                Transcription(row["filename"], row["transcription"].strip())
                for row in reader
                if row.get("transcription", "").strip()
            ]

        if not rows:
            raise ValueError(f"No transcriptions found in {self.metadata_path}")
        return rows
