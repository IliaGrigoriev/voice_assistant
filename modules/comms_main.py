"""Main application that connects the frontend to the assistant modules."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
import uvicorn

from intent_recogniton.main import parse
from intent_recogniton.predicate_graphs import PredicateGraphs
from sst.const import STT_MODEL
from sst.stt_model import STTModel


VOCABULARY = ["<blank>", " ", "'", *"abcdefghijklmnopqrstuvwxyz"]
HOP_LENGTH = 160
MAX_AUDIO_BYTES = 25 * 1024 * 1024


class Main:
    """Orchestrate assistant modules and expose them through FastAPI."""

    def __init__(self) -> None:
        PredicateGraphs.download_model()
        self._stt_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._stt_model = self._load_stt_model()
        self.app = FastAPI(title="Voice scheduling assistant")
        self._configure_cors()
        self._register_routes()

    def _load_stt_model(self) -> STTModel:
        """Build the STT architecture and load its trained weights once."""
        model_path = Path(STT_MODEL)
        if not model_path.is_file():
            raise FileNotFoundError(f"STT model not found at {model_path}")

        model = STTModel(
            vocab_size=len(VOCABULARY),
            hop_length=HOP_LENGTH,
        ).to(self._stt_device)
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=self._stt_device,
                weights_only=True,
            )
        )
        model.eval()
        return model

    def _configure_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _register_routes(self) -> None:
        self.app.add_api_route(
            "/recognize",
            self.recognize,
            methods=["POST"],
        )
        self.app.add_api_route(
            "/transcribe",
            self.transcribe,
            methods=["POST"],
        )

    def recognize(self, text: str) -> dict[str, Any]:
        """Parse an utterance and return its semantic action graph."""
        return self._parse_action_graph(text)

    async def transcribe(
        self,
        audio: UploadFile = File(...),
    ) -> dict[str, Any]:
        """Decode uploaded browser audio and transcribe it with the STT model."""
        audio_bytes = await audio.read(MAX_AUDIO_BYTES + 1)
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="The audio file is empty.")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="Audio must be 25 MB or smaller.")

        suffix = Path(audio.filename or "recording.webm").suffix or ".webm"
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
                temporary_file.write(audio_bytes)
                temporary_path = Path(temporary_file.name)

            waveform, sample_rate = torchaudio.load(temporary_path)
        except Exception as error:
            raise HTTPException(
                status_code=422,
                detail=f"The uploaded audio could not be decoded: {error}",
            ) from error
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        if waveform.numel() == 0:
            raise HTTPException(status_code=422, detail="The audio contains no samples.")

        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=16_000,
            )

        waveform = waveform.to(self._stt_device)
        with torch.inference_mode():
            token_ids = self._stt_model(waveform).argmax(dim=-1)[0].tolist()

        transcript = self._decode_ctc(token_ids)
        return {
            "text": transcript,
            "duration_seconds": waveform.shape[-1] / 16_000,
        }

    @staticmethod
    def _decode_ctc(token_ids: list[int]) -> str:
        """Collapse repeated CTC predictions and discard blank tokens."""
        blank_id = 0
        characters: list[str] = []
        previous_id: int | None = None

        for token_id in token_ids:
            if token_id != blank_id and token_id != previous_id:
                characters.append(VOCABULARY[token_id])
            previous_id = token_id

        return "".join(characters).strip()

    def _parse_action_graph(self, text: str) -> dict[str, Any]:
        """Run ``main.parse`` and decode the graph it currently prints."""
        output = StringIO()
        with redirect_stdout(output):
            parse(text)
        return json.loads(output.getvalue())

    def run(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Run the application with Uvicorn."""
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    Main().run()
