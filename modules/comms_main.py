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

from actions_from_intent.main import parse
from actions_from_intent.predicate_graphs import PredicateGraphs
from sst.const import STT_MODEL
from sst.custom_conformer import CustomConformer
from sst.stt_model import STTModel
from sst.text import TextProcessor
from sst.wav2vec import Wav2Vec


VOCABULARY = ["<blank>", " ", "'", *"abcdefghijklmnopqrstuvwxyz"]
HOP_LENGTH = 160
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_MODEL_BYTES = 2 * 1024 * 1024 * 1024
STT_MODELS_FOLDER = Path(STT_MODEL).parent.resolve()
UPLOADED_MODELS_FOLDER = STT_MODELS_FOLDER / "uploaded_models"


class Main:
    """Orchestrate assistant modules and expose them through FastAPI."""

    def __init__(self) -> None:
        PredicateGraphs.download_model()
        self._stt_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        default_model_path = str(Path(STT_MODEL).resolve())
        self._stt_models = {
            default_model_path: self._load_stt_model(default_model_path)
        }
        self.app = FastAPI(title="Voice scheduling assistant")
        self._configure_cors()
        self._register_routes()

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path:
        """Resolve a checkpoint while keeping access inside the model folder."""
        candidate = Path(model_path)
        if not candidate.is_absolute():
            candidate = STT_MODELS_FOLDER / candidate
        candidate = candidate.resolve()

        try:
            candidate.relative_to(STT_MODELS_FOLDER)
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=f"Model path must be inside {STT_MODELS_FOLDER}",
            ) from error

        if candidate.suffix != ".pt":
            raise HTTPException(
                status_code=400,
                detail="The selected model must be a .pt checkpoint.",
            )
        if not candidate.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"STT model not found at {candidate}",
            )
        return candidate

    def _load_stt_model(self, requested_path: str) -> STTModel:
        """Infer a checkpoint's architecture and load its trained weights."""
        model_path = self._resolve_model_path(requested_path)
        state_dict = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        architecture = self._infer_model_architecture(state_dict, model_path)
        if architecture == "CustomConformer":
            model = CustomConformer(
                vocab_size=len(VOCABULARY),
                hop_length=HOP_LENGTH,
                device=self._stt_device,
            )
        else:
            model = Wav2Vec(vocab_size=len(VOCABULARY))

        model.to(self._stt_device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @staticmethod
    def _infer_model_architecture(state_dict, model_path: Path) -> str:
        if any(key.startswith("_conformer.") for key in state_dict):
            return "CustomConformer"
        if any(key.startswith("encoder.") for key in state_dict):
            return "Wav2Vec"
        raise HTTPException(
            status_code=422,
            detail=f"Cannot infer the architecture of {model_path}",
        )

    def _get_stt_model(self, requested_path: str) -> STTModel:
        """Return a cached model, loading a checkpoint on first use."""
        model_path = str(self._resolve_model_path(requested_path))
        if model_path not in self._stt_models:
            self._stt_models[model_path] = self._load_stt_model(model_path)
        return self._stt_models[model_path]

    def list_stt_models(self) -> dict[str, Any]:
        """List selectable checkpoints under the configured model folder."""
        default_path = str(Path(STT_MODEL).resolve())
        model_paths = sorted(
            str(path.resolve())
            for path in STT_MODELS_FOLDER.rglob("*.pt")
            if path.is_file()
        )
        if default_path in model_paths:
            model_paths.remove(default_path)
            model_paths.insert(0, default_path)
        return {
            "default_model_path": default_path,
            "model_paths": model_paths,
        }

    async def upload_stt_model(
        self,
        model_file: UploadFile = File(...),
    ) -> dict[str, str]:
        """Store and validate a checkpoint selected in the browser."""
        filename = Path(model_file.filename or "").name
        if not filename or Path(filename).suffix != ".pt":
            raise HTTPException(
                status_code=400,
                detail="Choose a .pt model checkpoint.",
            )

        UPLOADED_MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
        destination = UPLOADED_MODELS_FOLDER / filename
        counter = 1
        while destination.exists():
            destination = UPLOADED_MODELS_FOLDER / (
                f"{Path(filename).stem}_{counter}.pt"
            )
            counter += 1

        bytes_written = 0
        try:
            with destination.open("xb") as model_output:
                while chunk := await model_file.read(8 * 1024 * 1024):
                    bytes_written += len(chunk)
                    if bytes_written > MAX_MODEL_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="Model files must be 2 GB or smaller.",
                        )
                    model_output.write(chunk)

            state_dict = torch.load(
                destination,
                map_location="cpu",
                weights_only=True,
            )
            architecture = self._infer_model_architecture(
                state_dict,
                destination,
            )
            del state_dict
        except HTTPException:
            destination.unlink(missing_ok=True)
            raise
        except Exception as error:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422,
                detail=f"The selected file is not a compatible model: {error}",
            ) from error
        finally:
            await model_file.close()

        return {
            "model_path": str(destination.resolve()),
            "architecture": architecture,
        }

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
        self.app.add_api_route(
            "/stt-models",
            self.list_stt_models,
            methods=["GET"],
        )
        self.app.add_api_route(
            "/stt-models/upload",
            self.upload_stt_model,
            methods=["POST"],
        )

    def recognize(self, text: str) -> dict[str, Any]:
        """Parse an utterance and return its semantic action graph."""
        return self._parse_action_graph(text)

    async def transcribe(
        self,
        audio: UploadFile = File(...),
        model_path: str = STT_MODEL,
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

        resolved_model_path = str(self._resolve_model_path(model_path))
        stt_model = self._get_stt_model(resolved_model_path)
        waveform = waveform.to(self._stt_device)
        with torch.inference_mode():
            if stt_model.uses_waveform_lengths:
                waveform_lengths = torch.tensor(
                    [waveform.shape[-1]],
                    device=self._stt_device,
                )
                logits, _ = stt_model(waveform, waveform_lengths)
            else:
                logits = stt_model(waveform)
            token_ids = logits.argmax(dim=-1)[0].tolist()

        transcript = self._decode_ctc(token_ids)
        return {
            "text": transcript,
            "duration_seconds": waveform.shape[-1] / 16_000,
            "model_path": resolved_model_path,
            "architecture": type(stt_model).__name__,
        }

    @staticmethod
    def _decode_ctc(token_ids: list[int]) -> str:
        """Collapse repeated CTC predictions and discard blank tokens."""
        return TextProcessor(VOCABULARY).decode(token_ids)

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
