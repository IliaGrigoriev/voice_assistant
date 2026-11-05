"""Main application that connects the frontend to the assistant modules."""

from contextlib import redirect_stdout
from io import BytesIO, StringIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
import wave

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
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
STT_MODELS_FOLDER = Path(STT_MODEL).resolve().parents[1]
DENOISER_CACHE = STT_MODELS_FOLDER / "noise_filter" / "metricgan"


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
        self._denoiser = None
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
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )
        state_dict = self._extract_model_state_dict(checkpoint, model_path)

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
    def _extract_model_state_dict(checkpoint, model_path: Path):
        """Return weights from either a raw state dict or trainer checkpoint."""
        if not isinstance(checkpoint, dict):
            raise HTTPException(
                status_code=422,
                detail=f"{model_path} does not contain a model state dictionary",
            )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise HTTPException(
                status_code=422,
                detail=f"{model_path} has an invalid model_state_dict",
            )
        return state_dict

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

    def _get_denoiser(self):
        """Load and cache the pretrained speech-enhancement model."""
        if self._denoiser is None:
            from speechbrain.inference.enhancement import (
                SpectralMaskEnhancement,
            )

            DENOISER_CACHE.mkdir(parents=True, exist_ok=True)
            self._denoiser = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank",
                savedir=str(DENOISER_CACHE),
                run_opts={"device": str(self._stt_device)},
            )
        return self._denoiser

    def _filter_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Enhance a mono 16 kHz waveform before speech recognition."""
        denoiser = self._get_denoiser()
        noisy = waveform.to(self._stt_device)
        lengths = torch.ones(noisy.shape[0], device=self._stt_device)
        with torch.inference_mode():
            filtered = denoiser.enhance_batch(noisy, lengths=lengths)
        return filtered.detach().to(dtype=torch.float32, device="cpu")

    @staticmethod
    def _wav_bytes(waveform: torch.Tensor, sample_rate: int) -> bytes:
        """Encode a waveform as an in-memory PCM WAV file."""
        mono = waveform.squeeze(0).clamp(-1, 1)
        pcm = (mono * 32767).to(torch.int16).numpy().tobytes()
        output = BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm)
        return output.getvalue()

    @staticmethod
    async def _read_audio(audio: UploadFile) -> torch.Tensor:
        """Decode uploaded audio and normalize it to mono 16 kHz."""
        audio_bytes = await audio.read(MAX_AUDIO_BYTES + 1)
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="The audio file is empty.")
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            raise HTTPException(
                status_code=413, detail="Audio must be 25 MB or smaller."
            )

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
            raise HTTPException(
                status_code=422, detail="The audio contains no samples."
            )
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=16_000
            )
        return waveform

    async def filter_audio(
        self, audio: UploadFile = File(...),
    ) -> Response:
        """Return a filtered recording as a browser-playable WAV response."""
        waveform = await self._read_audio(audio)
        try:
            filtered = self._filter_audio(waveform)
        except Exception as error:
            raise HTTPException(
                status_code=503, detail=f"Audio filtering failed: {error}"
            ) from error
        return Response(
            content=self._wav_bytes(filtered, 16_000),
            media_type="audio/wav",
        )

    def list_stt_models(self) -> dict[str, Any]:
        """List selectable checkpoints under the configured model root."""
        default_path = str(Path(STT_MODEL).resolve())
        model_paths = sorted(
            str(path.resolve())
            for path in STT_MODELS_FOLDER.rglob("*.pt")
            if path.is_file()
            and "uploaded_models" not in path.relative_to(
                STT_MODELS_FOLDER
            ).parts
        )
        if default_path in model_paths:
            model_paths.remove(default_path)
            model_paths.insert(0, default_path)
        return {
            "default_model_path": default_path,
            "model_paths": model_paths,
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
            "/filter-audio",
            self.filter_audio,
            methods=["POST"],
        )
        self.app.add_api_route(
            "/stt-models",
            self.list_stt_models,
            methods=["GET"],
        )

    def recognize(self, text: str) -> dict[str, Any]:
        """Parse an utterance and return its semantic action graph."""
        return self._parse_action_graph(text)

    async def transcribe(
        self,
        audio: UploadFile = File(...),
        model_path: str = STT_MODEL,
        apply_filter: bool = False,
    ) -> dict[str, Any]:
        """Decode uploaded browser audio and transcribe it with the STT model."""
        waveform = await self._read_audio(audio)
        if apply_filter:
            try:
                waveform = self._filter_audio(waveform)
            except Exception as error:
                raise HTTPException(
                    status_code=503,
                    detail=f"Audio filtering failed: {error}",
                ) from error

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
