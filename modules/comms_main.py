"""Main application that connects the frontend to the assistant modules."""

from contextlib import redirect_stdout
from io import StringIO
import json
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from intent_recogniton.main import parse
from intent_recogniton.predicate_graphs import PredicateGraphs


class Main:
    """Orchestrate assistant modules and expose them through FastAPI."""

    def __init__(self) -> None:
        PredicateGraphs.download_model()
        self.app = FastAPI(title="Voice scheduling assistant")
        self._configure_cors()
        self._register_routes()

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

    def recognize(self, text: str) -> dict[str, Any]:
        """Parse an utterance and return its semantic action graph."""
        return self._parse_action_graph(text)

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
