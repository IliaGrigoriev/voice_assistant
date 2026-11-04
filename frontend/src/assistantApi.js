const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function sendAssistantMessage(message) {
  const text = encodeURIComponent(message);
  const response = await fetch(`${API_BASE_URL}/recognize?text=${text}`, {
    method: "POST",
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

export async function uploadSpeechModel(modelFile) {
  const formData = new FormData();
  formData.append("model_file", modelFile, modelFile.name);

  const response = await fetch(`${API_BASE_URL}/stt-models/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    if (response.status === 404) {
      throw new Error("Model upload is unavailable. Restart the backend and try again.");
    }
    throw new Error(payload?.detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}

export async function transcribeAudio(audioBlob, modelPath) {
  const extension = audioBlob.type.includes("ogg") ? "ogg" : "webm";
  const filename = audioBlob instanceof File ? audioBlob.name : `recording.${extension}`;
  const formData = new FormData();
  formData.append("audio", audioBlob, filename);

  const selectedPath = encodeURIComponent(modelPath);
  const response = await fetch(`${API_BASE_URL}/transcribe?model_path=${selectedPath}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}
