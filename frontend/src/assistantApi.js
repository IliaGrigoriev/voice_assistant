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

export async function getSpeechModels() {
  const response = await fetch(`${API_BASE_URL}/stt-models`);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail || `Request failed with status ${response.status}`);
  }
  return response.json();
}

function audioFilename(audioBlob) {
  const extension = audioBlob.type.includes("wav")
    ? "wav"
    : audioBlob.type.includes("ogg") ? "ogg" : "webm";
  return audioBlob instanceof File ? audioBlob.name : `recording.${extension}`;
}

async function audioFormData(audioBlob) {
  const formData = new FormData();
  formData.append("audio", audioBlob, audioFilename(audioBlob));
  return formData;
}

export async function filterAudio(audioBlob) {
  const response = await fetch(`${API_BASE_URL}/filter-audio`, {
    method: "POST",
    body: await audioFormData(audioBlob),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail || `Request failed with status ${response.status}`);
  }
  return response.blob();
}

export async function transcribeAudio(audioBlob, modelPath, applyFilter = false) {
  const formData = await audioFormData(audioBlob);

  const selectedPath = encodeURIComponent(modelPath);
  const response = await fetch(`${API_BASE_URL}/transcribe?model_path=${selectedPath}&apply_filter=${applyFilter}`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}
