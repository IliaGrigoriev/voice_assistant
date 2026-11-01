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

export async function transcribeAudio(audioBlob) {
  const extension = audioBlob.type.includes("ogg") ? "ogg" : "webm";
  const filename = audioBlob instanceof File ? audioBlob.name : `recording.${extension}`;
  const formData = new FormData();
  formData.append("audio", audioBlob, filename);

  const response = await fetch(`${API_BASE_URL}/transcribe`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail || `Request failed with status ${response.status}`);
  }

  return response.json();
}
