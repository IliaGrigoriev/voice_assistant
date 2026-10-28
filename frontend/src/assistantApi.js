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
