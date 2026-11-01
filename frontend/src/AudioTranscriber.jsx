import { useEffect, useRef, useState } from "react";
import { transcribeAudio } from "./assistantApi.js";

function MicrophoneIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <rect x="8" y="3" width="8" height="12" rx="4" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3M9 21h6" />
    </svg>
  );
}

export default function AudioTranscriber() {
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
  }, [audioUrl]);

  function retainAudio(blob) {
    setAudioUrl((currentUrl) => {
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      return URL.createObjectURL(blob);
    });
    setAudioBlob(blob);
    setResult(null);
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const preferredType = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"]
        .find((type) => MediaRecorder.isTypeSupported(type));
      const recorder = new MediaRecorder(
        stream,
        preferredType ? { mimeType: preferredType } : undefined,
      );

      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        retainAudio(new Blob(chunksRef.current, { type: recorder.mimeType }));
        stream.getTracks().forEach((track) => track.stop());
      };
      recorderRef.current = recorder;
      recorder.start();
      setResult(null);
      setIsRecording(true);
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : "Microphone access failed." });
    }
  }

  function stopRecording() {
    recorderRef.current?.stop();
    setIsRecording(false);
  }

  function chooseAudio(event) {
    const [file] = event.target.files;
    if (file) retainAudio(file);
  }

  async function handleTranscribe() {
    if (!audioBlob || isTranscribing) return;

    setIsTranscribing(true);
    setResult(null);
    try {
      setResult(await transcribeAudio(audioBlob));
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : "Transcription failed." });
    } finally {
      setIsTranscribing(false);
    }
  }

  return (
    <section className="transcriber-card" aria-labelledby="transcriber-title">
      <div className="transcriber-copy">
        <div className="eyebrow">
          <span className="status-dot" aria-hidden="true" />
          Local speech recognition
        </div>
        <h2 id="transcriber-title">Audio to text</h2>
        <p>Record a message or select an audio file, then send it to the local STT model.</p>
      </div>

      <div className="audio-controls">
        <div className="record-actions">
          {isRecording ? (
            <button className="record-button is-recording" type="button" onClick={stopRecording}>
              <span className="stop-icon" aria-hidden="true" />
              Stop recording
            </button>
          ) : (
            <button className="record-button" type="button" onClick={startRecording}>
              <MicrophoneIcon />
              Record audio
            </button>
          )}

          <label className="file-button">
            Choose audio
            <input type="file" accept="audio/*" onChange={chooseAudio} />
          </label>
        </div>

        {audioUrl && <audio className="audio-preview" src={audioUrl} controls />}

        <button
          className="transcribe-button"
          type="button"
          disabled={!audioBlob || isRecording || isTranscribing}
          onClick={handleTranscribe}
        >
          {isTranscribing ? "Transcribing…" : "Transcribe audio"}
        </button>
      </div>

      <div className={`transcript-output ${result ? "has-result" : ""}`} aria-live="polite">
        {result?.error ? (
          <p className="error">{result.error}</p>
        ) : result ? (
          <>
            <span>Transcript</span>
            <p>{result.text || "No speech was decoded."}</p>
            <small>{result.duration_seconds.toFixed(1)} seconds of audio</small>
          </>
        ) : (
          <p className="empty-state">Your transcript will appear here.</p>
        )}
      </div>
    </section>
  );
}
