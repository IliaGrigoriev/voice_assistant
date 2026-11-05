import { useEffect, useRef, useState } from "react";
import { filterAudio, getSpeechModels, transcribeAudio } from "./assistantApi.js";

function modelLabel(path) {
  return path.split(/[\\/]/).pop() || path;
}

const DEFAULT_MODEL_FOLDER = "/home/ilia/Data/vsa/available_models";

function isModelInFolder(path, folder) {
  const normalizedFolder = folder.trim().replaceAll("\\", "/").replace(/\/$/, "");
  if (!normalizedFolder) return false;

  const normalizedPath = path.replaceAll("\\", "/");
  const parent = normalizedPath.slice(0, normalizedPath.lastIndexOf("/"));
  if (normalizedFolder.startsWith("/")) return parent === normalizedFolder;
  return normalizedFolder.includes("/")
    ? parent.endsWith(`/${normalizedFolder}`)
    : parent.split("/").pop() === normalizedFolder;
}

function MicrophoneIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      <rect x="8" y="3" width="8" height="12" rx="4" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3M9 21h6" />
    </svg>
  );
}

export default function AudioTranscriber({ onTranscription }) {
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const modelFolderInputRef = useRef(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [filteredAudioBlob, setFilteredAudioBlob] = useState(null);
  const [filteredAudioUrl, setFilteredAudioUrl] = useState("");
  const [applyNoiseFilter, setApplyNoiseFilter] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isFiltering, setIsFiltering] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [selectedModelPath, setSelectedModelPath] = useState("");
  const [allModelPaths, setAllModelPaths] = useState([]);
  const [modelPaths, setModelPaths] = useState([]);
  const [modelFolder, setModelFolder] = useState(DEFAULT_MODEL_FOLDER);
  const [defaultModelPath, setDefaultModelPath] = useState("");
  const [modelError, setModelError] = useState("");
  const [result, setResult] = useState(null);

  useEffect(() => () => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
  }, [audioUrl]);

  useEffect(() => () => {
    if (filteredAudioUrl) URL.revokeObjectURL(filteredAudioUrl);
  }, [filteredAudioUrl]);

  useEffect(() => {
    getSpeechModels()
      .then(({ default_model_path: defaultPath, model_paths: paths }) => {
        setAllModelPaths(paths);
        setDefaultModelPath(defaultPath);
      })
      .catch((error) => {
        setModelError(error instanceof Error ? error.message : "Could not list models.");
      });
  }, []);

  useEffect(() => {
    const paths = allModelPaths.filter((path) => isModelInFolder(path, modelFolder));
    setModelPaths(paths);
    setSelectedModelPath((currentPath) => (
      paths.includes(currentPath)
        ? currentPath
        : paths.includes(defaultModelPath)
          ? defaultModelPath
          : paths[0] ?? ""
    ));
  }, [allModelPaths, defaultModelPath, modelFolder]);

  function retainAudio(blob) {
    setAudioUrl((currentUrl) => {
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      return URL.createObjectURL(blob);
    });
    setAudioBlob(blob);
    setFilteredAudioBlob(null);
    setFilteredAudioUrl((currentUrl) => {
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      return "";
    });
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

  function chooseModelFolder(event) {
    const files = Array.from(event.target.files ?? []);
    const relativePath = files[0]?.webkitRelativePath;
    if (relativePath) {
      setModelFolder(relativePath.split("/")[0]);
    }
    event.target.value = "";
  }

  function retainFilteredAudio(filtered) {
    setFilteredAudioBlob(filtered);
    setFilteredAudioUrl((currentUrl) => {
      if (currentUrl) URL.revokeObjectURL(currentUrl);
      return URL.createObjectURL(filtered);
    });
  }

  async function handleFilter() {
    if (!audioBlob || !applyNoiseFilter || isFiltering || isTranscribing) return;
    setIsFiltering(true);
    setResult(null);
    try {
      const filtered = await filterAudio(audioBlob);
      retainFilteredAudio(filtered);
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : "Filtering failed." });
    } finally {
      setIsFiltering(false);
    }
  }

  async function handleTranscribe() {
    if (!audioBlob || isTranscribing) return;

    setIsTranscribing(true);
    setResult(null);
    try {
      let transcriptionAudio = audioBlob;
      if (applyNoiseFilter) {
        transcriptionAudio = filteredAudioBlob;
        if (!transcriptionAudio) {
          transcriptionAudio = await filterAudio(audioBlob);
          retainFilteredAudio(transcriptionAudio);
        }
      }
      const transcription = await transcribeAudio(
        transcriptionAudio,
        selectedModelPath,
        false,
      );
      setResult(transcription);
      if (transcription.text) onTranscription(transcription.text);
    } catch (error) {
      setResult({ error: error instanceof Error ? error.message : "Transcription failed." });
    } finally {
      setIsTranscribing(false);
    }
  }

  return (
    <section className="audio-input" aria-labelledby="audio-input-title">
      <div className="audio-input-heading">
        <div>
          <strong id="audio-input-title">Add from audio</strong>
          <span>Transcription will fill the editable request above.</span>
        </div>
        <MicrophoneIcon />
      </div>
      <div className="audio-controls">
        <div className="model-select">
          <label className="model-type-field">
            <span>Available models</span>
            <select
              value={selectedModelPath}
              aria-label="Available speech model"
              disabled={isRecording || isTranscribing}
              onChange={(event) => {
                setSelectedModelPath(event.target.value);
                setResult(null);
              }}
            >
              {!modelPaths.length && <option value="">No models available</option>}
              {modelPaths.map((path) => (
                <option key={path} value={path}>{modelLabel(path)}</option>
              ))}
            </select>
          </label>
          <div className="model-folder-settings">
            <span>Looking for models in</span>
            <code title={modelFolder}>{modelFolder}</code>
            <div className="model-folder-actions">
              <button
                type="button"
                className="folder-button"
                onClick={() => modelFolderInputRef.current?.click()}
              >
                Choose models folder
              </button>
            </div>
            <input
              ref={modelFolderInputRef}
              className="directory-input"
              type="file"
              accept=".pt,application/octet-stream"
              webkitdirectory=""
              directory=""
              multiple
              onChange={chooseModelFolder}
            />
          </div>
          {modelError && <small className="model-error">{modelError}</small>}
        </div>

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

        <label className="filter-option">
          <input
            type="checkbox"
            checked={applyNoiseFilter}
            disabled={!audioBlob || isFiltering || isTranscribing}
            onChange={(event) => {
              setApplyNoiseFilter(event.target.checked);
              setResult(null);
            }}
          />
          Apply noise filter before transcription
        </label>

        <div className="processing-actions">
          <button
            className="filter-button"
            type="button"
            disabled={!audioBlob || !applyNoiseFilter || isRecording || isFiltering || isTranscribing}
            onClick={handleFilter}
          >
            {isFiltering ? "Filtering…" : "Filter audio"}
          </button>
          <button
            className="transcribe-button"
            type="button"
            disabled={!audioBlob || !selectedModelPath || isRecording || isFiltering || isTranscribing}
            onClick={handleTranscribe}
          >
            {isTranscribing ? "Transcribing…" : "Transcribe audio"}
          </button>
        </div>

        {applyNoiseFilter && filteredAudioUrl && (
          <div className="filtered-preview">
            <span>Filtered audio</span>
            <audio className="audio-preview" src={filteredAudioUrl} controls />
          </div>
        )}
      </div>

      <div className="transcription-status" aria-live="polite">
        {result?.error ? (
          <p className="error">{result.error}</p>
        ) : result ? (
          <p>{result.text ? "Transcript added above — review or correct it before sending." : "No speech was decoded."}</p>
        ) : null}
      </div>
    </section>
  );
}
