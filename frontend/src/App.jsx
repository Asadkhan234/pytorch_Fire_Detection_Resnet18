import { useState, useRef, useEffect } from "react";
import "./App.css";

const API_URL = "https://pytorch-fire-detection-resnet18.onrender.com";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  const isFire = String(prediction).toLowerCase() === "fire";

  const state = loading
    ? "scanning"
    : !prediction
    ? "idle"
    : isFire
    ? "fire"
    : "clear";

  const loadFile = (f) => {
    if (!f || !f.type.startsWith("image/")) {
      setError("That file isn't an image. Choose a PNG or JPG.");
      return;
    }

    setFile(f);
    setPreview(URL.createObjectURL(f));
    setPrediction(null);
    setConfidence(null);
    setError("");
  };

  const analyze = async () => {
    if (!file) return;

    setLoading(true);
    setPrediction(null);
    setConfidence(null);
    setError("");

    const body = new FormData();
    body.append("file", file);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body,
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data = await res.json();

      setPrediction(data.prediction);
      setConfidence(Number(data.confidence));
    } catch (error) {
      console.error("Prediction error:", error);

      setError(
        "Unable to connect to the Fire Detection API. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setPrediction(null);
    setConfidence(null);
    setError("");

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  return (
    <div className={`app is-${state}`}>
      <header className="bar">
        <div className="logo">
          <span className="logo-mark" aria-hidden="true" />
          FireDetect
        </div>

        <span className="model-tag">
          ResNet18 · PyTorch
        </span>
      </header>

      <main className="stage">
        <section className="viewer" aria-label="Image to analyze">
          <div
            className={`frame ${dragging ? "drag" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              loadFile(e.dataTransfer.files[0]);
            }}
          >
            <i className="corner tl" />
            <i className="corner tr" />
            <i className="corner bl" />
            <i className="corner br" />

            {preview ? (
              <img
                src={preview}
                alt="Image selected for analysis"
              />
            ) : (
              <button
                className="drop"
                onClick={() => inputRef.current?.click()}
              >
                <strong>Drop an image here</strong>
                <span>
                  or click to browse. PNG and JPG work.
                </span>
              </button>
            )}

            {loading && (
              <div
                className="scanline"
                aria-hidden="true"
              />
            )}
          </div>

          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg"
            hidden
            onChange={(e) => loadFile(e.target.files[0])}
          />

          <div className="controls">
            <button
              className="btn primary"
              onClick={analyze}
              disabled={!file || loading}
            >
              {loading ? "Analyzing…" : "Check for fire"}
            </button>

            <button
              className="btn"
              onClick={() => inputRef.current?.click()}
            >
              {file ? "Change image" : "Choose image"}
            </button>

            {file && (
              <button
                className="btn ghost"
                onClick={reset}
              >
                Clear
              </button>
            )}

            {file && (
              <span
                className="fname"
                title={file.name}
              >
                {file.name} ·{" "}
                {(file.size / 1024).toFixed(0)} KB
              </span>
            )}
          </div>

          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </section>

        <aside
          className="readout"
          aria-live="polite"
        >
          <p className="verdict-label">
            Result
          </p>

          {state === "idle" && (
            <>
              <h2 className="verdict">
                No image checked yet
              </h2>

              <p className="note">
                Add an image and select Check for fire.
                The result and the model's confidence
                appear here.
              </p>
            </>
          )}

          {state === "scanning" && (
            <>
              <h2 className="verdict">
                Analyzing…
              </h2>

              <p className="note">
                The model is scanning the image.
              </p>
            </>
          )}

          {(state === "fire" || state === "clear") && (
            <>
              <h2 className="verdict">
                {isFire
                  ? "Fire detected"
                  : "No fire detected"}
              </h2>

              <div className="meter">
                <div className="meter-head">
                  <span>Confidence</span>

                  <strong>
                    {confidence?.toFixed(1)}%
                  </strong>
                </div>

                <div className="meter-track">
                  <span
                    className="meter-pin"
                    style={{
                      left: `${Math.min(
                        100,
                        Math.max(0, confidence || 0)
                      )}%`,
                    }}
                  />
                </div>

                <div className="meter-scale">
                  <span>0</span>
                  <span>50</span>
                  <span>100</span>
                </div>
              </div>

              <p className="note">
                {isFire
                  ? "The model found visual signs of fire in this image. Confirm with a person before acting."
                  : "The model found no visual signs of fire in this image. This is not a safety guarantee."}
              </p>
            </>
          )}
        </aside>
      </main>
    </div>
  );
}

export default App;