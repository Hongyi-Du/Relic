// Pack selection and configuration page
import { useEffect, useState } from "react";

type Pack = {
  id: string;
  name: string;
  version: string;
  description: string;
  issues: number;
  prs: number;
};

export function SetupApp() {
  const liaisonSetup = new URLSearchParams(window.location.search).get("next") === "liaison";
  const [packs, setPacks] = useState<Pack[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [warmup, setWarmup] = useState(liaisonSetup ? 0 : 24);
  const [clockSpeed, setClockSpeed] = useState(30);  // Default 30 seconds
  const [userId, setUserId] = useState("");  // User custom ID
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    document.title = liaisonSetup ? "Secretary · Setup" : "LanternForge - Setup";
  }, [liaisonSetup]);

  useEffect(() => {
    fetch("/api/org/packs")
      .then((r) => r.json())
      .then(setPacks)
      .catch((e) => setError(`Failed to load packs: ${e.message}`));
  }, []);

  const startOrg = async () => {
    if (!selected) {
      setError("Please select a pack");
      return;
    }

    setStarting(true);
    setError("");

    try {
      const res = await fetch("/api/org/init", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pack: selected,
          warmup,
          clock_speed: clockSpeed,
          user_id: userId || undefined,  // Include user ID if provided
        }),
      });

      const data = await res.json();

      if (data.error) {
        setError(data.error);
        setStarting(false);
        return;
      }

      window.location.href = liaisonSetup ? "/org/liaison" : "/org/seat";
    } catch (e: any) {
      setError(`Failed to initialize: ${e.message}`);
      setStarting(false);
    }
  };

  return (
    <div className="setup-app">
      <header className="setup-header">
        <h1>{liaisonSetup ? "Secretary" : "LanternForge"}</h1>
        <p className="setup-subtitle">{liaisonSetup ? "Choose Victor's project" : "Choose Your Project"}</p>
      </header>

      {error && (
        <div className="setup-error">
          {error}
        </div>
      )}

      <div className="pack-grid">
        {packs.map((pack) => (
          <button
            key={pack.id}
            className={`pack-card ${selected === pack.id ? "selected" : ""}`}
            onClick={() => setSelected(pack.id)}
            disabled={starting}
          >
            <div className="pack-name">{pack.name}</div>
            {pack.version && (
              <div className="pack-version">{pack.version}</div>
            )}
            <div className="pack-description">{pack.description}</div>
            <div className="pack-stats">
              {pack.issues > 0 && <span>{pack.issues} issues</span>}
              {pack.prs > 0 && <span>{pack.prs} PRs</span>}
            </div>
          </button>
        ))}
      </div>

      {packs.length === 0 && !error && (
        <div className="setup-loading">Loading packs...</div>
      )}

      <div className="setup-config">
        <h2>Configuration</h2>
        <div className="config-row">
          <label>
            <span className="config-label">Warmup</span>
            <div className="config-input-group">
              <button
                className="step-btn"
                onClick={() => setWarmup(Math.max(0, warmup - 24))}
                disabled={starting || warmup <= 0}
              >−</button>
              <input
                type="number"
                value={warmup}
                onChange={(e) => setWarmup(Math.max(0, parseInt(e.target.value) || 0))}
                min="0"
                max="168"
                step="24"
                disabled={starting}
              />
              <button
                className="step-btn"
                onClick={() => setWarmup(Math.min(168, warmup + 24))}
                disabled={starting || warmup >= 168}
              >+</button>
            </div>
            <span className="config-unit">organizational hours (step: 24h)</span>
          </label>
          <p className="config-help">
            Let the team work before you join, so there's history when you arrive.
          </p>
        </div>

        <div className="config-row">
          <label>
            <span className="config-label">Clock speed</span>
            <div className="config-input-group">
              <button
                className="step-btn"
                onClick={() => setClockSpeed(Math.max(5, clockSpeed - 5))}
                disabled={starting || clockSpeed <= 5}
              >−</button>
              <input
                type="number"
                value={clockSpeed}
                onChange={(e) => setClockSpeed(Math.max(0.1, parseFloat(e.target.value) || 30))}
                min="0.1"
                max="3600"
                step="5"
                disabled={starting}
              />
              <button
                className="step-btn"
                onClick={() => setClockSpeed(Math.min(3600, clockSpeed + 5))}
                disabled={starting || clockSpeed >= 3600}
              >+</button>
            </div>
            <span className="config-unit">seconds per org hour (step: 5s)</span>
          </label>
          <p className="config-help">
            How fast the world runs. Lower = faster.
          </p>
        </div>

        <div className="config-row">
          <label>
            <span className="config-label">Your ID (optional)</span>
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="e.g., researcher_001"
              maxLength={50}
              disabled={starting}
            />
            <span className="config-unit">custom identifier for trajectory</span>
          </label>
          <p className="config-help">
            Optional custom ID included in trajectory metadata for tracking.
          </p>
        </div>

        <button
          className="start-btn"
          disabled={!selected || starting}
          onClick={startOrg}
        >
          {starting ? "Initializing world..." : "Start Organization"}
        </button>
      </div>
    </div>
  );
}
