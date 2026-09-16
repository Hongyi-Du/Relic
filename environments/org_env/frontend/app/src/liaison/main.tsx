import React from "react";
import { createRoot } from "react-dom/client";
import { LiaisonApp } from "./LiaisonApp";
import "./liaison.css";

type LiaisonErrorBoundaryState = { error: Error | null };

class LiaisonErrorBoundary extends React.Component<React.PropsWithChildren, LiaisonErrorBoundaryState> {
  state: LiaisonErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): LiaisonErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("P3 liaison render failure", error, info);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return <main className="liaison-crash" role="alert">
      <span className="brand-mark">S</span>
      <h1>The secretary view hit a rendering error.</h1>
      <p>Your message and organization state are still stored. Reload the view to reconnect; the simulation is not restarted.</p>
      <code>{this.state.error.message || "Unknown rendering error"}</code>
      <button type="button" onClick={() => window.location.reload()}>Reload secretary view</button>
    </main>;
  }
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <LiaisonErrorBoundary>
      <LiaisonApp />
    </LiaisonErrorBoundary>
  </React.StrictMode>,
);
