import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { SetupApp } from "./SetupApp";
import "../seat/seat.css";
import "./setup.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <SetupApp />
  </StrictMode>
);
