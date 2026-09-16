import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Three entries against one build:
//   index.html -> the omniscient inspector, served at /org/inspector
//   seat.html  -> the human member workspace, served at /org/seat
//   setup.html -> the pack selection page, served at /org/setup
//   liaison.html -> the P3 organization-as-a-service façade, served at /org/liaison
// They are opposite views of the same world (everything, versus only what one
// member may see), so they share the build and the theme and nothing else.
//
// `base` must match the StaticFiles mount in backend/main.py, or the hashed
// asset URLs 404. In `npm run dev` we proxy /api to the running inspector
// server; it picks the first free port from 8100, so change this line if it
// does not land on 8101.
export default defineConfig({
  base: "/org/app/",
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1600,
    rollupOptions: {
      input: {
        inspector: "index.html",
        seat: "seat.html",
        setup: "setup.html",
        liaison: "liaison.html",
      },
    },
  },
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8101" },
  },
});
