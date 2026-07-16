import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  // Development requests stay same-origin in React and are forwarded to the
  // loopback-only Python service used by the installed desktop workflow.
  server: {
    port: 3000,
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});
