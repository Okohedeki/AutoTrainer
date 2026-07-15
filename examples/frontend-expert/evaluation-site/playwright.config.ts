import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  // Serial execution and a fixed local port keep paired evaluation arms on the
  // same deterministic browser harness while using only one local machine.
  fullyParallel: false,
  workers: 1,
  reporter: "line",
  use: {
    baseURL: "http://127.0.0.1:4174",
    browserName: "chromium",
  },
  webServer: {
    command: "npm run dev -- --host 0.0.0.0 --port 4174",
    url: "http://127.0.0.1:4174",
    reuseExistingServer: false,
  },
});
