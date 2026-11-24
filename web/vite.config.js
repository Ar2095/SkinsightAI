import { defineConfig } from "vite";

export default defineConfig({
  // The page imports eval/results/metrics.json and the figures from the repo root,
  // so the numbers on the site are the evaluation's output, not a copy.
  server: { fs: { allow: [".."] } },
  build: { assetsInlineLimit: 0 },
});
