import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/ai": "http://localhost:8000",
      "/operator": "http://localhost:8000",
      "/agent": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
  },
});
