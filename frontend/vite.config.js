import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backendOrigin = process.env.BACKEND_ORIGIN || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 4173,
    open: false,
    proxy: {
      "/api": {
        target: backendOrigin,
        changeOrigin: true,
      },
      "/reports": {
        target: backendOrigin,
        changeOrigin: true,
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 800,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/three")) {
            return "three-vendor";
          }

          if (id.includes("@react-three") || id.includes("@use-gesture") || id.includes("meshline")) {
            return "r3f-vendor";
          }

          if (id.includes("framer-motion")) {
            return "motion-vendor";
          }
        },
      },
    },
  },
});
