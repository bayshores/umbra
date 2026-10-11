import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, isFileLoadingAllowed, normalizePath } from "vite";
import react from "@vitejs/plugin-react";
import { nemotronMiddleware } from "./worker/nemotron-file-budget.mjs";

// fs.deny replaces Vite's defaults, so they are repeated. Private project context (.local/, handoffs, agent folders and
// instructions) and *.key files are never served, not even to other localhost origins that Vite's default CORS rule admits.
const ROOT = normalizePath(path.dirname(fileURLToPath(import.meta.url))).replace(/[*?[\]{}()!+@]/g, "\\$&");

// /api/nemotron for `npm run dev` and `npm run preview`, loopback only (allowedHosts does not apply). A key file this server
// could serve is refused.
function nemotronApi() {
  const mount = server => server.middlewares.use(nemotronMiddleware(server.config.root,
    { servable: file => isFileLoadingAllowed(server.config, normalizePath(file)) }));
  return {
    name: "sightline-nemotron-api",
    configureServer(server) { mount(server); },
    configurePreviewServer(server) { mount(server); },
  };
}

export default defineConfig({
  appType: 'mpa',
  build: {
    outDir: "dist/client",
  },
  optimizeDeps: {
    include: ["react", "react-dom/client"],
  },
  server: {
    host: "127.0.0.1",
    allowedHosts: ["terminal.local"],
    fs: {
      deny: [".env", ".env.*", "*.{crt,pem}", "**/.git/**", "*.key", `${ROOT}/.local/**`, "**/.handoffs/**", "**/.agents/**",
        "**/.claude/**", "**/.openai/**", "AGENTS.md", "CLAUDE.md", "design-qa.md", ".npmrc", "rulesync.jsonc", "**/.rulesync/**"],
    },
    warmup: {
      clientFiles: ["./src/main.jsx"],
    },
  },
  plugins: [react(), nemotronApi()],
});
