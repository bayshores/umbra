#!/usr/bin/env node
import { copyFileSync, existsSync, mkdirSync, rmSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");
const index = path.join(dist, "client", "index.html");
const worker = path.join(root, "worker", "index.js");
const assistant = path.join(root, "worker", "nemotron.js");
const hosting = path.join(root, ".openai", "hosting.json");

for (const file of [index, worker, assistant, hosting]) {
  if (!existsSync(file)) throw new Error("Missing Sites build input: " + file);
}

mkdirSync(path.join(dist, "server"), { recursive: true });
mkdirSync(path.join(dist, ".openai"), { recursive: true });
copyFileSync(worker, path.join(dist, "server", "index.js"));
copyFileSync(assistant, path.join(dist, "server", "nemotron.js"));
for (const obsolete of ['server/export.mjs', 'src/geometry.mjs', 'src/scene.mjs']) {
  rmSync(path.join(dist, obsolete), { force: true });
}
copyFileSync(hosting, path.join(dist, ".openai", "hosting.json"));

console.log("Prepared Sites build: dist/server/index.js, dist/server/nemotron.js and dist/.openai/hosting.json");
