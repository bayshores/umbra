// The app source of commit c7a39eb (placement protocol v3.2, ground.json schema 2), for the tests that still read the
// local grounded-v3.2 AV2 8c data. Since shape-v1 the app refuses schema 2 ground layers (src/scene.mjs) and an annotated
// vehicle without its LiDAR shape (src/geometry.mjs), so that data loads only with the app source of its own commit
// (shape-v1 protocol catalogAndServer.history; scripts/evaluation/metric-f.mjs --app-src does the same). These tests
// keep checking that the v3.2 contract numbers still hold for the code that produced them; the current app's refusal
// of grounded-v3.2 is tested in tests/av2-ground.test.mjs.
//
// The five files are read with git show from the pinned commit, checked against their SHA-256, written to a fresh
// temporary folder (src/ and scripts/evaluation/, so their relative imports resolve to each other) with .local linked
// to this repository's .local (pao-baseline.mjs reads the development data relative to its own folder), imported, and
// removed when the process exits. Only called when the local data is present; nothing here is used by the app.
import {execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {mkdtempSync,mkdirSync,writeFileSync,symlinkSync,rmSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';

export const HISTORY_COMMIT='c7a39eb3a04c023d6adcfc8b6d3d5cb6f81c48f2';
const FILES=Object.freeze({
  'src/geometry.mjs':'68cd5fed047aefc2383b60072a70da60756bc8e83be26e7b4838ad75f386ffdf',
  'src/scene.mjs':'7b5ace322eb3129fff0b3b30546da095f30150705dbdf23ac062164ca6cbedc2',
  'src/scenario-assistant.mjs':'00ffd0917d40c959c07bed1e4a3cef06770c494c4826dc3e9dea9b4ffa22373f',
  'src/suggestions.mjs':'e8fb67b93768dff2b4866f2ef83ae5c20d64eb2987a448369997e5ae74d928f4',
  'scripts/evaluation/pao-baseline.mjs':'82b40e7fb2513c356012f1087fe70e227bc4df74b2173988a0e4e505ee019f95',
});
const REPO=fileURLToPath(new URL('../',import.meta.url));
let loading=null;

async function load() {
  const root=mkdtempSync(path.join(os.tmpdir(),'sightline-v3.2-app-'));
  process.once('exit',()=>rmSync(root,{recursive:true,force:true}));
  for(const [file,sha256] of Object.entries(FILES)) {
    const bytes=execFileSync('git',['-C',REPO,'show',`${HISTORY_COMMIT}:${file}`],{maxBuffer:16*1024*1024});
    const actual=createHash('sha256').update(bytes).digest('hex');
    if(actual!==sha256)throw new Error(`${file} at ${HISTORY_COMMIT} has SHA-256 ${actual}, expected ${sha256}.`);
    mkdirSync(path.dirname(path.join(root,file)),{recursive:true});
    writeFileSync(path.join(root,file),bytes);
  }
  symlinkSync(path.join(REPO,'.local'),path.join(root,'.local'),'dir');
  const importFile=file=>import(pathToFileURL(path.join(root,file)).href);
  const [geometry,scene,assistant,suggestions,baseline]=await Promise.all(Object.keys(FILES).map(importFile));
  return {geometry,scene,assistant,suggestions,baseline,srcDir:path.join(root,'src')};
}
// {geometry, scene, assistant, suggestions, baseline (pao-baseline.mjs), srcDir}: the modules of HISTORY_COMMIT, loaded once.
export function historyApp() {
  loading??=load();
  return loading;
}
