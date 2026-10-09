import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import worker from '../worker/index.js';
import { scene } from './scene-input.mjs';
const assets={fetch:async request=>{try{return new Response(await readFile(new URL('../dist/client'+new URL(request.url).pathname,import.meta.url)));}catch(e){if(e.code==='ENOENT')return new Response(null,{status:404});throw e;}}};
test('unknown routes and missing assets stay 404 even when HTML is requested',async()=>{
  for(const route of ['/flow/step-two','/api/missing','/api/comparison','/scenes/missing/depth.bin'])assert.equal((await worker.fetch(new Request('https://local.test'+route,{headers:{accept:'text/html'}}),{ASSETS:assets})).status,404);
});
test('packaged worker serves the actual measured scene metadata',async()=>{
  const packaged=(await import('../dist/server/index.js')).default;
  const r=await packaged.fetch(new Request(`https://local.test/scenes/${scene.id}/scene.json`),{ASSETS:assets});
  assert.equal(r.status,200);const metadata=await r.json();assert.equal(metadata.id,scene.id);assert.deepEqual(metadata.artifactHashes,scene.artifactHashes);
});
