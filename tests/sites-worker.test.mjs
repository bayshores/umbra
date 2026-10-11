import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
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
test('packaged worker keeps unknown API routes at 404 and ships the Nemotron proxy',async()=>{
  assert.ok(existsSync(new URL('../dist/server/nemotron.js',import.meta.url)));
  const packaged=(await import('../dist/server/index.js')).default;
  for(const route of ['/api/missing','/api/comparison','/api/nemotron/extra'])assert.equal((await packaged.fetch(new Request('https://local.test'+route,{headers:{accept:'text/html'}}),{ASSETS:assets})).status,404);
});
const ask=()=>new Request('https://local.test/api/nemotron',{method:'POST',headers:{'content-type':'application/json',origin:'https://local.test'},body:JSON.stringify({mode:'ask',question:'box truck',context:{}})});
test('unconfigured Nemotron route answers 503 JSON without leaving the worker',async()=>{
  for(const w of [worker,(await import('../dist/server/index.js')).default]) {
    const r=await w.fetch(ask(),{ASSETS:{fetch:()=>assert.fail('assets must not serve the API')}});
    assert.equal(r.status,503);assert.match(r.headers.get('content-type'),/^application\/json/);assert.equal((await r.json()).error,'assistant_unavailable');
    assert.equal((await w.fetch(new Request('https://local.test/api/nemotron'),{ASSETS:assets})).status,405);
  }
});
test('hosted Nemotron stays off even when host variables are set',async t=>{
  t.mock.method(globalThis,'fetch',()=>assert.fail('the hosted worker must not contact Token Factory'));
  const env={NEMOTRON_ENABLED:'1',NEMOTRON_BUDGET_USD:'1',TOKEN_FACTORY_API_KEY:'tf-test-key-not-real',ASSETS:{fetch:()=>assert.fail('assets must not serve the API')}};
  for(const w of [worker,(await import('../dist/server/index.js')).default]) {
    const r=await w.fetch(ask(),env),text=await r.text();
    assert.equal(r.status,503);assert.equal(JSON.parse(text).error,'assistant_unavailable');assert.ok(!text.includes(env.TOKEN_FACTORY_API_KEY));
  }
});
