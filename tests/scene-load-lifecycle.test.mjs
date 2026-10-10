import assert from 'node:assert/strict';
import test from 'node:test';
import {createServer} from 'node:http';
import {readFile} from 'node:fs/promises';
import {loadScene} from '../src/scene.mjs';
import {scene,fetchAsset} from './scene-input.mjs';

// Controlled transport failures only; actual saved metadata and geometry remain unchanged.
test('caller abort cancels a stalled metadata body without waiting for its cancel hook',async()=>{
  let entered,cancelled=false;
  const reading=new Promise(resolve=>entered=resolve),controller=new AbortController();
  const pending=loadScene(async()=>new Response(new ReadableStream({
    pull(){entered();return new Promise(()=>{});},
    cancel(){cancelled=true;return new Promise(()=>{});},
  },{highWaterMark:0})),scene.id,undefined,controller.signal);
  const stopped=new Error('Caller stopped scene loading.');
  const rejected=assert.rejects(pending,error=>error===stopped);
  await reading;controller.abort(stopped);await rejected;
  assert.equal(cancelled,true);
});

test('failure cancels parallel sibling body reads and preserves the first error',async()=>{
  let entered,fail,cancelled=false;
  const reading=new Promise(resolve=>entered=resolve);
  const overflow=new Promise(resolve=>fail=resolve);
  const pending=loadScene(async(path,options)=>{
    assert.ok(options.signal instanceof AbortSignal);
    if(path.endsWith('/'+scene.buffers.positions))return overflow;
    if(path.endsWith('/'+scene.buffers.colors))return new Response(new ReadableStream({
      pull(){entered();return new Promise(()=>{});},
      cancel(){cancelled=true;throw new Error('Underlying cancellation failed.');},
    },{highWaterMark:0}));
    return fetchAsset(path);
  },scene.id);
  const rejected=assert.rejects(pending,/Artifact exceeds declared byte length/);
  await reading;fail(new Response(new Uint8Array(scene.pointCount*12+1)));await rejected;
  assert.equal(cancelled,true);
});

for(const stage of ['headers','body'])test(`30-second deadline rejects stalled metadata ${stage}`,async t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  let entered,observedSignal,cancelled=false;
  const reading=new Promise(resolve=>entered=resolve);
  const pending=loadScene(async(path,{signal})=>{
    observedSignal=signal;
    if(stage==='headers'){entered();return new Promise(()=>{});}
    return new Response(new ReadableStream({pull(){entered();return new Promise(()=>{});},cancel(){cancelled=true;}},{highWaterMark:0}));
  },scene.id);
  const rejected=assert.rejects(pending,/Scene loading exceeded 30 seconds/);
  await reading;t.mock.timers.tick(29_999);assert.equal(observedSignal.aborted,false);
  t.mock.timers.tick(1);await rejected;assert.equal(observedSignal.aborted,true);
  if(stage==='body')assert.equal(cancelled,true);
});

test('pre-aborted caller performs no fetch and late uncooperative headers are discarded',async()=>{
  const controller=new AbortController(),reason=new Error('No longer needed');controller.abort(reason);
  let requests=0;
  await assert.rejects(loadScene(async()=>{requests++;},scene.id,undefined,controller.signal),error=>error===reason);
  assert.equal(requests,0);
  const late=new AbortController();let resolveHeaders,cancelled=false,entered;
  const enteredFetch=new Promise(resolve=>entered=resolve);
  const pending=loadScene(async()=>{entered();return new Promise(resolve=>resolveHeaders=resolve);},scene.id,undefined,late.signal);
  const rejected=assert.rejects(pending,error=>error===reason);
  await enteredFetch;late.abort(reason);await rejected;
  resolveHeaders(new Response(new ReadableStream({cancel(){cancelled=true;}},{highWaterMark:0})));
  await new Promise(resolve=>setImmediate(resolve));assert.equal(cancelled,true);
});

test('successful actual load clears its deadline and caller listener',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  const caller=new AbortController();let observedSignal;
  const loaded=await loadScene((path,options)=>{observedSignal=options.signal;return fetchAsset(path);},scene.id,undefined,caller.signal);
  assert.deepEqual(loaded.positions,scene.positions);
  t.mock.timers.tick(30_000);caller.abort();assert.equal(observedSignal.aborted,false);
});

for(const stage of ['between_headers_and_body','non_ok_response'])test(`custom transport body is cancelled at ${stage}`,async()=>{
  const caller=new AbortController();let cancelled=false;
  const response=new Response(new ReadableStream({cancel(){cancelled=true;}},{highWaterMark:0}),{status:stage==='non_ok_response'?503:200});
  await assert.rejects(loadScene(async()=>{
    if(stage==='between_headers_and_body')queueMicrotask(()=>queueMicrotask(()=>caller.abort(new Error('Stopped between phases.'))));
    return response;
  },scene.id,undefined,caller.signal),stage==='non_ok_response'?/metadata unavailable \(503\)/:/Stopped between phases/);
  assert.equal(cancelled,true);
});

test('actual fetch abort closes a stalled loopback response after sibling overflow',async t=>{
  let started,closed;
  const positionsStarted=new Promise(resolve=>started=resolve),positionsClosed=new Promise(resolve=>closed=resolve);
  const allowed=new Set(['scene.json',...Object.values(scene.buffers)].map(name=>`/scenes/${scene.id}/${name}`));
  const server=createServer(async(req,res)=>{
    if(!allowed.has(req.url)){res.writeHead(404);res.end();return;}
    if(req.url.endsWith('/'+scene.buffers.positions)) {
      res.on('close',()=>closed(!res.writableEnded));res.write(Buffer.from([0]));started();return;
    }
    if(req.url.endsWith('/'+scene.buffers.colors)) {
      await positionsStarted;res.end(Buffer.alloc(scene.pointCount*12+1));return;
    }
    try{res.end(await readFile(new URL('../public'+req.url,import.meta.url)));}
    catch{res.writeHead(404);res.end();}
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  t.after(()=>new Promise(resolve=>{server.closeAllConnections();server.close(resolve);}));
  const base=`http://127.0.0.1:${server.address().port}`;
  await assert.rejects(loadScene((path,options)=>fetch(base+path,options),scene.id),/Artifact exceeds declared byte length/);
  let timer;
  try{assert.equal(await Promise.race([positionsClosed,new Promise((_,reject)=>timer=setTimeout(()=>reject(new Error('Stalled HTTP response was not closed.')),2000))]),true);}
  finally{clearTimeout(timer);}
});
