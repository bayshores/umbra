import assert from 'node:assert/strict';
import test from 'node:test';
import { runAssistant,AssistantError } from '../src/nemotron-client.mjs';

// Mocked transport only: no request leaves the process.
function server(replies) {
  const requests=[];
  // Called unbound, like window.fetch must be (a bound receiver would throw Illegal invocation in browsers).
  function fetch(url,init) {
    assert.equal(this,undefined);
    requests.push({url,init,body:JSON.parse(init.body)});
    const next=replies.shift();
    if(next instanceof Error)return Promise.reject(next);
    return Promise.resolve(new Response(typeof next.body==='string'?next.body:JSON.stringify(next.body),{status:next.status??200,headers:{'Content-Type':'application/json'}}));
  }
  return {fetch,requests};
}
const reply=(content,tool_calls=[])=>({body:{message:{content,tool_calls},finish_reason:tool_calls.length?'tool_calls':'stop',model:'nvidia/Nemotron-3_5-Lightning',
  usage:{prompt_tokens:100,completion_tokens:20},costUsd:.00001,budget:{spentUsd:.00001,limitUsd:2}}});
const call=(id,name='declare_scenario',args='{"vehicle_label":"SUV"}')=>({id,name,arguments:args});
const context={scene:'s',current:{state:'blocked'}};

test('a tool round executes locally, then the final answer returns with trace and usage',async()=>{
  const {fetch,requests}=server([reply('',[call('c1')]),reply('Blocked at 3.05 m.')]),executed=[];
  const result=await runAssistant({mode:'ask',question:'SUV 10 ft back',context},(name,args)=>{executed.push([name,args]);return {ok:true,proposed:{state:'blocked'}};},{fetch});
  assert.deepEqual(executed,[['declare_scenario',{vehicle_label:'SUV'}]]);
  assert.deepEqual(result,{answer:'Blocked at 3.05 m.',trace:[{name:'declare_scenario',args:{vehicle_label:'SUV'},result:{ok:true,proposed:{state:'blocked'}}}],
    usage:{calls:2,promptTokens:200,completionTokens:40,costUsd:.00002},model:'nvidia/Nemotron-3_5-Lightning'});
  assert.equal(requests.length,2);
  for(const {url,init} of requests){assert.equal(url,'/api/nemotron');assert.equal(init.method,'POST');assert.equal(init.headers['Content-Type'],'application/json');}
  assert.deepEqual(requests[0].body,{mode:'ask',question:'SUV 10 ft back',context,turns:[]});
  assert.deepEqual(requests[1].body.turns,[{role:'assistant',tool_calls:[call('c1')]},{role:'tool',tool_call_id:'c1',content:'{"ok":true,"proposed":{"state":"blocked"}}'}]);
});

test('check mode sends no question and invalid input never reaches the server',async()=>{
  const {fetch,requests}=server([reply('Summary.')]);
  assert.equal((await runAssistant({mode:'check',question:'ignored',context},()=>({}),{fetch})).answer,'Summary.');
  assert.deepEqual(Object.keys(requests[0].body),['mode','context','turns']);
  for(const bad of [{mode:'ask',question:'',context},{mode:'ask',question:'x'.repeat(501),context},{mode:'chat',context}])
    await assert.rejects(runAssistant(bad,()=>({}),{fetch}),e=>e instanceof AssistantError&&e.code==='invalid_request');
  assert.equal(requests.length,1);
});

test('invalid JSON arguments become a tool error without executing the tool',async()=>{
  const {fetch,requests}=server([reply('',[call('c1','sweep_setbacks','{"vehicles":[')]),reply('Change the vehicle list.')]);
  let executed=0;
  const result=await runAssistant({mode:'check',context},()=>{executed++;return {};},{fetch});
  assert.equal(executed,0);
  assert.deepEqual(result.trace,[{name:'sweep_setbacks',args:null,result:{error:'arguments were not valid JSON'}}]);
  assert.deepEqual(requests[1].body.turns[1],{role:'tool',tool_call_id:'c1',content:'{"error":"arguments were not valid JSON"}'});
});

test('tool exceptions and oversized results are returned to the model as errors',async()=>{
  const {fetch,requests}=server([reply('',[call('a'),call('b')]),reply('Done.')]);
  const results=[()=>{throw new Error('Vehicle width must be a positive number.');},()=>({rows:'x'.repeat(7000)})];
  await runAssistant({mode:'ask',question:'q',context},()=>results.shift()(),{fetch});
  assert.deepEqual(requests[1].body.turns.slice(1).map(t=>JSON.parse(t.content)),[{error:'Vehicle width must be a positive number.'},{error:'Tool result was missing or larger than 6 KiB.'}]);
});

test('the loop stops after the maximum tool rounds without executing further calls',async()=>{
  const {fetch,requests}=server(Array.from({length:6},(_,i)=>reply('',[call('c'+i)])));
  let executed=0;
  const result=await runAssistant({mode:'ask',question:'q',context},()=>{executed++;return {ok:true};},{fetch});
  assert.equal(result.error,'Assistant did not finish');assert.equal(result.answer,undefined);
  assert.equal(requests.length,4);assert.equal(executed,3);assert.equal(result.usage.calls,4);
  assert.ok(requests.every(r=>r.body.turns.length<=6));
  const once=server(Array.from({length:3},(_,i)=>reply('',[call('c'+i)])));
  assert.equal((await runAssistant({mode:'ask',question:'q',context},()=>({}),{fetch:once.fetch,maxRounds:1})).error,'Assistant did not finish');
  assert.equal(once.requests.length,2);
  // Two calls per round reach the six-turn request limit after two rounds.
  const wide=server(Array.from({length:4},(_,i)=>reply('',[call('a'+i),call('b'+i)])));
  assert.equal((await runAssistant({mode:'ask',question:'q',context},()=>({}),{fetch:wide.fetch})).error,'Assistant did not finish');
  assert.equal(wide.requests.length,3);assert.ok(wide.requests.every(r=>r.body.turns.length<=6));
});

test('server and network errors map to assistant errors with exactly one request',async()=>{
  for(const [status,code] of [[503,'assistant_unavailable'],[503,'budget_exhausted'],[503,'budget_guard_tripped'],[429,'busy'],[504,'upstream_timeout'],[502,'upstream_error']]) {
    const {fetch,requests}=server([{status,body:{error:code,message:'Short text.'}},reply('never sent')]);
    await assert.rejects(runAssistant({mode:'check',context},()=>({}),{fetch}),e=>e instanceof AssistantError&&e.code===code&&e.message.length>0&&!/Short text/.test(e.message));
    assert.equal(requests.length,1);
  }
  const unknown=server([{status:400,body:{error:'bad_request',message:'Unexpected field model.'}}]);
  await assert.rejects(runAssistant({mode:'check',context},()=>({}),{fetch:unknown.fetch}),{code:'bad_request',message:'Unexpected field model.'});
  const html=server([{status:500,body:'<html>'}]);
  await assert.rejects(runAssistant({mode:'check',context},()=>({}),{fetch:html.fetch}),{code:'server_error',message:'The assistant request failed (500).'});
  const offline=server([new TypeError('fetch failed'),reply('never sent')]);
  await assert.rejects(runAssistant({mode:'check',context},()=>({}),{fetch:offline.fetch}),{code:'network'});
  assert.equal(offline.requests.length,1);
  const controller=new AbortController();controller.abort();
  const aborted=server([new DOMException('Aborted','AbortError')]);
  await assert.rejects(runAssistant({mode:'check',context},()=>({}),{fetch:aborted.fetch,signal:controller.signal}),{code:'aborted'});
  for(const body of [{message:{content:'x'}},{message:{content:1,tool_calls:[]}},{message:{content:'',tool_calls:[{id:'a',name:'x'}]}},'not json']) {
    const bad=server([{body}]);
    await assert.rejects(runAssistant({mode:'check',context},()=>({}),{fetch:bad.fetch}),{code:'bad_response'});
  }
});

test('a failure after a tool round is not retried',async()=>{
  const {fetch,requests}=server([reply('',[call('c1')]),{status:504,body:{error:'upstream_timeout',message:'Timed out.'}},reply('never sent')]);
  let executed=0;
  await assert.rejects(runAssistant({mode:'ask',question:'q',context},()=>{executed++;return {};},{fetch}),{code:'upstream_timeout'});
  assert.equal(requests.length,2);assert.equal(executed,1);
});
