// Browser loop for /api/nemotron: one request at a time, explicit calls only, no retries or timers.
const ENDPOINT='/api/nemotron',MAX_TURNS=6,MAX_TOOL_BYTES=6*1024;
const MESSAGES={
  assistant_unavailable:'Nemotron is not configured on this server.',
  budget_exhausted:'The Nemotron spending cap has been reached.',
  budget_guard_tripped:'Nemotron was stopped by its spending guard.',
  busy:'Nemotron is answering another request. Try again shortly.',
  upstream_timeout:'Nemotron did not answer within 30 seconds.',
  upstream_error:'Nemotron returned an error.',
  network:'The assistant server could not be reached.',
  aborted:'The assistant request was cancelled.',
  bad_response:'The assistant server returned an unreadable reply.',
  invalid_request:'The assistant request was not valid.',
};
export class AssistantError extends Error {
  constructor(code,message=MESSAGES[code]??'The assistant request failed.') {super(message);this.name='AssistantError';this.code=code;}
}
const text=v=>typeof v==='string';
const count=n=>Number.isFinite(n)?n:0;
const validReply=r=>r!==null&&typeof r==='object'&&text(r.message?.content)&&Array.isArray(r.message.tool_calls)&&
  r.message.tool_calls.every(c=>c!==null&&typeof c==='object'&&text(c.id)&&text(c.name)&&text(c.arguments));

async function post(send,body,signal) {
  let response,data=null;
  try{response=await send(ENDPOINT,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal});}
  catch{throw new AssistantError(signal?.aborted?'aborted':'network');}
  try{data=await response.json();}catch{}
  if(signal?.aborted)throw new AssistantError('aborted');
  if(!response.ok) {
    const code=text(data?.error)&&/^[a-z_]{1,40}$/.test(data.error)?data.error:'server_error';
    throw new AssistantError(code,MESSAGES[code]??(text(data?.message)&&data.message.length<=200?data.message:`The assistant request failed (${response.status}).`));
  }
  if(!validReply(data))throw new AssistantError('bad_response');
  return data;
}

// executeTool(name,args) is synchronous and returns a JSON-serializable result; its errors go back to the model.
export async function runAssistant({mode,question,context},executeTool,{fetch:send=globalThis.fetch,signal,maxRounds=3}={}) {
  if(mode!=='ask'&&mode!=='check')throw new AssistantError('invalid_request','Assistant mode must be ask or check.');
  if(mode==='ask'&&!(text(question)&&question.trim()&&question.length<=500))throw new AssistantError('invalid_request','Enter a question of 1 to 500 characters.');
  const turns=[],trace=[],usage={calls:0,promptTokens:0,completionTokens:0,costUsd:0};
  let model=null;
  for(let round=0;;round++) {
    const reply=await post(send,mode==='ask'?{mode,question,context,turns}:{mode,context,turns},signal);
    usage.calls++;usage.promptTokens+=count(reply.usage?.prompt_tokens);usage.completionTokens+=count(reply.usage?.completion_tokens);usage.costUsd+=count(reply.costUsd);
    if(text(reply.model))model=reply.model;
    const calls=reply.message.tool_calls;
    if(!calls.length)return {answer:reply.message.content,trace,usage,model};
    if(round>=maxRounds||turns.length+1+calls.length>MAX_TURNS)return {error:'Assistant did not finish',trace,usage,model};
    turns.push({role:'assistant',tool_calls:calls.map(({id,name,arguments:a})=>({id,name,arguments:a}))});
    for(const call of calls) {
      let args=null,result;
      try{args=JSON.parse(call.arguments);}catch{result={error:'arguments were not valid JSON'};}
      if(!result)try{result=executeTool(call.name,args);}catch(e){result={error:e?.message||'Tool failed.'};}
      let content;try{content=JSON.stringify(result);}catch{}
      if(!text(content)||new TextEncoder().encode(content).length>MAX_TOOL_BYTES)content=JSON.stringify({error:'Tool result was missing or larger than 6 KiB.'});
      trace.push({name:call.name,args,result});
      turns.push({role:'tool',tool_call_id:call.id,content});
    }
  }
}
