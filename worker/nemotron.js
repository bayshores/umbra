// Server-owned Nemotron proxy (Nebius Token Factory). Fetch API only: shared by the Vite middleware and the Sites worker.
// The model only declares what-if values and words results; Sightline's geometry stays the only source of blocked/unknown.
export const MODEL='nvidia/Nemotron-3_5-Lightning';
export const MAX_COMPLETION_TOKENS=600;
export const ENDPOINT='https://api.tokenfactory.nebius.com/v1/chat/completions';
export const MAX_BODY_BYTES=16384;
export const TIMEOUT_MS=30000;
const IN_USD=0.06e-6,OUT_USD=0.24e-6,PROMPT_SLACK=4096,MAX_RESPONSE_BYTES=100000;
const enc=new TextEncoder(),utf8=new TextDecoder('utf-8',{fatal:true}),bytes=s=>enc.encode(s).length;
const isObject=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const exact=(o,keys)=>isObject(o)&&Object.keys(o).length===keys.length&&keys.every(k=>Object.hasOwn(o,k));
const validId=id=>typeof id==='string'&&/^[\x21-\x7e]{1,128}$/.test(id);

const num=(minimum,maximum,description)=>({type:'number',minimum,maximum,description});
const label=description=>({type:'string',minLength:1,maxLength:40,description:description+'; plain name, letters only, no numbers or results'});
const fn=(name,description,properties,required)=>({type:'function',function:{name,description,parameters:{type:'object',properties,required,additionalProperties:false}}});
const width=num(0.5,3.5,'Vehicle width in meters'),height=num(0.5,4.5,'Vehicle height in meters'),length=num(1,20,'Vehicle length in meters');
export const TOOLS={
  declare_scenario:fn('declare_scenario','Declare the what-if vehicle box, the person target height and, only when the user gives one, the distance from the crossing to the vehicle front. Sightline applies the values and computes the result.',{
    vehicle_label:label('Short vehicle type, e.g. box truck'),vehicle_width_m:width,vehicle_height_m:height,vehicle_length_m:length,
    target_label:label('Short person type, e.g. child'),target_height_m:num(0.3,2.5,'Height in meters of the tested point on the person, e.g. head height'),
    setback_value:{type:'number',exclusiveMinimum:0,description:'Distance from the crossing to the vehicle front, as the user gave it'},
    setback_unit:{type:'string',enum:['m','ft'],description:'Unit of setback_value'},
  },['vehicle_label','vehicle_width_m','vehicle_height_m','vehicle_length_m','target_label','target_height_m']),
  sweep_setbacks:fn('sweep_setbacks','Test each vehicle at a range of distances from the crossing. Sightline computes where the vehicle blocks the driver-eye-to-person line.',{
    vehicles:{type:'array',minItems:1,maxItems:4,items:{type:'object',properties:{label:label('Short vehicle type'),width_m:width,height_m:height,length_m:length},required:['label','width_m','height_m','length_m'],additionalProperties:false}},
  },['vehicles']),
};
export const MODE_TOOLS={ask:['declare_scenario','sweep_setbacks'],check:['sweep_setbacks']};

const CORE=`You set up what-if scenarios for Sightline. Sightline compares, on a real measured street, whether a declared vehicle (a box) crosses the straight line from the recorded driver eye to one point at a person's declared height. The crossing, vehicle front and person are fixed on the scene's measured or mapped road (by the user or labeled suggestions).
Use the tools to declare values. Sightline computes every result; never compute, estimate or guess one yourself.
- For vehicle or person types the user names, use typical real-world dimensions for that type (never copy the current size) and say they are typical assumptions.
- If the user speaks in feet, give distances in feet (setback_unit "ft"); otherwise use meters.
- After tool results arrive, answer in at most two short sentences (under 240 characters) using only numbers that appear in the tool results.
- Describe results only as "blocked" or "not blocked by the vehicle (rest unknown)". Unknown means the vehicle misses the line but the rest of the scene is not established. Overlap means the box would contain the person or the driver eye, so that distance is not tested.
- Never say clear, visible, safe, legal, compliant, or that anyone can see anything.
- If a tool returns an error, briefly say what to change.`;
export const SYSTEM_PROMPTS={
  ask:`${CORE}
Mode: scenario setup.
- Call declare_scenario for the scenario the user describes. For a vehicle or person the user does not mention, reuse the facts' values and labels. Include setback_value and setback_unit only when the user gives a distance from the crossing.
- Call sweep_setbacks only if the user asks how far back the vehicle must be or asks about distances. Labels are plain names like "box truck".
- If the request is off-topic, say in one sentence that you only set up Sightline scenarios; call no tool.`,
  check:`${CORE}
Mode: distance check.
- Call sweep_setbacks exactly once with the facts' declared vehicle labeled "your vehicle" plus up to three common vehicle types (e.g. compact car, SUV, delivery van) with typical dimensions. Use plain-name labels.
- Then write at most two short sentences: state each vehicle's "short" phrase exactly; vehicles sharing the same phrase may be grouped. Do not mention the 15 ft or 20 ft marks or the Vehicle Code; the app shows them itself. Do not call a tool again.`,
};
const CHECK_INSTRUCTION='Run the distance check for this scenario.';

const HEADERS={'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'};
const json=(status,body,headers)=>new Response(JSON.stringify(body),{status,headers:{...HEADERS,...headers}});
const fail=(status,error,message,fields,headers)=>json(status,{error,message,...fields},headers);

async function readCapped(stream,limit) {
  if(!stream)return new Uint8Array();
  const reader=stream.getReader(),chunks=[];let size=0;
  for(;;) {
    const {done,value}=await reader.read();
    if(done)break;
    size+=value.byteLength;
    if(size>limit){reader.cancel().catch(()=>{});return null;}
    chunks.push(value);
  }
  const out=new Uint8Array(size);let at=0;
  for(const c of chunks){out.set(c,at);at+=c.byteLength;}
  return out;
}

// Returns a plain-text problem for the client body contract, or null when valid.
function invalid(body) {
  if(!isObject(body))return 'Body must be a JSON object.';
  if(Object.keys(body).some(k=>!['mode','question','context','turns'].includes(k)))return 'Body has an unexpected field.';
  const {mode,question,context,turns=[]}=body;
  if(!Object.hasOwn(MODE_TOOLS,mode))return 'mode must be "ask" or "check".';
  if(mode==='ask'?typeof question!=='string'||question.length<1||question.length>500:question!==undefined)return 'question must be 1-500 characters, for ask only.';
  if(!isObject(context)||bytes(JSON.stringify(context))>4096)return 'context must be a JSON object of at most 4 KiB.';
  if(!Array.isArray(turns)||turns.length>6)return 'turns must be a list of at most 6 entries.';
  let open=null;// tool_call ids of the preceding assistant turn still waiting for a tool result
  for(const t of turns) {
    if(t?.role==='assistant') {
      if(open?.size)return 'Every tool call needs a tool result before the next assistant turn.';
      if(!exact(t,['role','tool_calls'])||!Array.isArray(t.tool_calls)||!t.tool_calls.length)return 'Assistant turns carry only tool_calls.';
      open=new Set();
      for(const c of t.tool_calls) {
        if(!exact(c,['id','name','arguments'])||!validId(c.id)||open.has(c.id))return 'Each tool call needs a unique id, a name and arguments.';
        if(!MODE_TOOLS[mode].includes(c.name))return 'Tool is not allowed in this mode.';
        if(typeof c.arguments!=='string'||bytes(c.arguments)>2048)return 'Tool arguments must be a string of at most 2 KiB.';
        open.add(c.id);
      }
    } else if(t?.role==='tool') {
      if(!exact(t,['role','tool_call_id','content'])||!open?.delete(t.tool_call_id))return 'Each tool result must answer a call from the preceding assistant turn.';
      if(typeof t.content!=='string'||bytes(t.content)>6144)return 'Tool result content must be a string of at most 6 KiB.';
    } else return 'Turn role must be assistant or tool.';
  }
  return open?.size?'Every tool call needs a tool result.':null;
}

function upstreamBody({mode,question,context,turns=[]}) {
  const facts=`\nScenario facts computed by Sightline (JSON): ${JSON.stringify(context)}`;
  return {model:MODEL,temperature:0,n:1,stream:false,service_tier:'default',
    max_completion_tokens:MAX_COMPLETION_TOKENS,tool_choice:'auto',tools:MODE_TOOLS[mode].map(n=>TOOLS[n]),
    chat_template_kwargs:{enable_thinking:false},
    messages:[{role:'system',content:SYSTEM_PROMPTS[mode]},{role:'user',content:(mode==='ask'?`Question: ${question}`:CHECK_INSTRUCTION)+facts},
      ...turns.map(t=>t.role==='assistant'
        ?{role:'assistant',content:'',tool_calls:t.tool_calls.map(c=>({id:c.id,type:'function',function:{name:c.name,arguments:c.arguments}}))}
        :{role:'tool',tool_call_id:t.tool_call_id,content:t.content})]};
}

// Budget {reserve(usd)->token|null, settle(token,usd), snapshot()}. Pure and per instance; an unsettled reservation counts in full.
export function createMemoryBudget(limitUsd) {
  const limit=Number.isFinite(limitUsd)&&limitUsd>0?limitUsd:0,held=new Map();
  let settled=0;
  const spent=()=>[...held.values()].reduce((s,v)=>s+v,settled);
  return {
    reserve(usd) {
      if(!(Number.isFinite(usd)&&usd>0)||spent()+usd>limit)return null;
      const token=crypto.randomUUID();
      held.set(token,usd);
      return token;
    },
    settle(token,usd) {
      if(!held.has(token)||!(Number.isFinite(usd)&&usd>=0))return;
      held.delete(token);settled+=usd;
    },
    snapshot:()=>({spentUsd:spent(),limitUsd:limit}),
  };
}

// At most one upstream call in flight, and a tripped guard stays off until restart. handleNemotron keeps this state on globalThis:
// a Vite config reload evaluates this module again in the same process and must not reset it.
const freshState=()=>({busy:false,tripped:false});
export function createNemotronHandler(state=freshState()) {
  return async function handleNemotron(request,env={},deps={}) {
    if(request.method!=='POST')return fail(405,'method_not_allowed','Use POST.',{},{Allow:'POST'});
    if(!(request.headers.get('content-type')??'').toLowerCase().startsWith('application/json'))return fail(415,'unsupported_media_type','Send a JSON body.');
    const site=request.headers.get('sec-fetch-site');
    if(request.headers.get('origin')!==new URL(request.url).origin||(site!==null&&site!=='same-origin'))return fail(403,'forbidden','Only same-origin requests are accepted.');
    const raw=Number(request.headers.get('content-length'))>MAX_BODY_BYTES?null:await readCapped(request.body,MAX_BODY_BYTES);
    if(!raw)return fail(413,'too_large','Request body is over 16 KiB.');
    let body;
    try{body=JSON.parse(utf8.decode(raw));}catch{return fail(400,'bad_request','Body must be valid JSON.');}
    const problem=invalid(body);
    if(problem)return fail(400,'bad_request',problem);
    const key=env.TOKEN_FACTORY_API_KEY;
    // An unset or zero NEMOTRON_BUDGET_USD means off, not used up.
    if(env.NEMOTRON_ENABLED!=='1'||typeof key!=='string'||!key||typeof deps.fetch!=='function'||!(deps.budget?.snapshot().limitUsd>0))
      return fail(503,'assistant_unavailable','The Nemotron assistant is not configured.');
    if(state.tripped)return fail(503,'budget_guard_tripped','Nemotron usage exceeded its guard; the assistant is off until restart.');
    if(state.busy)return fail(429,'busy','Nemotron is already answering; try again when it finishes.');
    const upstream=JSON.stringify(upstreamBody(body)),size=bytes(upstream),worst=(size+PROMPT_SLACK)*IN_USD+MAX_COMPLETION_TOKENS*OUT_USD;
    let token;
    try{token=deps.budget.reserve(worst);}catch{return fail(503,'assistant_unavailable','The Nemotron spending log could not be checked.');}
    if(token==null)return fail(503,'budget_exhausted','The Nemotron budget is used up.');
    state.busy=true;
    try {
      let response,raw=null;
      try {
        response=await deps.fetch(ENDPOINT,{method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${key}`},
          body:upstream,signal:AbortSignal.timeout(TIMEOUT_MS),redirect:'manual'});
        if(response.status===200)raw=await readCapped(response.body,MAX_RESPONSE_BYTES);
        else response.body?.cancel().catch(()=>{});
      } catch(error) {
        deps.budget.settle(token,worst);// timeout or network failure: keep the full worst case as spent
        return error?.name==='TimeoutError'?fail(504,'upstream_timeout','Nemotron did not answer within 30 seconds.')
          :fail(502,'upstream_error','Could not reach Nemotron.',{status:0});
      }
      let data=null;
      try{data=raw&&JSON.parse(utf8.decode(raw));}catch{}
      const choice=data?.choices?.[0],message=choice?.message,usage=data?.usage;
      const counted=['prompt_tokens','completion_tokens'].every(k=>Number.isInteger(usage?.[k])&&usage[k]>=0);
      if(response.status!==200||!isObject(message)||!counted) {
        deps.budget.settle(token,worst);
        return fail(502,'upstream_error','Nemotron returned an unusable response.',{status:response.status});
      }
      const costUsd=usage.prompt_tokens*IN_USD+usage.completion_tokens*OUT_USD;
      deps.budget.settle(token,costUsd);
      if(usage.prompt_tokens>size+PROMPT_SLACK||usage.completion_tokens>MAX_COMPLETION_TOKENS) {
        state.tripped=true;
        return fail(503,'budget_guard_tripped','Nemotron usage exceeded its guard; the assistant is off until restart.');
      }
      const allowed=MODE_TOOLS[body.mode],seen=new Set();// a repeated id keeps its first call, as the next request requires unique ids
      const tool_calls=(Array.isArray(message.tool_calls)?message.tool_calls:[])
        .filter(c=>validId(c?.id)&&allowed.includes(c.function?.name)&&typeof c.function.arguments==='string'&&bytes(c.function.arguments)<=2048)
        .filter(c=>!seen.has(c.id)&&seen.add(c.id))
        .map(c=>({id:c.id,name:c.function.name,arguments:c.function.arguments}));
      return json(200,{message:{content:typeof message.content==='string'?message.content:'',tool_calls},
        finish_reason:typeof choice.finish_reason==='string'?choice.finish_reason:null,
        model:typeof data.model==='string'?data.model.slice(0,100):MODEL,
        usage:{prompt_tokens:usage.prompt_tokens,completion_tokens:usage.completion_tokens},costUsd,budget:deps.budget.snapshot()});
    } finally {state.busy=false;}
  };
}
export const handleNemotron=createNemotronHandler(globalThis[Symbol.for('sightline.nemotron')]??=freshState());
