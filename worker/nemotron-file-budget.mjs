// Node-only pieces of the local /api/nemotron route (never packaged to Sites): durable spending log, guarded key-file read
// and the Vite middleware itself.
import { closeSync, constants, fstatSync, mkdirSync, openSync, readSync, realpathSync, statSync, unlinkSync, writeSync } from 'node:fs';
import path from 'node:path';
import { MAX_BODY_BYTES, handleNemotron } from './nemotron.js';

const {O_APPEND,O_CREAT,O_EXCL,O_NOFOLLOW,O_RDONLY,O_RDWR,O_WRONLY}=constants;
const privateFile=st=>st.isFile()&&st.uid===process.getuid?.()&&(st.mode&0o777)===0o600;
const LOCK_WAIT_MS=1000,STALE_LOCK_MS=10000,pause=new Int32Array(new SharedArrayBuffer(4));
export const USAGE_LOG_NAME='nemotron-usage.jsonl';

// Appends {utc, kind:'reserve'|'settle', id, usd} JSON lines. Every reserve and settle holds `<log>.lock` and first reads what
// other processes appended, so restarts and a concurrent dev + preview server share one cap. An unsettled reservation counts at
// its full worst case. A malformed or partial log fails closed.
export function createFileBudget(file,limitUsd) {
  mkdirSync(path.dirname(file),{recursive:true,mode:0o700});
  const fd=openSync(file,O_RDWR|O_CREAT|O_APPEND|O_NOFOLLOW,0o600);
  if(!privateFile(fstatSync(fd))){closeSync(fd);throw new Error('Nemotron usage log must be a regular mode 0600 file owned by this user.');}
  const limit=Number.isFinite(limitUsd)&&limitUsd>0?limitUsd:0,lock=`${file}.lock`,amounts=new Map(),own=new Set();
  let offset=0,lines=0;
  const spent=()=>[...amounts.values()].reduce((s,v)=>s+v,0);
  const scan=()=>{
    const size=fstatSync(fd).size,buffer=Buffer.alloc(Math.max(0,size-offset));
    if(size<offset||readSync(fd,buffer,0,buffer.length,offset)!==buffer.length)throw new Error('Nemotron usage log changed unexpectedly.');
    const text=buffer.toString('utf8');
    if(text&&!text.endsWith('\n'))throw new Error(`Nemotron usage log line ${lines+1} is incomplete.`);
    for(const line of text.split('\n').slice(0,-1)) {
      lines++;
      if(!line)continue;
      let entry;
      try{entry=JSON.parse(line);}catch{}
      if(!['reserve','settle'].includes(entry?.kind)||typeof entry.id!=='string'||!(Number.isFinite(entry.usd)&&entry.usd>=0))
        throw new Error(`Nemotron usage log line ${lines} is malformed.`);
      amounts.set(entry.id,entry.usd);
    }
    offset=size;
  };
  const locked=fn=>{
    let lfd;
    for(const start=Date.now();lfd===undefined;) {
      try{lfd=openSync(lock,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0o600);}
      catch(error) {
        if(error.code!=='EEXIST')throw error;
        // The lock covers one read and one append; an old one was left by a process that died holding it.
        try{if(Date.now()-statSync(lock).mtimeMs>STALE_LOCK_MS)unlinkSync(lock);}catch{}
        if(Date.now()-start>LOCK_WAIT_MS)throw new Error(`Nemotron usage log is locked; remove ${lock} if no umbra server is running.`);
        Atomics.wait(pause,0,0,5);
      }
    }
    try{scan();return fn();}finally{closeSync(lfd);try{unlinkSync(lock);}catch{}}
  };
  const append=(kind,id,usd)=>{
    offset+=writeSync(fd,JSON.stringify({utc:new Date().toISOString(),kind,id,usd})+'\n');
    lines++;amounts.set(id,usd);
  };
  try{locked(()=>{});}catch(error){closeSync(fd);throw error;}
  return {
    reserve(usd) {
      if(!(Number.isFinite(usd)&&usd>0&&limit))return null;
      return locked(()=>{
        if(spent()+usd>limit)return null;
        const id=crypto.randomUUID();
        append('reserve',id,usd);own.add(id);
        return id;
      });
    },
    settle(token,usd) {
      if(!own.has(token)||!(Number.isFinite(usd)&&usd>=0))return;
      locked(()=>append('settle',token,usd));own.delete(token);
    },
    snapshot:()=>({spentUsd:spent(),limitUsd:limit}),
  };
}

// Reads the Token Factory key: a *.key regular file, not a symlink, with no other hard links, owned by this user, mode 0600,
// 2-8193 bytes, printable ASCII. Vite matches its *.key deny rule on the requested name, so the name keeps the key unservable
// through any symlinked folder. `servable(path)` says whether the dev server could hand a file to a browser; a key it could
// serve by its given or real path is refused. Throws a reason that never contains the key.
export function readKeyFile(file,servable=()=>false) {
  if(!file)throw new Error('TOKEN_FACTORY_KEY_FILE is not set.');
  if(!/\.key$/i.test(file))throw new Error('Key file name must end in .key.');
  const fd=openSync(file,O_RDONLY|O_NOFOLLOW);
  try {
    const st=fstatSync(fd);
    if(!privateFile(st))throw new Error('Key file must be a regular mode 0600 file owned by this user.');
    if(st.nlink!==1)throw new Error('Key file must not have other hard links.');
    if(servable(path.resolve(file))||servable(realpathSync(file)))throw new Error('The dev server could serve the key file; keep it outside the project or under .local/.');
    const buffer=Buffer.alloc(8194),n=readSync(fd,buffer,0,buffer.length,0);
    if(n<2||n>8193)throw new Error('Key file must be 2-8193 bytes.');
    const key=buffer.toString('latin1',0,n).replace(/\r?\n$/,'');
    if(!/^[\x21-\x7e]+$/.test(key))throw new Error('Key file must hold one line of printable ASCII.');
    return key;
  } finally {closeSync(fd);}
}

// Loopback names and addresses only, whatever Vite's allowedHosts say: another machine can answer for a name such as a .local
// mDNS host and then rebind it to 127.0.0.1, which would make its page same-origin with this route.
const LOOPBACK_HOSTS=['localhost','127.0.0.1','[::1]'];
export const hostAllowed=header=>LOOPBACK_HOSTS.includes(String(header??'').replace(/:\d+$/,'').toLowerCase());
export const loopbackAddress=address=>address==='::1'||/^(::ffff:)?127(\.\d{1,3}){3}$/.test(String(address));

// One file budget per log for the whole process, kept beside the handler's busy/tripped state: a Vite config reload reuses it
// instead of opening the log again.
const budgets=(globalThis[Symbol.for('umbra.nemotron')]??={}).budgets??=new Map();
function sharedFileBudget(file,limitUsd) {
  const id=`${limitUsd}\0${file}`;
  if(!budgets.has(id))budgets.set(id,createFileBudget(file,limitUsd));
  return budgets.get(id);
}

// Collects at most `limit` bytes, then discards the rest so the handler can answer 413.
export const readRequestBody=(req,limit)=>new Promise((resolve,reject)=>{
  const chunks=[];let size=0;
  const finish=()=>{
    req.off('data',onData).off('end',finish).off('error',reject);
    if(!req.readableEnded)req.resume();
    resolve(Buffer.concat(chunks).subarray(0,limit));
  };
  const onData=chunk=>{chunks.push(chunk);size+=chunk.length;if(size>=limit)finish();};
  req.on('data',onData).once('end',finish).once('error',reject);
});

const sendJson=(res,status,body)=>{
  res.writeHead(status,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'});
  res.end(JSON.stringify(body));
};

// Mounted ahead of Vite's own middleware (no Vite host check or CORS), so it checks the Host header and peer address itself.
// The key file and usage log are touched only when a request arrives with NEMOTRON_ENABLED=1. The log defaults to the key file's
// real folder, so every checkout or worktree using that key shares one cap; NEMOTRON_USAGE_LOG must be an absolute path.
export function nemotronMiddleware(root,{env=process.env,servable,fetch=(url,init)=>globalThis.fetch(url,init),handle=handleNemotron}={}) {
  return async(req,res,next)=>{
    if(req.url?.split('?')[0]!=='/api/nemotron')return next();
    try {
      if(!hostAllowed(req.headers.host)||!loopbackAddress(req.socket?.remoteAddress))
        return sendJson(res,403,{error:'forbidden',message:'Only loopback requests are accepted.'});
      const config={NEMOTRON_ENABLED:env.NEMOTRON_ENABLED};let budget;
      if(config.NEMOTRON_ENABLED==='1') {
        try {
          const keyFile=env.TOKEN_FACTORY_KEY_FILE&&path.resolve(root,env.TOKEN_FACTORY_KEY_FILE),log=env.NEMOTRON_USAGE_LOG;
          if(log&&!path.isAbsolute(log))throw new Error('NEMOTRON_USAGE_LOG must be an absolute path.');
          const key=readKeyFile(keyFile,servable);
          budget=sharedFileBudget(log||path.join(path.dirname(realpathSync(keyFile)),USAGE_LOG_NAME),Number(env.NEMOTRON_BUDGET_USD||0));
          config.TOKEN_FACTORY_API_KEY=key;
        } catch(error) {
          console.warn(`Nemotron assistant unavailable: ${error.message}`);
        }
      }
      const headers=new Headers();
      for(const name of ['content-type','origin','sec-fetch-site'])if(typeof req.headers[name]==='string')headers.set(name,req.headers[name]);
      const request=new Request(`${req.socket?.encrypted?'https':'http'}://${req.headers.host}${req.url}`,{method:req.method,headers,
        body:req.method==='POST'?await readRequestBody(req,MAX_BODY_BYTES+1):undefined});
      const response=await handle(request,config,{fetch,budget});
      res.writeHead(response.status,Object.fromEntries(response.headers));
      res.end(Buffer.from(await response.arrayBuffer()));
    } catch {
      if(res.headersSent)res.end();
      else sendJson(res,500,{error:'internal_error',message:'The Nemotron route failed.'});
    }
  };
}
