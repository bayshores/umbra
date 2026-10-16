import { readFile } from 'node:fs/promises';
import { loadScene } from '../src/scene.mjs';
import { placementFromPixels } from '../src/geometry.mjs';
export async function fetchAsset(path) {
  try{return new Response(await readFile(new URL('../public'+path,import.meta.url)));}
  catch(e){if(e.code==='ENOENT')return new Response(null,{status:404});throw e;}
}
export const catalog=await (await fetchAsset('/scenes/index.json')).json();
// The measured-depth fixture these pixels belong to; mapped-ground (AV2) entries may come first in a local catalog.
export const scene=await loadScene(fetchAsset,catalog.find(entry=>!entry.ground).id);
// Test annotations on actual measured pixels. These are never imported by the app.
const pixels={crossing:[589,233],parking:[600,250],target:[621,235]};
const placement=placementFromPixels(scene,pixels);
export const input={pixels,vehicle:{width:1.9,height:2.4,length:5.4},targetHeight:1.3,proposedSetback:placement.baselineSetback,driverDistance:placement.captureDistance};
