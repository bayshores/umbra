// The analytic mapped-ground fixture (never a runtime asset), shared by tests/av2-ground.test.mjs and
// tests/car-shape.test.mjs. analytic() builds the scene, ground.json and buffers in memory; f.bytes may gain more files
// (e.g. suggestions.json) before loading, and f.fetch serves them.
import {createHash} from 'node:crypto';
import {deflateSync,crc32} from 'node:zlib';
import {loadScene} from '../src/scene.mjs';

export const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
// Camera at the origin looking along -Z (fx = fy = 20, 8x10 pixels); flat mapped ground at world Y -1.4375 except a
// raised cell (step) and a no-value cell; one drivable rectangle |x| <= 3, -30 <= z <= 0; one annotated box at z -12.
// Ground schema 3 (placement protocol v3.3): placement refused beyond 15 m (horizontal), along-ray height band 0.075 m, and
// no car shape (no suggestion names an annotated vehicle while loading).
function png(width,height) {
  const chunk=(type,data)=>{
    const body=Buffer.concat([Buffer.from(type),data]),size=Buffer.alloc(4),crc=Buffer.alloc(4);
    size.writeUInt32BE(data.length);crc.writeUInt32BE(crc32(body));return Buffer.concat([size,body,crc]);
  };
  const header=Buffer.alloc(13);header.writeUInt32BE(width,0);header.writeUInt32BE(height,4);header[8]=8;header[9]=2;
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',header),chunk('IDAT',deflateSync(Buffer.alloc((width*3+1)*height))),chunk('IEND',Buffer.alloc(0))]);
}
export const FLAT=8.5625,COLUMNS=25,ROWS=50;// cityZ; world Y = cityZ - 10
export function analytic({scene:mutateScene=()=>{},ground:mutateGround=()=>{},terrain:mutateTerrain=()=>{}}={}) {
  const image=png(8,10),c={eye:[0,0,0],forward:[0,0,-1],up:[0,1,0],fx:20,fy:20,cx:2,cy:2,width:8,height:10,near:.01,far:100};
  const xyz=[[0,1.4375,10],[1,1.4375,20],[0,0,-10],[NaN,NaN,NaN]],
    uv=xyz.map(([x,y,z])=>z>0?[c.fx*x/z+c.cx,c.fy*y/z+c.cy]:[NaN,NaN]),
    eligible=uv.map(([u,v])=>Number(u>=0&&u<=7&&v>=0&&v<=9)),world=xyz.map(([x,y,z])=>[x,-y,-z]),draw=world.filter((_,i)=>eligible[i]),
    arrays={positions:new Float32Array(draw.flat()),colors:new Float32Array(draw.flat().map(()=>.5)),triangles:new Uint32Array(),
      sourceIndices:new Uint32Array(xyz.map((_,i)=>i)),referencePositions:new Float64Array(world.flat()),referenceCameraPositions:new Float64Array(xyz.flat()),
      referenceUV:new Float64Array(uv.flat()),referenceEligible:new Uint8Array(eligible)};
  const scene={id:'mapped-fixture',units:'meters',pointCount:draw.length,triangleCount:0,camera:c,
    transform:{worldFromCamera:[[1,0,0],[0,-1,0],[0,0,-1]],worldFromCity:[[1,0,0],[0,0,1],[0,-1,0]],cityOrigin:[101,201,10],cameraOrigin:[0,0,0],originRole:'leftCamera'},
    bounds:{min:draw[0].map((_,k)=>Math.min(...draw.map(p=>Math.fround(p[k])))),max:draw[0].map((_,k)=>Math.max(...draw.map(p=>Math.fround(p[k]))))},
    processing:{},buffers:{},artifactHashes:{},recordedImage:{file:'recorded.png',mimeType:'image/png',bytes:image.length,sha256:sha(image)},
    measurements:{kind:'indexedSparse',evidence:'geometricallyProjected',groundPolicy:'mappedGround',binaryEncoding:'littleEndian',pixelCellPolicy:'nearestIntegerCenterRejectCollisions',
      count:xyz.length,provenance:{preparedManifestSha256:'a'.repeat(64),referencesSha256:'b'.repeat(64),logId:'81d2b40a-c579-3e9c-b520-bee26cda947d',
        cameraTimestampNs:'123',lidarTimestampNs:'124',crossingId:'5'},buffers:{},artifactHashes:{}}};
  const heights=new Float32Array(COLUMNS*ROWS).fill(FLAT);
  heights[10*COLUMNS+10]=9;// raised cell: x in [0,1), z in (-6,-5], world Y -1
  heights[19*COLUMNS+8]=NaN;// no value: x in [-2,-1), z in (-15,-14]
  mutateTerrain(heights);
  const terrain=Buffer.from(heights.buffer),finiteCells=heights.reduce((n,h)=>n+Number.isFinite(h),0);
  const ground={schemaVersion:3,carShape:null,kind:'av2MappedGround',sceneId:scene.id,baseSceneSha256:'a'.repeat(64),
    protocol:{path:'../protocol-v3.1.json',sha256:'b'.repeat(64),bytes:1,readOnlyExternal:true},
    sources:Object.fromEntries(['preparedManifest','references','assembledScene','map','groundRaster','groundTransform','poses','annotations']
      .map((key,i)=>[key,{path:`../${key}.bin`,sha256:key==='assembledScene'?'a'.repeat(64):String(i).repeat(64),bytes:1,readOnlyExternal:true}])),
    frame:{cityOrigin:[101,201,10],worldFromCity:[[1,0,0],[0,0,1],[0,-1,0]],convention:'analytic'},
    roadToleranceMeters:.08,supportHeightAgreementMeters:.08,supportRadiusMeters:.5,maxRangeMeters:20,boundaryEpsilonMeters:.01,placementRangeLimitMeters:15,
    alongRayUncertainty:{rule:'heightBandMeters / sin(depression)',heightBandMeters:.075,method:'fitThisDevelopmentScene',fit:{count:0},coverage:{}},
    terrain:{file:'terrain.bin',bytes:terrain.length,sha256:sha(terrain),dtype:'float32',byteOrder:'little',layout:'row-major',columns:COLUMNS,rows:ROWS,column0:0,row0:0,
      rasterShape:[ROWS,COLUMNS],sim2:{R:[1,0,0,1],t:[-91,-196],s:1},cellMeters:1,finiteCells,nanCells:heights.length-finiteCells,values:'analytic',samplingRule:'analytic'},
    drivableAreas:[{id:'1',outline:[[-3,0],[3,0],[3,-30],[-3,-30]]}],
    crossing:{id:'5',edge1:[[-3,-1.4375,-14.375],[3,-1.4375,-14.375]],edge2:[[-3,-1.4375,-17],[3,-1.4375,-17]]},
    egoLane:{id:'7',laneType:'VEHICLE',leftMarkType:'SOLID_WHITE',rightMarkType:'SOLID_WHITE',leftBoundary:[[-1.5,-1.4375,0],[-1.5,-1.4375,-30]],rightBoundary:[[1.5,-1.4375,0],[1.5,-1.4375,-30]]},
    cuboids:{available:true,frame:'analytic',timestampNs:'124',items:[{category:'REGULAR_VEHICLE',center:[0,-.75,-12],height:1.5,length:1.5,motion:null,numInteriorPoints:1,
      track:'analytic',width:1.5,worldFromBox:[[1,0,0],[0,1,0],[0,0,1]]}]},
    support:{rows:[0],count:1,rule:'analytic',cameraEligibleRows:2,eligibleBehindCuboid:1,eligibleAgreeing:1,eligibleWithoutTerrainValue:0,cuboidOcclusionApplied:true,preparedLookupMismatches:0},
    processingCodeHashes:{'analytic.py':'d'.repeat(64)},dependencies:{numpy:'analytic'},
    license:{id:'CC-BY-NC-SA-4.0',attribution:'(c) analytic',source:'analytic',changes:'analytic'},limitations:['Analytic fixture only.']};
  mutateGround(ground);
  const bytes=new Map([['recorded.png',image],['terrain.bin',terrain]]);
  for(const [key,array] of Object.entries(arrays)) {
    const target=key.startsWith('reference')||key==='sourceIndices'?scene.measurements:scene,name=key.toLowerCase()+'.bin',buffer=Buffer.from(array.buffer,array.byteOffset,array.byteLength);
    target.buffers[key]=name;target.artifactHashes[key]=sha(buffer);bytes.set(name,buffer);
  }
  const groundBytes=Buffer.from(JSON.stringify(ground));bytes.set('ground.json',groundBytes);
  scene.measurements.ground={file:'ground.json',bytes:groundBytes.length,sha256:sha(groundBytes)};
  mutateScene(scene);
  const pin={file:'ground.json',sha256:sha(groundBytes)};
  return {scene,ground,bytes,pin,fetch:async path=>{
    const name=path.split('/').at(-1);
    return name==='scene.json'?new Response(JSON.stringify(scene)):bytes.has(name)?new Response(bytes.get(name)):new Response('',{status:404});
  }};
}
// The catalog pin defaults to the fixture's own; passing undefined explicitly means the catalog entry has none.
export const loadAnalytic=(f,...pin)=>loadScene(f.fetch,f.scene.id,sha(JSON.stringify(f.scene)),undefined,undefined,pin.length?pin[0]:f.pin);
