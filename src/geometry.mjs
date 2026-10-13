const EPS = 1e-9;
const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
const sub = (a,b) => a.map((v,i)=>v-b[i]);
const add = (a,b) => a.map((v,i)=>v+b[i]);
const mul = (a,s) => a.map(v=>v*s);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const finite3 = p => Array.isArray(p)&&p.length===3&&Number.isFinite(p[0])&&Number.isFinite(p[1])&&Number.isFinite(p[2]);
const fmt2=n=>String(Number(n.toFixed(2)));
// A typed value is a finite number or a plain decimal string ("1.88", ".5", "2e0"); hex ("0x2"), signs, spaces and
// empty strings are refused, never read as something else.
const DECIMAL=/^(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/;
const decimal=value=>typeof value==='number'?value:typeof value==='string'&&DECIMAL.test(value)?Number(value):NaN;
function positive(value,label) {
  const n = decimal(value);
  if (!Number.isFinite(n) || n <= 0) throw new Error(`${label} must be a positive number.`);
  return n;
}
// Declared sizes, metres. One set for typed fields and Nemotron declarations (the tool ranges of the Nemotron spec and
// worker/nemotron.js): vehicle width 0.5-3.5, height 0.5-4.5, length 1-20 (a motorcycle or low trailer up to an
// oversize load or articulated bus); tested person height 0.3-2.5. Out-of-range values are refused, never clamped.
// Why 20 m: it holds an articulated bus (about 18 m) and long rigid trucks. A semi tractor-trailer (about 21-23 m
// overall) is refused with the length message, never shortened. The AV2 8c approach from the crossing to the recording
// camera is 17.6 m, so no setback there could hold a longer vehicle anyway. An annotated stationary vehicle outside these
// sizes fails the suggestion load (suggestions.mjs), so a scene never opens straight into a size refusal.
export const SIZE_LIMITS=Object.freeze({width:Object.freeze([.5,3.5]),height:Object.freeze([.5,4.5]),length:Object.freeze([1,20]),targetHeight:Object.freeze([.3,2.5])});
function sized(value,label,[lo,hi]) {
  const n=decimal(value);
  if(!Number.isFinite(n)||n<lo||n>hi)throw new Error(`${label} must be a number from ${lo} to ${hi} m.`);
  return n;
}
const vehicleSizes=vehicle=>Object.fromEntries(['width','height','length'].map(k=>[k,sized(vehicle?.[k],`Vehicle ${k}`,SIZE_LIMITS[k])]));
export function measuredPoint(scene,pixel) {
  const [u,v] = pixel ?? [], c = scene.camera;
  if (!Number.isInteger(u)||!Number.isInteger(v)||u<0||u>=c.width||v<0||v>=c.height) throw new Error('Select a pixel inside the recorded image.');
  if(scene.measurements?.kind==='indexedSparse') {
    const row=scene.pixelRows[v*c.width+u];
    if(row===-1)throw new Error('No projected source measurement at this pixel.');
    if(row===-2)throw new Error('Multiple projected source measurements share this pixel.');
    return Array.from(scene.measurements.referencePositions.subarray(row*3,row*3+3));
  }
  const raw = scene.depth[v*c.width+u];
  if (!raw) throw new Error('No measured depth at this pixel. Choose a supported road point.');
  const d=raw/scene.processing.depthDivisor;
  const p=sub([(u-c.cx)*d/c.fx,(v-c.cy)*d/c.fy,d],scene.transform.cameraOriginGround);
  return scene.transform.worldFromCamera.map(row=>dot(row,p));
}
// Mapped ground (map-first AV2 scenes, groundPolicy 'mappedGround'): a pixel maps to the first hit of its camera ray with
// the loader-verified publisher ground raster inside the mapped drivable areas. Mapped ground is placement support only,
// never observed road or free-space evidence; results stay Blocked or Unknown.
export const isMappedGround=scene=>scene?.measurements?.groundPolicy==='mappedGround';
// The road tolerance lives in ground.json for mapped scenes; indexed scenes never carry processing.roadToleranceMeters.
export const roadTolerance=scene=>isMappedGround(scene)?scene.ground.roadToleranceMeters:scene.processing.roadToleranceMeters;
export const SUPPORT_LABELS=Object.freeze({'observed-nearby':'observed nearby','map-only':'mapped, not observed',unknown:'unknown'});
// The same classes for Nemotron's context: no digits and none of the answer filter's withheld words ("observed", "hidden").
export const SUPPORT_CONTEXT=Object.freeze({'observed-nearby':'with nearby LiDAR returns','map-only':'map only'});
// prepare_av2.ground_sample's affine raster coordinates for a world (x, z).
function terrainImage(g,x,z) {
  const o=g.frame.cityOrigin,{R,t,s}=g.terrain.sim2,cityX=x+o[0],cityY=-z+o[1];
  return [s*((R[0]*cityX+R[1]*cityY)+t[0]),s*((R[2]*cityX+R[3]*cityY)+t[1])];
}
// One raster cell, no interpolation: {cityZ, cell:[column,row]} or null when there is no value.
function terrainSample(scene,x,z) {
  const t=scene.ground.terrain,[ix,iy]=terrainImage(scene.ground,x,z);
  if(!Number.isFinite(ix)||!Number.isFinite(iy)||ix<0||iy<0)return null;
  const column=Math.trunc(ix),row=Math.trunc(iy),localColumn=column-t.column0,localRow=row-t.row0;
  if(column>=t.rasterShape[1]||row>=t.rasterShape[0]||localColumn<0||localColumn>=t.columns||localRow<0||localRow>=t.rows)return null;
  const cityZ=scene.terrainHeights[localRow*t.columns+localColumn];
  return Number.isNaN(cityZ)?null:{cityZ,cell:[column,row]};
}
// Mapped ground height (world Y) under a world (x, z), or null when the raster has no value there.
export function mappedGroundHeight(scene,x,z) {
  const cell=terrainSample(scene,x,z);
  return cell&&cell.cityZ-scene.ground.frame.cityOrigin[2];
}
// Even-odd membership per outline, union over outlines (prepare_av2_ground.inside_polygons).
function insideDrivable(areas,x,z) {
  return areas.some(({outline})=>{
    let inside=false;
    for(let i=0,j=outline.length-1;i<outline.length;j=i++) {
      const [xi,zi]=outline[i],[xj,zj]=outline[j];
      if((zi>z)!==(zj>z)&&x<(xj-xi)*(z-zi)/(zj-zi)+xi)inside=!inside;
    }
    return inside;
  });
}
// Parameters in (0, limit) where start + s*direction (world X/Z) crosses an integer raster coordinate.
function terrainCuts(g,start,direction,limit,cuts) {
  const {R,s}=g.terrain.sim2,a=terrainImage(g,start[0],start[1]),b=[s*(R[0]*direction[0]+R[1]*-direction[1]),s*(R[2]*direction[0]+R[3]*-direction[1])];
  for(let k=0;k<2;k++) {
    if(b[k]===0)continue;
    const end=a[k]+limit*b[k];
    for(let n=Math.floor(Math.min(a[k],end))+1;n<Math.ceil(Math.max(a[k],end));n++){const t=(n-a[k])/b[k];if(t>0&&t<limit)cuts.push(t);}
  }
  return cuts;
}
// Closed segment camera origin -> point against every annotated cuboid (box-local inclusive slab test); null when the
// annotations are unavailable (not evaluated), never false.
function hiddenByCuboids(g,point) {
  if(!g.cuboids.available)return null;
  return g.cuboids.items.some(box=>{
    const M=box.worldFromBox,c=box.center,local=p=>[0,1,2].map(k=>(p[0]-c[0])*M[0][k]+(p[1]-c[1])*M[1][k]+(p[2]-c[2])*M[2][k]);
    const half=[box.length/2,box.width/2,box.height/2];
    return segmentIntersectsBox(local([0,0,0]),local(point),half.map(n=>-n),half);
  });
}
// Support class and hidden flag of a mapped ground point (protocol supportClasses, hidden): observed nearby when a pinned
// support return lies within the support radius (horizontal), else mapped, not observed.
function groundEvidence(scene,point) {
  const support=scene.groundSupportXZ,radius=scene.ground.supportRadiusMeters;
  let observed=false;
  for(let k=0;k<support.length&&!observed;k+=2){const dx=support[k]-point[0],dz=support[k+1]-point[2];observed=Math.sqrt(dx*dx+dz*dz)<=radius;}
  const supportClass=observed?'observed-nearby':'map-only';
  return {supportClass,supportLabel:SUPPORT_LABELS[supportClass],hidden:hiddenByCuboids(scene.ground,point)};
}
// A crop position (protocol v3 pixelRay.pixelCenter): finite [u, v] with 0 <= u <= width-1 and 0 <= v <= height-1;
// integers are pixel centres and fractional values are allowed.
const cropPosition=(c,p)=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite)&&p[0]>=0&&p[0]<=c.width-1&&p[1]>=0&&p[1]<=c.height-1;
// A user position (a click, typed pixel fields, a Nemotron declaration or a suggestion's integer anchor) is a whole pixel
// inside the crop, as before protocol v3 and as protocol v3.1 currentVehicle.declaredReplacement says ("an integer
// pixel"). A fractional position reaches mappedGroundHit only as an untouched suggestion's pixelUV (evaluatedPosition),
// in the suggestion loader and in the evaluation harness. Why (v3.1 review): namesAnnotatedFront takes whole pixels
// only, so a typed fractional front next to the anchor (e.g. [374.5, 191] on AV2 8c) became a declared box at its own
// raster-shadowed ray hit, 1.47 m from the crossing and Blocked, instead of naming the stationary vehicle.
function userPixel(scene,pixel) {
  const c=scene.camera;
  if(!Array.isArray(pixel)||pixel.length!==2||!pixel.every(Number.isInteger)||pixel[0]<0||pixel[0]>=c.width||pixel[1]<0||pixel[1]>=c.height)
    throw new Error('Select a pixel inside the recorded image.');
  return pixel;
}
// The runtime crop position -> mapped ground rule (same as scripts/california/prepare_av2_ground.py pixel_ground_hit, protocol
// v3): first hit of the camera ray with the raster, then the placement range limit, then the drivable test. Throws the
// Unknown reason. alongRayUncertaintyM = the ground layer's fitted height band / sin(depression).
export function mappedGroundHit(scene,position) {
  const c=scene.camera,g=scene.ground;
  if(!cropPosition(c,position))throw new Error('Select a pixel inside the recorded image.');
  const [u,v]=position,q=[(u-c.cx)/c.fx,(v-c.cy)/c.fy,1],raw=scene.transform.worldFromCamera.map(row=>row[0]*q[0]+row[1]*q[1]+row[2]*q[2]);
  const norm=Math.sqrt(raw[0]*raw[0]+raw[1]*raw[1]+raw[2]*raw[2]),d=raw.map(n=>n/norm),max=g.maxRangeMeters;
  if(!(d[1]<0))throw new Error('Pixel ray does not descend toward the ground.');
  const cuts=terrainCuts(g,[0,0],[d[0],d[2]],max,[0,max]).sort((x,y)=>x-y);
  for(let i=1;i<cuts.length;i++) {
    const ta=cuts[i-1],tb=cuts[i];
    if(!(tb>ta))continue;
    const tm=(ta+tb)/2,cell=terrainSample(scene,tm*d[0],tm*d[2]);
    if(!cell)throw new Error('Ray crosses a cell without mapped ground height before reaching the ground.');
    const ground=cell.cityZ-g.frame.cityOrigin[2];
    let t,hitKind;
    if(ta*d[1]<=ground){t=ta;hitKind='step';}
    else if(ground/d[1]<=tb){t=ground/d[1];hitKind='top';}
    else continue;
    if(t<=0)throw new Error('Camera origin is not above the mapped ground.');
    const point=[t*d[0],t*d[1],t*d[2]],limit=g.placementRangeLimitMeters;
    // Sean (October 12): placement beyond 30 m is refused (the along-ray uncertainty there would be useless).
    if(Math.hypot(point[0],point[2])>limit)throw new Error(`Too far to place (over ${limit} m).`);
    if(!insideDrivable(g.drivableAreas,point[0],point[2]))throw new Error('First mapped-ground hit is outside the mapped drivable area.');
    return {point,...groundEvidence(scene,point),alongRayUncertaintyM:g.alongRayUncertainty.heightBandMeters/-d[1],
      rangeM:t,hitKind,terrainCityZ:cell.cityZ,terrainCell:cell.cell};
  }
  throw new Error(`No mapped ground within ${max} m along the pixel ray.`);
}
// How far (pixels, each image axis) the vehicle front may be nudged from the suggested annotated front and still name the
// annotated stationary vehicle (protocol v3 currentVehicle.frontNudgePx; reason: plan M1(d)). A 1-2 px nudge corrects a
// click; it does not name a different vehicle. Near the horizon one pixel row moves a ground point by up to about a metre
// (AV2 8c: about its shown +/-0.99 m along-ray uncertainty), so before this rule a 1 px nudge swapped the real car for a
// declared box or was refused. The suggested pixel is the one containing the exact projection of the averaged front.
export const ANNOTATED_FRONT_NUDGE_PX=2;
export const namesAnnotatedFront=(suggested,pixel)=>Array.isArray(pixel)&&pixel.length===2&&Number.isInteger(pixel[0])&&Number.isInteger(pixel[1])&&
  Array.isArray(suggested)&&Math.abs(pixel[0]-suggested[0])<=ANNOTATED_FRONT_NUDGE_PX&&Math.abs(pixel[1]-suggested[1])<=ANNOTATED_FRONT_NUDGE_PX;
// The suggestion record of an anchor key (crossing, parking = vehicle front, target = person).
const ROLE={crossing:'crossing',parking:'vehicle',target:'person'};
// The position the app evaluates for an anchor (protocol v3 pixelSelection.app): a placed suggestion's exact sub-pixel
// position (pixelUV) while its integer pixel is still the suggested anchor, otherwise the position as given. An unknown
// suggestion (e.g. refused as too far) may still carry pixel and pixelUV; it was never placed, so a click on its pixel is
// evaluated as given. On AV2 8c the integer anchors alone land 0.98 m (crossing) and 0.31 m (person) from their mapped
// points.
export function evaluatedPosition(scene,key,pixel) {
  const e=isMappedGround(scene)?scene.suggestions?.[ROLE[key]]:null;
  return e&&e.status!=='unknown'&&Array.isArray(e.pixelUV)&&Array.isArray(pixel)&&Array.isArray(e.pixel)&&pixel[0]===e.pixel[0]&&pixel[1]===e.pixel[1]
    ?[...e.pixelUV]:pixel;
}
// The annotated stationary vehicle's averaged box from ground.json (suggestions.json names its track): exact annotated
// centre, horizontal heading and dimensions, never re-derived from the road axis or the front pixel. classifySightline
// takes a horizontal axis, so the heading is the box's horizontal forward direction (its sub-degree pitch and roll are
// not used). The axis points toward the camera, like the lane axis.
function annotatedBox(scene,v) {
  const boxes=scene.ground.cuboids.items.filter(box=>box.track===v.track);
  if(boxes.length!==1)throw new Error('The annotated vehicle is missing from the mapped ground annotations.');
  const [box]=boxes,M=box.worldFromBox,n=Math.hypot(M[0][0],M[2][0]),axis=[-M[0][0]/n,0,-M[2][0]/n],travel=mappedTravel(scene);
  // The front face is the end the heading points to, so the heading must follow the direction of travel (within the
  // preparation script's 15 degrees); a reversed annotation would make the camera-side end the front.
  if(-(axis[0]*travel[0]+axis[2]*travel[2])<Math.cos(15*Math.PI/180))throw new Error('The annotated vehicle does not face the direction of travel.');
  return {track:box.track,center:[...box.center],axis,dimensions:{width:box.width,height:box.height,length:box.length}};
}
// The annotated stationary vehicle of a mapped-ground scene. While the vehicle front pixel stays within
// ANNOTATED_FRONT_NUDGE_PX of the suggestion and the three size fields still hold it, the current vehicle is its averaged
// box exactly (a nudged front pixel changes nothing about the box). Returns null for any other vehicle (moved further,
// resized, or not annotated): that one is a declared box.
export function annotatedVehicle(scene,pixels,vehicle) {
  const v=isMappedGround(scene)?scene.suggestions?.vehicle:null;
  // Sizes as createScenario reads them (plain decimals), so a value it would refuse never names the annotated vehicle.
  if(v?.status!=='annotated'||!namesAnnotatedFront(v.pixel,pixels?.parking)||['width','height','length'].some(k=>decimal(vehicle?.[k])!==v.dimensions[k]))return null;
  return annotatedBox(scene,v);
}
// Horizontal centre of an annotated pose's front face (the end its heading points to, away from the camera).
const annotatedFront=pose=>{const half=pose.dimensions.length/2;return [pose.center[0]-pose.axis[0]*half,0,pose.center[2]-pose.axis[2]*half];};
// Protocol v3.1 currentVehicle.declaredReplacement: while the front pixel names the annotated stationary vehicle (within
// the nudge), the vehicle front point is that vehicle's averaged front point at the mapped ground height of its cell (the
// suggestion's mappedPoint), for the annotated box and for a declared or resized vehicle in its place. The front pixel's
// own ray hit is never used then: on AV2 8c the raster blocks it 1.03 m short of the front, where a same-size declared box
// turned Unknown into Blocked. Null when the pixel names no annotated vehicle.
function namedFront(scene,pixel) {
  const v=isMappedGround(scene)?scene.suggestions?.vehicle:null;
  if(v?.status!=='annotated'||!namesAnnotatedFront(v.pixel,pixel))return null;
  const f=annotatedFront(annotatedBox(scene,v)),y=mappedGroundHeight(scene,f[0],f[2]);
  if(y===null)throw new Error('No mapped ground height at the stationary vehicle front.');
  return [f[0],y,f[2]];
}
// The ground point and its evidence for an anchor key on a mapped-ground scene: the named stationary vehicle front
// (namedFront; it is not placed by a ray, so alongRayUncertaintyM is null), else the mapped ground hit at the evaluated
// position. The anchor's own position must reach mapped ground either way (a nudged front pixel is still placed). The
// pixel must be a whole pixel (userPixel); only an untouched suggestion is evaluated at its fractional pixelUV.
export function anchorPoint(scene,key,pixel) {
  userPixel(scene,pixel);
  const hit=mappedGroundHit(scene,evaluatedPosition(scene,key,pixel)),front=key==='parking'?namedFront(scene,pixel):null;
  if(front)return {point:front,...groundEvidence(scene,front),alongRayUncertaintyM:null};
  return {point:hit.point,supportClass:hit.supportClass,supportLabel:hit.supportLabel,hidden:hit.hidden,alongRayUncertaintyM:hit.alongRayUncertaintyM};
}
// Setback of an annotated pose: the distance along the mapped lane axis from the app's crossing point to the centre of
// the front face of the box the app tests (the annotated cuboid, sub-degree tilt removed). Current and proposed
// setbacks, the distance check and Nemotron's facts all use this definition for an annotated vehicle.
function annotatedSetback(crossing,axis,pose) {
  const front=annotatedFront(pose);
  return (front[0]-crossing[0])*axis[0]+(front[2]-crossing[2])*axis[2];
}
// The what-if box at a setback. Annotated: the annotated pose (heading, centre, dimensions) moved along the lane axis by
// the setback change, so at its own setback it is the annotated box exactly. Declared: the box on the lane axis through
// its origin with the retained front height.
export function whatIfBox(w,setback) {
  if(w.pose==='annotated')return {center:add(w.center,mul(w.axis,setback-w.setback)),axis:w.boxAxis,vehicle:w.vehicle,pose:'annotated'};
  return {center:vehicleCenter(w.origin,w.axis,setback,w.vehicle,w.groundHeight),axis:w.axis,vehicle:w.vehicle,pose:'declared'};
}
// Starting comparison values: the what-if starts on the current vehicle (createScenario's baselineSetback) and the driver
// at the recorded camera. vehicleDeclared: a declaration (Nemotron or an applied distance-check row) set the vehicle,
// which ends the annotated pose even when the sizes equal the annotation.
export function comparisonStart(scene,pixels,vehicle,vehicleDeclared=false) {
  const pose=vehicleDeclared?null:annotatedVehicle(scene,pixels,vehicle),p=placementFromPixels(scene,pixels,{annotated:!!pose});
  return {proposedSetback:pose?annotatedSetback(p.crossing,p.axis,pose):p.baselineSetback,driverDistance:p.captureDistance};
}
// Unit direction of travel: chord of the mapped ego lane's boundary midpoints (prepare suggestions use the same chord).
export function mappedTravel(scene) {
  const {leftBoundary:left,rightBoundary:right}=scene.ground.egoLane,mid=(l,r)=>[(l[0]+r[0])/2,(l[2]+r[2])/2];
  const a=mid(left[0],right[0]),b=mid(left.at(-1),right.at(-1)),dx=b[0]-a[0],dz=b[1]-a[1],n=Math.sqrt(dx*dx+dz*dz);
  return [dx/n,0,dz/n];
}
// Mapped-scene refusals for a current vehicle at or past the crossing; part names the point tested (the declared front
// or an annotated box's nearest front corner).
const cameraSide=(part,past)=>`Place the current vehicle on the camera side of the crossing (its ${part} is ${fmt2(past)} m past the crossing along the lane).`;
const separated=(part,gap,tolerance)=>`Current vehicle front must be separated from the crossing (its ${part} is ${fmt2(Math.max(gap,0))} m from it along the lane; more than ${fmt2(tolerance)} m is needed).`;
// The declared vehicle runs along the mapped lane axis (toward the camera) through its own front (a declared vehicle in
// place of the stationary one: that vehicle's front point, anchorPoint). The crossing and front are less than a metre
// apart, so the axis never comes from those two points. vehicleOrigin is the vehicle's line at the crossing's distance
// along the axis. check false: the front names the annotated vehicle, whose own box gets the same rules in createScenario
// (its nearest front corner), so the front point is not refused here.
function mappedVehicleLine(scene,crossing,parking,check=true) {
  const travel=mappedTravel(scene),axis=[-travel[0],0,-travel[2]];
  const baselineSetback=(parking[0]-crossing[0])*axis[0]+(parking[2]-crossing[2])*axis[2],tolerance=scene.ground.roadToleranceMeters;
  // A front beyond the crossing (negative gap along the axis) needs moving, not separating. Near the horizon one image
  // pixel can move a ground point by a metre (AV2 8c: the suggested crossing and front pixels are 3 px apart), so a
  // small edit of a declared front can land here; the refusal names the measured gap.
  if(check&&baselineSetback<-tolerance)throw new Error(cameraSide('front',-baselineSetback));
  if(check&&baselineSetback<=tolerance)throw new Error(separated('front',baselineSetback,tolerance));
  return {axis,baselineSetback,vehicleOrigin:[parking[0]-axis[0]*baselineSetback,parking[1],parking[2]-axis[2]*baselineSetback]};
}
// End of the contiguous run from s=0 along origin + s*axis (horizontal) that stays inside the mapped drivable areas
// with a mapped ground value, up to limit. Mapped ground is the captured road extent for these scenes.
function mappedRunEnd(scene,origin,axis,limit) {
  const g=scene.ground,x0=origin[0],z0=origin[2],dx=axis[0],dz=axis[2],cuts=terrainCuts(g,[x0,z0],[dx,dz],limit,[0,limit]);
  for(const {outline} of g.drivableAreas)for(let i=0,j=outline.length-1;i<outline.length;j=i++) {
    const [px,pz]=outline[j],ex=outline[i][0]-px,ez=outline[i][1]-pz,den=dx*ez-dz*ex;
    if(Math.abs(den)<EPS)continue;
    const fx=px-x0,fz=pz-z0,s=(fx*ez-fz*ex)/den,w=(fx*dz-fz*dx)/den;
    if(w>=0&&w<=1&&s>0&&s<limit)cuts.push(s);
  }
  cuts.sort((a,b)=>a-b);
  let end=0;
  for(let i=1;i<cuts.length;i++) {
    const ta=cuts[i-1],tb=cuts[i];
    if(!(tb>ta))continue;
    const m=(ta+tb)/2,x=x0+m*dx,z=z0+m*dz;
    if(!insideDrivable(g.drivableAreas,x,z)||!terrainSample(scene,x,z))break;
    end=tb;
  }
  return end;
}
function mappedPlacement(scene,pixels,annotated) {
  const points={},support={};
  for (const key of ['crossing','parking','target']) {
    const {point,...evidence}=anchorPoint(scene,key,pixels?.[key]);
    points[key]=point;support[key]=evidence;
  }
  const {axis,baselineSetback,vehicleOrigin}=mappedVehicleLine(scene,points.crossing,points.parking,!annotated);
  const captureDistance=dot(sub(scene.camera.eye,points.crossing),axis);
  if (captureDistance<=0) throw new Error('Place the current vehicle on the camera side of the crossing.');
  // The vehicle stays between the crossing and the recorded camera, on contiguous mapped drivable ground.
  const availableSetback=mappedRunEnd(scene,vehicleOrigin,axis,captureDistance);
  return {...points,axis,baselineSetback,captureDistance,availableSetback,vehicleOrigin,support};
}
// A user-selected point: whole pixels only on every scene (mapped ground: userPixel; measured depth: measuredPoint).
export function roadPoint(scene,pixel,label) {
  if(isMappedGround(scene))return mappedGroundHit(scene,userPixel(scene,pixel)).point;
  if(scene.measurements?.kind==='indexedSparse')throw new Error('Ground placement has no validated support in this observation.');
  const point=measuredPoint(scene,pixel);
  if (Math.abs(point[1])>scene.processing.roadToleranceMeters*2) throw new Error(`Choose ${label} on the measured road surface.`);
  return point;
}
function vehicleCenter(crossing,axis,setback,vehicle,groundHeight) {
  const center=add(crossing,mul(axis,setback+vehicle.length/2));
  center[1]=groundHeight+vehicle.height/2;
  return center;
}
export function placementPreview(scene,pixels,vehicle,targetHeight,vehicleDeclared=false) {
  const preview={};
  const span=Math.hypot(...sub(scene.bounds.max,scene.bounds.min));
  if(isMappedGround(scene)) {
    if(pixels?.crossing&&pixels?.parking&&['width','height','length'].every(k=>vehicle?.[k]!==''&&vehicle?.[k]!=null)) {
      // The same anchor points as the comparison (suggested sub-pixel positions, the named stationary vehicle front).
      const crossing=anchorPoint(scene,'crossing',pixels.crossing).point,parking=anchorPoint(scene,'parking',pixels.parking).point;
      // The annotated stationary vehicle is drawn in its annotated pose, like the comparison's current vehicle.
      const pose=vehicleDeclared?null:annotatedVehicle(scene,pixels,vehicle);
      const line=mappedVehicleLine(scene,crossing,parking,!pose);
      const dimensions=vehicleSizes(vehicle);
      if(Object.values(dimensions).some(n=>n>span))throw new Error('Actor dimensions exceed the measured scene extent.');
      preview.vehicle=pose?{dimensions:pose.dimensions,axis:pose.axis,center:pose.center,annotated:true}
        :{dimensions,axis:line.axis,center:vehicleCenter(line.vehicleOrigin,line.axis,line.baselineSetback,dimensions,parking[1])};
    }
    if(pixels?.target&&targetHeight!==''&&targetHeight!=null) {
      const height=sized(targetHeight,'Target height',SIZE_LIMITS.targetHeight);
      if(height>span)throw new Error('Actor dimensions exceed the measured scene extent.');
      const hit=anchorPoint(scene,'target',pixels.target);
      preview.targetGround=hit.point;preview.targetHidden=hit.hidden;
      preview.target=add(preview.targetGround,[0,height,0]);
    }
    return preview;
  }
  if(pixels?.crossing&&pixels?.parking&&['width','height','length'].every(k=>vehicle?.[k]!==''&&vehicle?.[k]!=null)) {
    const crossing=roadPoint(scene,pixels.crossing,'crossing'),parking=roadPoint(scene,pixels.parking,'vehicle');
    const delta=[parking[0]-crossing[0],0,parking[2]-crossing[2]],distance=Math.hypot(...delta);
    if(distance<=scene.processing.roadToleranceMeters)throw new Error('Current vehicle front must be separated from the crossing.');
    const dimensions=vehicleSizes(vehicle);
    if(Object.values(dimensions).some(n=>n>span))throw new Error('Actor dimensions exceed the measured scene extent.');
    const axis=mul(delta,1/distance),center=vehicleCenter(crossing,axis,distance,dimensions,parking[1]);
    preview.vehicle={dimensions,axis,center};
  }
  if(pixels?.target&&targetHeight!==''&&targetHeight!=null) {
    const height=sized(targetHeight,'Target height',SIZE_LIMITS.targetHeight);
    if(height>span)throw new Error('Actor dimensions exceed the measured scene extent.');
    preview.targetGround=roadPoint(scene,pixels.target,'person');
    preview.target=add(preview.targetGround,[0,height,0]);
  }
  return preview;
}
// annotated (mapped-ground scenes): the front pixel names the annotated stationary vehicle (annotatedVehicle), so its
// ground hit only marks that vehicle and is not refused for its gap to the crossing (createScenario tests the box).
export function placementFromPixels(scene,pixels,{annotated=false}={}) {
  if(isMappedGround(scene))return mappedPlacement(scene,pixels,annotated);
  const points={};
  for (const key of ['crossing','parking','target']) {
    points[key]=roadPoint(scene,pixels?.[key],key);
  }
  const delta=[points.parking[0]-points.crossing[0],0,points.parking[2]-points.crossing[2]], baselineSetback=Math.hypot(...delta);
  if (baselineSetback<=scene.processing.roadToleranceMeters) throw new Error('Current vehicle front must be separated from the crossing.');
  const axis=mul(delta,1/baselineSetback), captureDistance=dot(sub(scene.camera.eye,points.crossing),axis);
  if (captureDistance<=0) throw new Error('Place the current vehicle on the camera side of the crossing.');
  let availableSetback=0;
  const positions=scene.positions,roadTolerance=scene.processing.roadToleranceMeters*2,[x,y,z]=points.crossing;
  for (let i=0;i<scene.positions.length;i+=3) {
    if (Math.abs(positions[i+1])<=roadTolerance) {
      const setback=(positions[i]-x)*axis[0]+(positions[i+1]-y)*axis[1]+(positions[i+2]-z)*axis[2];
      availableSetback=Math.max(availableSetback,setback);
    }
  }
  return {...points,axis,baselineSetback,captureDistance,availableSetback};
}
export function segmentIntersectsBox(start,end,min,max) {
  if (![start,end,min,max].every(finite3)||min.some((v,i)=>v>max[i])) throw new Error('Invalid finite box geometry.');
  let enter=0,exit=1;
  for (let i=0;i<3;i++) {
    const d=end[i]-start[i];
    if (Math.abs(d)<EPS) { if (start[i]<min[i]||start[i]>max[i]) return false; continue; }
    const a=(min[i]-start[i])/d,b=(max[i]-start[i])/d;
    enter=Math.max(enter,Math.min(a,b)); exit=Math.min(exit,Math.max(a,b));
    if (enter>exit) return false;
  }
  return true;
}
// Exact contact with a supplied triangle; contact alone does not validate a reconstructed physical surface.
export function triangleHit(start,delta,a,b,c) {
  if (!finite3(start)||!finite3(delta)||!finite3(a)||!finite3(b)||!finite3(c))
    throw new Error('Triangle vectors must be ordinary arrays of three finite numbers.');
  return triangleHit3(start,delta,a,b,c);
}
function triangleHit3(start,delta,a,b,c) {
  const e1x=b[0]-a[0],e1y=b[1]-a[1],e1z=b[2]-a[2];
  const e2x=c[0]-a[0],e2y=c[1]-a[1],e2z=c[2]-a[2];
  const hx=delta[1]*e2z-delta[2]*e2y,hy=delta[2]*e2x-delta[0]*e2z,hz=delta[0]*e2y-delta[1]*e2x;
  // Preserve dot's zero seed and accumulation order without per-triangle arrays/callbacks.
  const det=0+e1x*hx+e1y*hy+e1z*hz;
  // Relative roundoff bound: a fixed determinant epsilon incorrectly discards small/shallow crossings.
  const parallelTolerance=32*Number.EPSILON*Math.hypot(e1x,e1y,e1z)*Math.hypot(e2x,e2y,e2z)*Math.hypot(...delta);
  if (Math.abs(det)<=parallelTolerance) {
    const e1=[e1x,e1y,e1z],e2=[e2x,e2y,e2z];
    const normal=cross(e1,e2),length=Math.hypot(...normal),end=add(start,delta);
    if (!length||[start,end].some(p=>Math.abs(dot(normal,sub(p,a)))>EPS*length)) return false;
    // Coplanar segment: clip against the triangle's three half-planes in its best-conditioned 2D projection.
    const drop=normal.reduce((best,n,i)=>Math.abs(n)>Math.abs(normal[best])?i:best,0);
    const project=p=>p.filter((_,i)=>i!==drop),vertices=[a,b,c].map(project),p=project(start),q=project(end);
    const side=(a,b,p)=>(b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]);
    const winding=Math.sign(side(...vertices));
    let enter=0,exit=1;
    for (let i=0;i<3;i++) {
      const a=vertices[i],b=vertices[(i+1)%3],s=winding*side(a,b,p),e=winding*side(a,b,q);
      if (s<0&&e<0) return false;
      if (s<0) enter=Math.max(enter,-s/(e-s));
      if (e<0) exit=Math.min(exit,-s/(e-s));
      if (enter>exit) return false;
    }
    // Only interior segment contact occludes; target/camera endpoint contact alone remains unresolved.
    return exit>EPS&&enter<1-EPS;
  }
  const sx=start[0]-a[0],sy=start[1]-a[1],sz=start[2]-a[2];
  const u=(0+sx*hx+sy*hy+sz*hz)/det;
  if (u<0||u>1) return false;
  const qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x;
  const v=(0+delta[0]*qx+delta[1]*qy+delta[2]*qz)/det;
  if (v<0||u+v>1) return false;
  const t=(0+e2x*qx+e2y*qy+e2z*qz)/det;
  return t>EPS&&t<1-EPS;
}
function cameraPoint(scene,p) {
  const R=scene.transform.worldFromCamera;
  return scene.transform.cameraOriginGround.map((o,i)=>o+R.reduce((s,row,j)=>s+row[i]*p[j],0));
}
export function classifySightline(scene,start,end,vehicle,axis,center) {
  if (![start,end,axis,center].every(finite3)) throw new Error('Invalid sightline coordinates.');
  if (Math.abs(axis[1])>EPS||Math.abs(Math.hypot(...axis)-1)>EPS) throw new Error('Vehicle axis must be a horizontal unit vector.');
  const side=[axis[2],0,-axis[0]];
  const local=p=>{const d=sub(p,center);return [dot(d,side),d[1],dot(d,axis)];};
  const half=[vehicle.width/2,vehicle.height/2,vehicle.length/2];
  if (segmentIntersectsBox(local(start),local(end),mul(half,-1),half)) return {state:'blocked',reason:'Declared vehicle volume intersects the sightline.',supportedFraction:null};
  if(scene.measurements?.kind==='indexedSparse')return {state:'unknown',reason:'Projected source returns do not establish a supported free-space path.',supportedFraction:null};
  const delta=sub(end,start);
  const positions=scene.positions,triangles=scene.triangles,aVertex=[0,0,0],bVertex=[0,0,0],cVertex=[0,0,0];
  for (let i=0;i<triangles.length;i+=3) {
    const a=triangles[i]*3,b=triangles[i+1]*3,c=triangles[i+2]*3;
    // Reuse call-local vectors; triangleHit reads them without retaining or mutating them.
    aVertex[0]=positions[a];aVertex[1]=positions[a+1];aVertex[2]=positions[a+2];
    bVertex[0]=positions[b];bVertex[1]=positions[b+1];bVertex[2]=positions[b+2];
    cVertex[0]=positions[c];cVertex[1]=positions[c+1];cVertex[2]=positions[c+2];
    // Adjacent returns constrain vertices, not opaque continuity between them. A short-edge gate is not physical validation.
    // Scene loading validates finite coordinates; these call-local vectors are always dense triples.
    if (triangleHit3(start,delta,aVertex,bVertex,cVertex)) return {state:'unknown',reason:'Reconstructed surface contact has unvalidated continuity between depth samples.',supportedFraction:null};
  }
  // Partition at every projected pixel boundary; missing-depth pixels cannot be skipped.
  const a=cameraPoint(scene,start),b=cameraPoint(scene,end),d=sub(b,a),c=scene.camera,cuts=[0,1];
  for (const [component,focal,principal,size] of [[0,c.fx,c.cx,c.width],[1,c.fy,c.cy,c.height]]) {
    for(let k=0;k<=size;k++) {
      const boundary=k-.5-principal,denominator=focal*d[component]-boundary*d[2];
      if(Math.abs(denominator)<EPS) continue;
      const t=(boundary*a[2]-focal*a[component])/denominator;
      if(t>0&&t<1) cuts.push(t);
    }
  }
  if(Math.abs(d[2])>EPS) {const t=-a[2]/d[2];if(t>0&&t<1)cuts.push(t);}
  cuts.sort((x,y)=>x-y);
  let supported=0;
  for(let i=1;i<cuts.length;i++) {
    const lo=cuts[i-1],hi=cuts[i];if(hi-lo<EPS)continue;
    const p=add(a,mul(d,(lo+hi)/2));if(p[2]<=EPS)continue;
    const u=Math.round(c.fx*p[0]/p[2]+c.cx),v=Math.round(c.fy*p[1]/p[2]+c.cy);
    if(u<0||u>=c.width||v<0||v>=c.height)continue;
    const depth=scene.depth[v*c.width+u]/scene.processing.depthDivisor;
    const furthest=Math.max(a[2]+d[2]*lo,a[2]+d[2]*hi);
    if(depth>0&&furthest<depth-scene.processing.roadToleranceMeters) supported+=hi-lo;
  }
  // Depth coverage is a diagnostic, not a validated free-space footprint. Never promote it to physical Clear.
  return {state:'unknown',reason:supported>=1-EPS
    ? 'Depth samples cover this angular path, but their free-space footprint is unvalidated.'
    : 'Part of the sightline has no supported free-space evidence.',supportedFraction:supported};
}
// ---------------------------------------------------------------- results: Blocked, or Unknown with a reason
// Sean (2026-10-12): a sightline that cuts into a vehicle box by less than this is "too close to call" (Unknown), not
// Blocked. The margin still has to be justified by the locked error budget. It only turns Blocked into Unknown.
export const TOO_CLOSE_TO_CALL_M=.1;
// Why a result is Unknown instead of Blocked (results are only Blocked or Unknown; Clear does not exist). Short texts for
// labels such as "Unknown (too close to call)"; each result also carries a full sentence in its reason.
export const UNKNOWN_REASONS=Object.freeze({
  'driver-inside-vehicle':'driver inside a vehicle','person-inside-vehicle':'person inside a vehicle',
  'driver-inside-object':'driver inside a real object','person-inside-object':'person inside a real object',
  'overlaps-real-object':'overlaps a real object','too-close-to-call':'too close to call'});
const fmt3=n=>String(Number(n.toFixed(3)));
// Signed sightline-to-box gap in the box frame (a, b: segment ends; half: half sizes), from the same closed
// segment-box test. A segment that touches the box gets minus its cut depth: the largest d at which it still touches the
// box shrunk by d on every side (0 for grazing contact). One that misses gets the smallest d at which it touches the
// box grown by d on every side. Bisection to 1e-9 m.
export function sightlineGap(a,b,half) {
  const touches=d=>half.every(n=>n-d>=0)&&segmentIntersectsBox(a,b,half.map(n=>d-n),half.map(n=>n-d));
  let lo=0,hi;
  if(touches(0)) {
    hi=Math.min(...half);
    if(touches(hi))return -hi;
    while(hi-lo>1e-9){const m=(lo+hi)/2;if(touches(m))lo=m;else hi=m;}
    return -lo;
  }
  hi=Math.max(...[a,b].map(p=>Math.max(...p.map((n,i)=>Math.abs(n)-half[i]))));
  while(hi-lo>1e-9){const m=(lo+hi)/2;if(touches(-m))hi=m;else lo=m;}
  return hi;
}
// Real annotated objects of a mapped-ground scene: every annotated cuboid except the suggestion's stationary vehicle,
// which is the current vehicle itself or the vehicle a declaration replaced. Each keeps its full 3D pose.
const OBJECTS=new WeakMap();
function realObjects(scene) {
  if(!isMappedGround(scene)||!scene.ground?.cuboids?.available)return [];
  const own=scene.suggestions?.vehicle?.status==='annotated'?scene.suggestions.vehicle.track:null,cached=OBJECTS.get(scene);
  if(cached&&cached.own===own&&cached.items===scene.ground.cuboids.items)return cached.list;
  const list=scene.ground.cuboids.items.filter(b=>b.track!==own).map(b=>{
    const M=b.worldFromBox,u=[0,1,2].map(k=>[M[0][k],M[1][k],M[2][k]]),e=[b.length/2,b.width/2,b.height/2];
    return {track:b.track,category:b.category,c:b.center,u,e,radius:Math.hypot(...e),local:p=>{const d=sub(p,b.center);return u.map(axis=>dot(d,axis));}};
  });
  OBJECTS.set(scene,{own,items:scene.ground.cuboids.items,list});
  return list;
}
const objectName=o=>o.category.toLowerCase().replace(/_/g,' ');
const insideObject=(o,p)=>o.local(p).every((n,i)=>Math.abs(n)<=o.e[i]);
const crossesObject=(o,a,b)=>segmentIntersectsBox(o.local(a),o.local(b),o.e.map(n=>-n),o.e);
// Separating-axis test of the tested upright box against a real object (touching counts as overlapping).
function overlapsObject(box,o) {
  const A={c:box.center,u:[[box.axis[2],0,-box.axis[0]],[0,1,0],box.axis],e:[box.vehicle.width/2,box.vehicle.height/2,box.vehicle.length/2]};
  const T=sub(o.c,A.c);
  if(Math.hypot(...T)>Math.hypot(...A.e)+o.radius)return false;
  const axes=[...A.u,...o.u];
  for(const a of A.u)for(const b of o.u){const c=cross(a,b),n=Math.hypot(...c);if(n>1e-9)axes.push(mul(c,1/n));}
  return axes.every(L=>Math.abs(dot(T,L))<=A.e.reduce((s,e,i)=>s+e*Math.abs(dot(A.u[i],L)),0)+o.e.reduce((s,e,i)=>s+e*Math.abs(dot(o.u[i],L)),0));
}
// The one result rule for a tested vehicle box: createScenario's current and proposed results and every distance-check
// row. s: {eye, target, targetGround}; box: {center, axis, vehicle}. Unknown, with a reason code and sentence, when (in
// order) the driver eye is inside the box; the person's ground-to-head column touches it; on a mapped-ground scene the
// eye or the person is inside a real annotated object, or the box overlaps one (ownPosition: the box is the annotated
// stationary vehicle where it really is, so that last test is skipped); or the sightline cuts into the box by less than
// TOO_CLOSE_TO_CALL_M. Otherwise classifySightline decides (diagnostics false: its box test alone, for sweep rows; the
// later diagnostics can only give Unknown). gapM is the signed gap (positive: the line passes the box by that much;
// negative: it cuts in that deep).
export function assessSightline(scene,s,box,{diagnostics=true,ownPosition=false}={}) {
  if(![s?.eye,s?.target,s?.targetGround,box?.center,box?.axis].every(finite3))throw new Error('Invalid sightline coordinates.');
  const side=[box.axis[2],0,-box.axis[0]],half=[box.vehicle.width/2,box.vehicle.height/2,box.vehicle.length/2],low=half.map(n=>-n);
  const local=p=>{const d=sub(p,box.center);return [dot(d,side),d[1],dot(d,box.axis)];};
  const e=local(s.eye),t=local(s.target),gapM=sightlineGap(e,t,half);
  const unknown=(reasonCode,reason,extra)=>({state:'unknown',reason,supportedFraction:null,reasonCode,gapM,...extra});
  if(e.every((n,i)=>Math.abs(n)<=half[i]))return unknown('driver-inside-vehicle','The driver eye is inside the vehicle box, so this placement is not tested.');
  if(segmentIntersectsBox(local(s.targetGround),t,low,half))return unknown('person-inside-vehicle','The person stands inside the vehicle box, so this placement is not tested.');
  const objects=realObjects(scene),ref=o=>({object:{category:o.category,track:o.track}});
  let o=objects.find(x=>insideObject(x,s.eye));
  if(o)return unknown('driver-inside-object',`The driver eye is inside a real annotated object (${objectName(o)}), so this placement is not tested.`,ref(o));
  o=objects.find(x=>crossesObject(x,s.targetGround,s.target));
  if(o)return unknown('person-inside-object',`The person stands inside a real annotated object (${objectName(o)}), so this placement is not tested.`,ref(o));
  o=ownPosition?null:objects.find(x=>overlapsObject(box,x));
  if(o)return unknown('overlaps-real-object',`The vehicle box overlaps a real annotated object (${objectName(o)}), so this placement is not tested.`,ref(o));
  const touches=segmentIntersectsBox(e,t,low,half),deep=TOO_CLOSE_TO_CALL_M;
  if(touches&&!(half.every(n=>n-deep>=0)&&segmentIntersectsBox(e,t,half.map(n=>deep-n),half.map(n=>n-deep))))
    return unknown('too-close-to-call',`The sightline cuts only ${fmt3(-gapM)} m into the vehicle box, less than the ${fmt2(deep)} m margin: too close to call.`);
  const v=diagnostics?classifySightline(scene,s.eye,s.target,box.vehicle,box.axis,box.center)
    :touches?{state:'blocked',reason:'Declared vehicle volume intersects the sightline.',supportedFraction:null}:{state:'unknown',reason:null,supportedFraction:null};
  return {...v,gapM};
}
export function createScenario(scene,input) {
  const mapped=isMappedGround(scene);
  // The current (baseline) vehicle: the annotated stationary vehicle's exact pose while it is still the suggested one
  // (front within the nudge, same sizes) and no declaration replaced it, otherwise the declared box on the lane axis. The
  // what-if is that same box moved along the lane axis, so at the current setback it equals the current vehicle.
  const pose=mapped&&!input?.vehicleDeclared?annotatedVehicle(scene,input?.pixels,input?.vehicle):null;
  const p=placementFromPixels(scene,input?.pixels,{annotated:!!pose});
  const vehicle=vehicleSizes(input?.vehicle);
  const targetHeight=sized(input?.targetHeight,'Target height',SIZE_LIMITS.targetHeight);
  const tolerance=roadTolerance(scene);
  const baselineSetback=pose?annotatedSetback(p.crossing,p.axis,pose):p.baselineSetback;
  // A turned annotated pose has one front corner nearer the crossing than its front-face centre, by half its width times
  // the sine of its turn from the lane axis (0 for a declared box). The box's nearest point gets the declared front's
  // rules: the current vehicle more than the road tolerance from the crossing, a what-if on the camera side of it, and
  // the slider and distance check starting at the road tolerance.
  const cornerOffset=pose?pose.dimensions.width/2*Math.abs(pose.axis[0]*p.axis[2]-pose.axis[2]*p.axis[0]):0;
  if(pose&&baselineSetback-cornerOffset<-tolerance)throw new Error(cameraSide('nearest front corner',cornerOffset-baselineSetback));
  if(pose&&baselineSetback-cornerOffset<=tolerance)throw new Error(separated('nearest front corner',baselineSetback-cornerOffset,tolerance));
  const proposedSetback=positive(input?.proposedSetback,'Proposed setback'),driverDistance=positive(input?.driverDistance,'Driver distance');
  if(proposedSetback-cornerOffset<=0)throw new Error('Keep the proposed vehicle on the camera side of the crossing.');
  // minimumSetback: the slider and distance-check start (the box's nearest point at the road tolerance). A typed or
  // declared setback below it is refused like the slider would (1e-9 m allows browser number serialization).
  const minimumSetback=tolerance+cornerOffset;
  if(proposedSetback<minimumSetback-EPS)throw new Error(`Proposed setback must be at least ${fmt2(Math.ceil(minimumSetback*100-1e-6)/100)} m from the crossing (the slider minimum).`);
  // An annotated box keeps its own centre line: the contiguous mapped run along the lane axis through its front face.
  const front=pose&&annotatedFront(pose),availableSetback=pose
    ?mappedRunEnd(scene,[front[0]-p.axis[0]*baselineSetback,0,front[2]-p.axis[2]*baselineSetback],p.axis,p.captureDistance):p.availableSetback;
  // The box's lane-axis extent behind its front-face centre: a turned box reaches length x cos further, plus its rear corner.
  const maximumSetback=availableSetback-(pose?pose.dimensions.length*Math.abs(pose.axis[0]*p.axis[0]+pose.axis[2]*p.axis[2])+cornerOffset:vehicle.length);
  const minimumDriverDistance=Math.max(tolerance,dot(sub(p.target,p.crossing),p.axis)+tolerance);
  if(maximumSetback<=baselineSetback||proposedSetback>maximumSetback+EPS)throw new Error('Vehicle extends beyond the captured road extent. Reduce its length or choose another placement.');
  if(driverDistance>p.captureDistance+EPS||driverDistance<minimumDriverDistance-EPS)throw new Error('Driver must stay within the captured approach and behind the target.');
  const span=Math.hypot(...sub(scene.bounds.max,scene.bounds.min));
  if([vehicle.width,vehicle.height,targetHeight].some(n=>n>span))throw new Error('Actor dimensions exceed the measured scene extent.');
  // Declared horizontal translation retains the measured current-front height; terrain contact is unvalidated.
  // Mapped scenes keep the vehicle on its own line along the lane axis (vehicleOrigin); otherwise it runs from the crossing.
  const whatIf=pose?{pose:'annotated',center:pose.center,setback:baselineSetback,axis:p.axis,boxAxis:pose.axis,vehicle:pose.dimensions,cornerOffset}
    :{pose:'declared',origin:mapped?p.vehicleOrigin:p.crossing,axis:p.axis,vehicle,groundHeight:p.parking[1],cornerOffset};
  // Driver eye: the recording camera's position and height, moved along the lane axis by the driver position. On AV2
  // that is the stereo camera (about 1.47 m above the mapped ground), not a seated driver's eye.
  const eye=add(scene.camera.eye,mul(p.axis,driverDistance-p.captureDistance)),target=add(p.target,[0,targetHeight,0]);
  const eyeGround=mapped?mappedGroundHeight(scene,eye[0],eye[2]):0;
  const baseline=whatIfBox(whatIf,baselineSetback),proposed=whatIfBox(whatIf,proposedSetback),line={eye,target,targetGround:p.target};
  return {crossing:p.crossing,axis:p.axis,vehicle,vehicleGroundHeight:p.parking[1],targetGround:p.target,target,eye,baseline,proposed,whatIf,baselineSetback,proposedSetback,driverDistance,
    eyeLabel:'recording camera height',eyeHeightM:eyeGround===null?null:eye[1]-eyeGround,
    limits:{minimumSetback,maximumSetback,captureDistance:p.captureDistance,minimumDriverDistance},
    currentVisibility:assessSightline(scene,line,baseline,{ownPosition:!!pose}),
    proposedVisibility:assessSightline(scene,line,proposed,{ownPosition:!!pose&&proposedSetback===baselineSetback}),
    ...(mapped&&{vehicleOrigin:p.vehicleOrigin,support:p.support})};
}
