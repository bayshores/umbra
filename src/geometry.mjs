const EPS = 1e-9;
const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
const sub = (a,b) => a.map((v,i)=>v-b[i]);
const add = (a,b) => a.map((v,i)=>v+b[i]);
const mul = (a,s) => a.map(v=>v*s);
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const finite3 = p => Array.isArray(p)&&p.length===3&&Number.isFinite(p[0])&&Number.isFinite(p[1])&&Number.isFinite(p[2]);
function positive(value,label) {
  const n = Number(value);
  if (value == null || value === '' || !Number.isFinite(n) || n <= 0) throw new Error(`${label} must be a positive number.`);
  return n;
}
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
function halfUlpFloat16(value) {
  const a=Math.abs(value);
  if(a<2**-14)return 2**-25;
  let e=Math.floor(Math.log2(a));
  if(2**e>a)e--;
  if(2**(e+1)<=a)e++;
  if(Math.round(a/2**(e-10))*2**(e-10)>=2**(e+1))e++;
  return 2**(e-10)/2;
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
// The runtime pixel -> mapped ground rule (same as scripts/california/prepare_av2_ground.py). Throws the Unknown reason.
export function mappedGroundHit(scene,pixel) {
  const [u,v]=pixel??[],c=scene.camera,g=scene.ground;
  if(!Number.isInteger(u)||!Number.isInteger(v)||u<0||u>=c.width||v<0||v>=c.height)throw new Error('Select a pixel inside the recorded image.');
  const q=[(u-c.cx)/c.fx,(v-c.cy)/c.fy,1],raw=scene.transform.worldFromCamera.map(row=>row[0]*q[0]+row[1]*q[1]+row[2]*q[2]);
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
    const point=[t*d[0],t*d[1],t*d[2]];
    if(!insideDrivable(g.drivableAreas,point[0],point[2]))throw new Error('First mapped-ground hit is outside the mapped drivable area.');
    const support=scene.groundSupportXZ,radius=g.supportRadiusMeters;
    let observed=false;
    for(let k=0;k<support.length&&!observed;k+=2){const dx=support[k]-point[0],dz=support[k+1]-point[2];observed=Math.sqrt(dx*dx+dz*dz)<=radius;}
    const supportClass=observed?'observed-nearby':'map-only';
    return {point,supportClass,supportLabel:SUPPORT_LABELS[supportClass],hidden:hiddenByCuboids(g,point),
      alongRayUncertaintyM:(g.roadToleranceMeters+halfUlpFloat16(cell.cityZ))/-d[1],rangeM:t,hitKind,terrainCityZ:cell.cityZ,terrainCell:cell.cell};
  }
  throw new Error(`No mapped ground within ${max} m along the pixel ray.`);
}
// The annotated stationary vehicle of a mapped-ground scene: suggestions.json names it (vehicle status "annotated", its
// track) and ground.json holds its cuboid. While the vehicle front pixel and the three size fields still hold that
// suggestion, the current vehicle is this cuboid's exact annotated centre, heading and dimensions; it is never re-derived
// from the road axis or the front pixel. Returns null for any other vehicle (moved, resized, or not annotated): that one is
// a declared box. classifySightline takes a horizontal axis, so the heading is the box's horizontal forward direction
// (its sub-degree pitch and roll are not used). The axis points toward the camera, like the lane axis.
export function annotatedVehicle(scene,pixels,vehicle) {
  const v=isMappedGround(scene)?scene.suggestions?.vehicle:null,front=pixels?.parking;
  if(v?.status!=='annotated'||!Array.isArray(front)||front[0]!==v.pixel[0]||front[1]!==v.pixel[1]||
    ['width','height','length'].some(k=>vehicle?.[k]===''||vehicle?.[k]==null||Number(vehicle[k])!==v.dimensions[k]))return null;
  const boxes=scene.ground.cuboids.items.filter(box=>box.track===v.track);
  if(boxes.length!==1)throw new Error('The annotated vehicle is missing from the mapped ground annotations.');
  const [box]=boxes,M=box.worldFromBox,n=Math.hypot(M[0][0],M[2][0]),axis=[-M[0][0]/n,0,-M[2][0]/n],travel=mappedTravel(scene);
  // The front face is the end the heading points to, so the heading must follow the direction of travel (within the
  // preparation script's 15 degrees); a reversed annotation would make the camera-side end the front.
  if(-(axis[0]*travel[0]+axis[2]*travel[2])<Math.cos(15*Math.PI/180))throw new Error('The annotated vehicle does not face the direction of travel.');
  return {track:box.track,center:[...box.center],axis,dimensions:{width:box.width,height:box.height,length:box.length}};
}
// Horizontal centre of an annotated pose's front face (the end its heading points to, away from the camera).
const annotatedFront=pose=>{const half=pose.dimensions.length/2;return [pose.center[0]-pose.axis[0]*half,0,pose.center[2]-pose.axis[2]*half];};
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
  const p=placementFromPixels(scene,pixels),pose=vehicleDeclared?null:annotatedVehicle(scene,pixels,vehicle);
  return {proposedSetback:pose?annotatedSetback(p.crossing,p.axis,pose):p.baselineSetback,driverDistance:p.captureDistance};
}
// Unit direction of travel: chord of the mapped ego lane's boundary midpoints (prepare suggestions use the same chord).
export function mappedTravel(scene) {
  const {leftBoundary:left,rightBoundary:right}=scene.ground.egoLane,mid=(l,r)=>[(l[0]+r[0])/2,(l[2]+r[2])/2];
  const a=mid(left[0],right[0]),b=mid(left.at(-1),right.at(-1)),dx=b[0]-a[0],dz=b[1]-a[1],n=Math.sqrt(dx*dx+dz*dz);
  return [dx/n,0,dz/n];
}
// The declared vehicle runs along the mapped lane axis (toward the camera) through its own front. The crossing and front
// are less than a metre apart, so the axis never comes from those two points. vehicleOrigin is the vehicle's line at
// the crossing's distance along the axis.
function mappedVehicleLine(scene,crossing,parking) {
  const travel=mappedTravel(scene),axis=[-travel[0],0,-travel[2]];
  const baselineSetback=(parking[0]-crossing[0])*axis[0]+(parking[2]-crossing[2])*axis[2],tolerance=scene.ground.roadToleranceMeters;
  // A front beyond the crossing (negative gap along the axis) needs moving, not separating.
  if(baselineSetback<-tolerance)throw new Error('Place the current vehicle on the camera side of the crossing.');
  if(baselineSetback<=tolerance)throw new Error('Current vehicle front must be separated from the crossing.');
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
function mappedPlacement(scene,pixels) {
  const points={},support={};
  for (const key of ['crossing','parking','target']) {
    const hit=mappedGroundHit(scene,pixels?.[key]);
    points[key]=hit.point;support[key]={supportClass:hit.supportClass,supportLabel:hit.supportLabel,hidden:hit.hidden,alongRayUncertaintyM:hit.alongRayUncertaintyM};
  }
  const {axis,baselineSetback,vehicleOrigin}=mappedVehicleLine(scene,points.crossing,points.parking);
  const captureDistance=dot(sub(scene.camera.eye,points.crossing),axis);
  if (captureDistance<=0) throw new Error('Place the current vehicle on the camera side of the crossing.');
  // The vehicle stays between the crossing and the recorded camera, on contiguous mapped drivable ground.
  const availableSetback=mappedRunEnd(scene,vehicleOrigin,axis,captureDistance);
  return {...points,axis,baselineSetback,captureDistance,availableSetback,vehicleOrigin,support};
}
export function roadPoint(scene,pixel,label) {
  if(isMappedGround(scene))return mappedGroundHit(scene,pixel).point;
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
      const crossing=roadPoint(scene,pixels.crossing,'crossing'),parking=roadPoint(scene,pixels.parking,'vehicle');
      const line=mappedVehicleLine(scene,crossing,parking);
      const dimensions=Object.fromEntries(['width','height','length'].map(k=>[k,positive(vehicle[k],`Vehicle ${k}`)]));
      if(Object.values(dimensions).some(n=>n>span))throw new Error('Actor dimensions exceed the measured scene extent.');
      // The annotated stationary vehicle is drawn in its annotated pose, like the comparison's current vehicle.
      const pose=vehicleDeclared?null:annotatedVehicle(scene,pixels,vehicle);
      preview.vehicle=pose?{dimensions:pose.dimensions,axis:pose.axis,center:pose.center,annotated:true}
        :{dimensions,axis:line.axis,center:vehicleCenter(line.vehicleOrigin,line.axis,line.baselineSetback,dimensions,parking[1])};
    }
    if(pixels?.target&&targetHeight!==''&&targetHeight!=null) {
      const height=positive(targetHeight,'Target height');
      if(height>span)throw new Error('Actor dimensions exceed the measured scene extent.');
      const hit=mappedGroundHit(scene,pixels.target);
      preview.targetGround=hit.point;preview.targetHidden=hit.hidden;
      preview.target=add(preview.targetGround,[0,height,0]);
    }
    return preview;
  }
  if(pixels?.crossing&&pixels?.parking&&['width','height','length'].every(k=>vehicle?.[k]!==''&&vehicle?.[k]!=null)) {
    const crossing=roadPoint(scene,pixels.crossing,'crossing'),parking=roadPoint(scene,pixels.parking,'vehicle');
    const delta=[parking[0]-crossing[0],0,parking[2]-crossing[2]],distance=Math.hypot(...delta);
    if(distance<=scene.processing.roadToleranceMeters)throw new Error('Current vehicle front must be separated from the crossing.');
    const dimensions=Object.fromEntries(['width','height','length'].map(k=>[k,positive(vehicle[k],`Vehicle ${k}`)]));
    if(Object.values(dimensions).some(n=>n>span))throw new Error('Actor dimensions exceed the measured scene extent.');
    const axis=mul(delta,1/distance),center=vehicleCenter(crossing,axis,distance,dimensions,parking[1]);
    preview.vehicle={dimensions,axis,center};
  }
  if(pixels?.target&&targetHeight!==''&&targetHeight!=null) {
    const height=positive(targetHeight,'Target height');
    if(height>span)throw new Error('Actor dimensions exceed the measured scene extent.');
    preview.targetGround=roadPoint(scene,pixels.target,'person');
    preview.target=add(preview.targetGround,[0,height,0]);
  }
  return preview;
}
export function placementFromPixels(scene,pixels) {
  if(isMappedGround(scene))return mappedPlacement(scene,pixels);
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
export function createScenario(scene,input) {
  const p=placementFromPixels(scene,input?.pixels);
  const vehicle=Object.fromEntries(['width','height','length'].map(k=>[k,positive(input?.vehicle?.[k],`Vehicle ${k}`)]));
  const targetHeight=positive(input?.targetHeight,'Target height');
  const tolerance=roadTolerance(scene),mapped=isMappedGround(scene);
  // The current (baseline) vehicle: the annotated stationary vehicle's exact pose while it is still the suggested one and
  // no declaration replaced it, otherwise the declared box on the lane axis. The what-if is that same box moved along the
  // lane axis, so at the current setback it equals the current vehicle.
  const pose=mapped&&!input?.vehicleDeclared?annotatedVehicle(scene,input?.pixels,vehicle):null;
  const baselineSetback=pose?annotatedSetback(p.crossing,p.axis,pose):p.baselineSetback;
  // A turned annotated pose has one front corner nearer the crossing than its front-face centre, by half its width times
  // the sine of its turn from the lane axis (0 for a declared box). The box's nearest point gets the declared front's
  // rules: the current vehicle more than the road tolerance from the crossing, a what-if on the camera side of it, and
  // the slider and distance check starting at the road tolerance.
  const cornerOffset=pose?pose.dimensions.width/2*Math.abs(pose.axis[0]*p.axis[2]-pose.axis[2]*p.axis[0]):0;
  if(pose&&baselineSetback-cornerOffset<-tolerance)throw new Error('Place the current vehicle on the camera side of the crossing.');
  if(pose&&baselineSetback-cornerOffset<=tolerance)throw new Error('Current vehicle front must be separated from the crossing.');
  const proposedSetback=positive(input?.proposedSetback,'Proposed setback'),driverDistance=positive(input?.driverDistance,'Driver distance');
  if(proposedSetback-cornerOffset<=0)throw new Error('Keep the proposed vehicle on the camera side of the crossing.');
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
  const eye=add(scene.camera.eye,mul(p.axis,driverDistance-p.captureDistance)),target=add(p.target,[0,targetHeight,0]);
  const baseline=whatIfBox(whatIf,baselineSetback),proposed=whatIfBox(whatIf,proposedSetback);
  return {crossing:p.crossing,axis:p.axis,vehicle,vehicleGroundHeight:p.parking[1],targetGround:p.target,target,eye,baseline,proposed,whatIf,baselineSetback,proposedSetback,driverDistance,
    // minimumSetback: the slider and distance-check start (the box's nearest point at the road tolerance).
    limits:{minimumSetback:tolerance+cornerOffset,maximumSetback,captureDistance:p.captureDistance,minimumDriverDistance},
    currentVisibility:classifySightline(scene,eye,target,baseline.vehicle,baseline.axis,baseline.center),
    proposedVisibility:classifySightline(scene,eye,target,proposed.vehicle,proposed.axis,proposed.center),
    ...(mapped&&{vehicleOrigin:p.vehicleOrigin,support:p.support})};
}
