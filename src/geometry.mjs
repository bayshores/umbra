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
export function roadPoint(scene,pixel,label) {
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
export function placementPreview(scene,pixels,vehicle,targetHeight) {
  const preview={};
  const span=Math.hypot(...sub(scene.bounds.max,scene.bounds.min));
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
  const proposedSetback=positive(input?.proposedSetback,'Proposed setback'),driverDistance=positive(input?.driverDistance,'Driver distance');
  const maximumSetback=p.availableSetback-vehicle.length;
  const minimumDriverDistance=Math.max(scene.processing.roadToleranceMeters,dot(sub(p.target,p.crossing),p.axis)+scene.processing.roadToleranceMeters);
  if(maximumSetback<=p.baselineSetback||proposedSetback>maximumSetback+EPS)throw new Error('Vehicle extends beyond the captured road extent. Reduce its length or choose another placement.');
  if(driverDistance>p.captureDistance+EPS||driverDistance<minimumDriverDistance-EPS)throw new Error('Driver must stay within the captured approach and behind the target.');
  const span=Math.hypot(...sub(scene.bounds.max,scene.bounds.min));
  if([vehicle.width,vehicle.height,targetHeight].some(n=>n>span))throw new Error('Actor dimensions exceed the measured scene extent.');
  // Declared horizontal translation retains the measured current-front height; terrain contact is unvalidated.
  const center=s=>vehicleCenter(p.crossing,p.axis,s,vehicle,p.parking[1]);
  const eye=add(scene.camera.eye,mul(p.axis,driverDistance-p.captureDistance)),target=add(p.target,[0,targetHeight,0]);
  const baseline={center:center(p.baselineSetback)},proposed={center:center(proposedSetback)};
  return {crossing:p.crossing,axis:p.axis,vehicle,vehicleGroundHeight:p.parking[1],targetGround:p.target,target,eye,baseline,proposed,baselineSetback:p.baselineSetback,proposedSetback,driverDistance,
    limits:{maximumSetback,captureDistance:p.captureDistance,minimumDriverDistance},
    currentVisibility:classifySightline(scene,eye,target,vehicle,p.axis,baseline.center),
    proposedVisibility:classifySightline(scene,eye,target,vehicle,p.axis,proposed.center)};
}
