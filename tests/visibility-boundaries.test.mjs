import assert from 'node:assert/strict';
import test from 'node:test';
import {classifySightline,triangleHit} from '../src/geometry.mjs';

// Analytic isolated evidence with independently known plane x+z=10. Never a runtime street fixture.
const transform={cameraOriginGround:[0,0,0],worldFromCamera:[[1,0,0],[0,1,0],[0,0,1]]};
const processing={depthDivisor:1,roadToleranceMeters:.08};
const vehicle={width:1,height:1,length:1},axis=[0,0,1],farVehicle=[100,100,100];
const mesh={camera:{fx:1,fy:1,cx:0,cy:0,width:2,height:2},transform,processing,
  depth:new Uint16Array([10,5,10,5]),positions:new Float32Array([0,0,10,5,0,5,0,10,10]),triangles:new Uint32Array([0,1,2])};
const classify=(scene,a,b)=>classifySightline(scene,a,b,vehicle,axis,farVehicle);
const intersects=(scene,start,end)=>triangleHit(start,end.map((n,i)=>n-start[i]),...Array.from(scene.triangles.slice(0,3),index=>Array.from(scene.positions.slice(index*3,index*3+3))));

test('triangle contact rejects malformed and ambiguous vector representations',()=>{
  const valid=[[.25,.25,-1],[0,0,2],[0,0,0],[1,0,0],[0,1,0]];
  assert.equal(triangleHit(...valid),true);
  for (const bad of [Array(3),[0,,0],[0,0,0,1],new Float32Array([0,0,0]),{0:0,1:0,2:0},'000',null,[0,NaN,0],[0,Infinity,0]]) {
    for(let i=0;i<valid.length;i++) {
      const args=valid.map((v,j)=>j===i?bad:v);
      assert.throws(()=>triangleHit(...args),/three finite numbers/);
    }
  }
  assert.throws(()=>classifySightline(mesh,[0,,0],[1,0,1],vehicle,axis,farVehicle),/Invalid sightline coordinates/);
});

for (const [name,start,end] of [
  ['coplanar interior',[1,1,9],[2,1,8]],
  ['near-coplanar crossing',[1,1,9.000000000001],[2,1,7.999999999999]],
  ['coplanar entering edge',[-1,1,11],[2,1,8]],
  ['coplanar grazing edge',[0,1,10],[0,2,10]],
  ['coplanar vertex touch',[-1,1,11],[1,-1,9]],
]) test(`${name} intersects the exact triangle for either winding and segment direction`,()=>{
  // Independent geometric facts: interior overlap, edge overlap or vertex contact occurs on this plane.
  for (const triangles of [new Uint32Array([0,1,2]),new Uint32Array([0,2,1])]) {
    assert.equal(intersects({...mesh,triangles},start,end),true);
    assert.equal(intersects({...mesh,triangles},end,start),true);
  }
});

test('parallel offset and coplanar disjoint segments are not triangle hits',()=>{
  for (const [start,end] of [[[1,1,10],[2,1,9]],[[6,1,4],[7,1,3]]]) {
    assert.equal(intersects(mesh,start,end),false);
    assert.equal(classify(mesh,start,end).state,'unknown');
  }
});

test('small and shallow noncoplanar interior crossings are not treated as parallel',()=>{
  for (const [edge,dz] of [[.1,2e-9],[.0001,.005]]) {
    const positions=new Float32Array([0,0,10,edge,0,10,0,edge,10]);
    const start=[edge*.2,edge*.2,10-dz],end=[edge*.3,edge*.2,10+dz];
    // z=10 is crossed at t=.5; x+y=.45*edge lies strictly inside the triangle.
    for (const order of [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]) {
      const scene={...mesh,positions,triangles:new Uint32Array(order)};
      assert.equal(intersects(scene,start,end),true);
      assert.equal(intersects(scene,end,start),true);
    }
  }
});

test('sparse returns compatible with separated opaque discs cannot certify mesh-interior blockage',()=>{
  const returns=[[0,0,10],[.1,0,10],[0,.1,10]],radius=.005,hit=[.025,.025,10];
  assert.ok(returns.every(p=>Math.hypot(hit[0]-p[0],hit[1]-p[1])>radius));
  for(const [u,v] of [[0,0],[1,0],[0,1],[1,1]])assert.equal(returns.some(p=>Math.hypot(u*.1-p[0],v*.1-p[1])<=radius),u+v<2);
  const scene={...mesh,camera:{fx:100,fy:100,cx:0,cy:0,width:2,height:2},depth:new Uint16Array([10,10,10,0]),positions:new Float32Array(returns.flat())};
  const start=[.025,.025,9],end=[.025,.025,11];
  assert.equal(intersects(scene,start,end),true);
  for(const [a,b] of [[start,end],[end,start]]) {
    const result=classify(scene,a,b);
    assert.equal(result.state,'unknown');assert.match(result.reason,/unvalidated continuity/);assert.equal(result.supportedFraction,null);
  }
  assert.equal(classify(mesh,[1,1,9],[2,1,8]).state,'unknown');
});

test('invalid vehicle orientation cannot scale or collapse its declared volume',()=>{
  const scene={...mesh,positions:new Float32Array(),triangles:new Uint32Array()},start=[2,0,-2],end=[2,0,2],center=[0,0,0];
  assert.equal(classifySightline(scene,start,end,vehicle,axis,center).state,'unknown');
  for(const invalid of [[0,0,.1],[0,0,0],[0,1,0]])assert.throws(()=>classifySightline(scene,start,end,vehicle,invalid,center),/horizontal unit vector/);
  assert.equal(classifySightline(scene,[-2,0,0],[2,0,0],vehicle,axis,center).state,'blocked');
});

test('endpoint-only triangle contact remains unresolved for coplanar and transverse segments',()=>{
  const positions=new Float32Array([0,0,10,1,0,10,0,1,10]);
  for (const end of [[-.1,.25,10],[0,.25,11]])for (const order of [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]) {
    const scene={...mesh,positions,triangles:new Uint32Array(order)},start=[0,.25,10];
    assert.equal(intersects(scene,start,end),false);
    assert.equal(intersects(scene,end,start),false);
    assert.equal(classify(scene,start,end).state,'unknown');
    assert.equal(classify(scene,end,start).state,'unknown');
  }
});

test('sampled rays compatible with a hidden subpixel blocker cannot produce Clear',()=>{
  const camera={fx:10,fy:10,cx:1,cy:1,width:3,height:3};
  const scene={camera,transform,processing,depth:new Uint16Array(9).fill(20),positions:new Float32Array(),triangles:new Uint32Array()};
  const start=[.04,0,5],end=[.12,0,15];
  // This ray x=.008z passes through x=[.07,.09], z=[9.9,10.1]. Integer sampling rays x=0,+/-z/10 miss it.
  assert.ok(.008*9.9>.07&&.008*10.1<.09);
  assert.ok(9.9/10>.09);
  const result=classify(scene,start,end);
  assert.equal(result.state,'unknown');assert.equal(result.supportedFraction,1);
});
