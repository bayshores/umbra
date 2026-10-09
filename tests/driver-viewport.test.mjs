import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import { driverViewport, comparisonScissors, calibrationCamera } from '../src/driver-viewport.mjs';

test('comparison clips tile one calibrated viewport without changing its size or registration',()=>{
  const camera={width:879,height:400};
  for(const [width,height] of [[1440,540],[390,300],[1800,900]]) {
    const viewport=driverViewport(width,height,camera),original={...viewport};
    assert.ok(Math.abs(viewport.width/viewport.height-camera.width/camera.height)<.01);
    for(const split of [0,.25,.5,.75,1]) {
      const [a,b]=comparisonScissors(viewport,width,split);
      assert.equal(a[0],viewport.left);assert.equal(a[0]+a[2],b[0]);
      assert.equal(b[0]+b[2],viewport.left+viewport.width);
      assert.equal(a[2]+b[2],viewport.width);
      assert.ok(a[2]>=0&&b[2]>=0);
      assert.deepEqual([a[1],a[3]],[viewport.bottom,viewport.height]);
      assert.deepEqual([b[1],b[3]],[viewport.bottom,viewport.height]);
      assert.deepEqual(viewport,original);
    }
  }
});


test('calibrated rendering maps source UV to image pixel centers without changing intrinsics',()=>{
  for(const [width,height,forward,up] of [[879,400,[0,0,-1],[0,1,0]],[736,320,[1,0,0],[0,1,0]]]) {
    const c={width,height,fx:876.848,fy:842.33,cx:211.7019,cy:97.1204,near:.01,forward,up},original=structuredClone(c),eye=[10,2,-4];
    const camera=calibrationCamera(c,eye,100);camera.updateMatrixWorld();
    const right=new THREE.Vector3(...forward).cross(new THREE.Vector3(...up)).normalize(),down=new THREE.Vector3(...up).negate();
    for(const [u,v,z] of [[0,0,4],[width-1,height-1,30],[365.655565,4.967037,36],[473.589478,100.555841,26]]) {
      const point=new THREE.Vector3(...eye).addScaledVector(right,(u-c.cx)*z/c.fx).addScaledVector(down,(v-c.cy)*z/c.fy).addScaledVector(new THREE.Vector3(...forward),z);
      const ndc=point.project(camera);
      for(const [hostWidth,hostHeight] of [[width,height],[1440,540],[390,300]]) {
        const viewport=driverViewport(hostWidth,hostHeight,c);
        const actual=[viewport.left+(ndc.x+1)/2*viewport.width,hostHeight-viewport.bottom-(ndc.y+1)/2*viewport.height],
          expected=[viewport.left+(u+.5)/width*viewport.width,hostHeight-viewport.bottom-viewport.height+(v+.5)/height*viewport.height];
        for(let k=0;k<2;k++)assert.ok(Math.abs(actual[k]-expected[k])<1e-9,`${actual[k]} != ${expected[k]}`);
      }
    }
    assert.deepEqual(c,original); // Rendering convention never edits the source principal point.
  }
});
