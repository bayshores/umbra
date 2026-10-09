import * as THREE from 'three';

export function driverViewport(width,height,camera) {
  const aspect=camera.width/camera.height;
  const viewWidth=width/height>aspect?Math.round(height*aspect):width;
  const viewHeight=width/height>aspect?height:Math.round(width/aspect);
  return {left:Math.round((width-viewWidth)/2),bottom:Math.round((height-viewHeight)/2),width:viewWidth,height:viewHeight};
}

export function comparisonScissors(viewport,hostWidth,split) {
  const cut=Math.max(viewport.left,Math.min(viewport.left+viewport.width,Math.round(hostWidth*split)));
  return [
    [viewport.left,viewport.bottom,cut-viewport.left,viewport.height],
    [cut,viewport.bottom,viewport.left+viewport.width-cut,viewport.height],
  ];
}

// Source UV denotes integer pixel centers; the rendering frustum uses image-edge coordinates.
export function calibrationCamera(intrinsics, eye, far) {
  const camera = new THREE.PerspectiveCamera();
  camera.position.set(...eye);
  camera.up.set(...intrinsics.up).normalize();
  camera.lookAt(camera.position.clone().add(new THREE.Vector3(...intrinsics.forward).normalize()));
  const near = intrinsics.near, cx = intrinsics.cx + .5, cy = intrinsics.cy + .5;
  camera.projectionMatrix.makePerspective(
    -cx * near / intrinsics.fx,
    (intrinsics.width - cx) * near / intrinsics.fx,
    cy * near / intrinsics.fy,
    -(intrinsics.height - cy) * near / intrinsics.fy,
    near, far,
  );
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  return camera;
}

