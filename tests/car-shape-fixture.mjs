// A synthetic ground.json carShape record (shape-v1 protocol dataSchema) that passes the loader checks, for tests only.
// top(i, j): the solid top of cell (i, j) (i from the front face, j from the left side), or null (not proven); the
// default is the box height. A refused record has every top null and every count 0.
export const FRAME_SENTENCE='x forward along the upright box heading (front face at x = +L/2), y to the car\'s left, z above the box bottom (box centre Y - H/2). Same as metric F upright_frame/car_local.';
export const GRID_ORDER='index = i*lateral + j; i from the front face, j from the left side';
const hex=c=>c.repeat(64);
export function carShapeRecord({track,length:L,width:W,height:H,top=()=>H,refusal=null}) {
  const n=Math.ceil(L/.1-1e-9),m=Math.ceil(W/.1-1e-9),refused=refusal!==null;
  const solidTopM=[];
  for(let i=0;i<n;i++)for(let j=0;j<m;j++)solidTopM.push(refused?null:top(i,j));
  const seen=solidTopM.map(v=>v===null?0:1);
  const sweep=k=>({timestampNs:String(1000+k*100),sha256:hex('0123456789'[k])});
  return {schema:'shape-v1',status:refused?'refused':'measured',refusal,track,box:{length:L,width:W,height:H},frame:FRAME_SENTENCE,
    grid:{along:n,lateral:m,cellAlongM:L/n,cellLateralM:W/m,maxCellM:.1,order:GRID_ORDER},
    solidTopM,counts:{returns:seen.map(k=>k*7),returnSweeps:seen.map(k=>k*3)},removed:{belowFloor:2,aboveBoxTop:1,otherObjects:0},
    sweeps:{rule:{maxEgoDistanceM:30,minEgoStepM:1,maxKept:40,split:'alternate',minShapeSweeps:2,downloadAttempts:4,retryWaitsS:[5,10,20]},
      planSha256:hex('e'),listing:{status:'ok',attempts:1},candidates:7,thinned:refused?1:6,kept:refused?1:6,
      shape:refused?[sweep(0)]:[sweep(0),sweep(2),sweep(4)],scoring:refused?[]:[sweep(1),sweep(5)],
      downloadDropped:refused?[]:[{timestampNs:sweep(3).timestampNs,set:'scoring',reason:'download-failed',attempts:4}]},
    diagnostics:{frontFaceToFirstSolidCellM:refused?null:.02,returnsBeyondEnds:{front:0,rear:3}},
    protocol:{placementProtocolVersion:'3.3',sha256:hex('a'),shapeProtocolSha256:'07b9535ffc6ca3d5ba0c8d7b9e8f471cd5642cbfa8e1b0da853ab2c40bb85825'},
    code:{path:'scripts/california/car_shape.py',sha256:hex('c')}};
}
