async function boundedBytes(response,limit,exact=false) {
  const reader=response.body?.getReader();
  if(!reader)throw new Error('Artifact stream unavailable.');
  const bytes=new Uint8Array(limit);let offset=0;
  try {
    for(;;){const {done,value}=await reader.read();if(done)break;
      if(offset+value.byteLength>limit)throw new Error('Artifact exceeds declared byte length.');
      bytes.set(value,offset);offset+=value.byteLength;}
    if(exact&&offset!==limit)throw new Error('Artifact byte length mismatch.');
    return exact?bytes.buffer:bytes.slice(0,offset).buffer;
  }catch(error){await reader.cancel();throw error;}finally{reader.releaseLock();}
}

function recordedCpuConfiguration(execution) {
  const threads=execution?.usage?.threads,cpu=execution?.cpuConfiguration;
  if(![1,2,4].includes(threads))return false;
  // Original pinned one-thread artifacts predate the explicit CPU configuration.
  if(cpu===undefined)return threads===1;
  return cpu!==null&&typeof cpu==='object'&&!Array.isArray(cpu)&&
    Object.keys(cpu).sort().join()==='interOpThreads,intraOpThreads,provider'&&
    cpu.intraOpThreads===threads&&cpu.interOpThreads===1&&cpu.provider==='CPUExecutionProvider';
}

function matchingBounds(positions,bounds) {
  const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
  for(let i=0;i<positions.length;i++) {
    const k=i%3;min[k]=Math.min(min[k],positions[i]);max[k]=Math.max(max[k],positions[i]);
  }
  // Legacy descriptors retain pre-Float32 extrema; preserve the existing rendering tolerance.
  return min.every((n,k)=>Math.abs(n-bounds.min[k])<=2e-5&&Math.abs(max[k]-bounds.max[k])<=2e-5);
}

export async function loadScene(fetchResource,id,expectedMetadataSha256) {
  if(typeof id!=='string'||!/^[a-zA-Z0-9_-]+$/.test(id))throw new Error('Invalid scene identifier.');
  const hash=s=>typeof s==='string'&&/^[a-f0-9]{64}$/.test(s),
    digest=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),n=>n.toString(16).padStart(2,'0')).join('');
  if(expectedMetadataSha256!==undefined&&!hash(expectedMetadataSha256))throw new Error('Invalid expected scene metadata checksum.');
  const base=`/scenes/${id}/`,response=await fetchResource(base+'scene.json');
  if(!response.ok)throw new Error(`Scene metadata unavailable (${response.status}).`);
  const metadataBytes=await boundedBytes(response,128*1024);
  if(expectedMetadataSha256!==undefined&&await digest(metadataBytes)!==expectedMetadataSha256)throw new Error('Scene metadata checksum mismatch.');
  const scene=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(metadataBytes)),finite3=p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite),
    positive=n=>Number.isFinite(n)&&n>0,integer=n=>Number.isSafeInteger(n)&&n>0;
  const indexed=Object.hasOwn(scene,'measurements'),measurement=scene.measurements,
    cameraOrigin=indexed?scene.transform?.cameraOrigin:scene.transform?.cameraOriginGround;
  if(indexed&&!hash(expectedMetadataSha256))throw new Error('Indexed scene requires a caller-pinned metadata checksum.');
  if(indexed&&(measurement?.kind!=='indexedSparse'||measurement.evidence!=='geometricallyProjected'||measurement.groundPolicy!=='unvalidated'||
    measurement.pixelCellPolicy!=='nearestIntegerCenterRejectCollisions'||measurement.binaryEncoding!=='littleEndian'||!integer(measurement.count)||measurement.count>1_000_000||scene.triangleCount!==0||
    scene.transform?.originRole!=='leftCamera'||!finite3(scene.transform.cityOrigin)||Object.hasOwn(scene.transform,'cameraOriginGround')||
    JSON.stringify(scene.transform.worldFromCity)!=='[[1,0,0],[0,0,1],[0,-1,0]]'||
    !Array.isArray(cameraOrigin)||cameraOrigin.some(n=>n!==0)||scene.camera?.eye?.some(n=>n!==0)||scene.pointCount>measurement.count||scene.camera.width*scene.camera.height>16_000_000||
    Object.hasOwn(scene.buffers??{},'depth')||Object.hasOwn(scene.processing??{},'depthDivisor')||Object.hasOwn(scene.processing??{},'roadToleranceMeters')))
    throw new Error('Invalid indexed measurement policy.');
  if(scene.id!==id||scene.units!=='meters'||!finite3(scene.camera?.eye)||!finite3(scene.camera?.forward)||!finite3(scene.camera?.up)||!finite3(cameraOrigin)||
    !Array.isArray(scene.transform?.worldFromCamera)||scene.transform.worldFromCamera.length!==3||!scene.transform.worldFromCamera.every(finite3)||
    !['fx','fy','near','far'].every(k=>positive(scene.camera[k]))||
    !['width','height'].every(k=>integer(scene.camera[k]))||!Number.isSafeInteger(scene.camera.width*scene.camera.height)||scene.camera.width*scene.camera.height>16_000_000||scene.camera.near>=scene.camera.far||
    !['cx','cy'].every(k=>Number.isFinite(scene.camera[k]))||(!indexed&&(!positive(scene.processing?.depthDivisor)||!positive(scene.processing?.roadToleranceMeters))))throw new Error('Invalid measured calibration.');
  if(!integer(scene.pointCount)||scene.pointCount>1_000_000||!Number.isSafeInteger(scene.triangleCount)||scene.triangleCount<0||scene.triangleCount>2_000_000||
    !finite3(scene.bounds?.min)||!finite3(scene.bounds?.max)||scene.bounds.max.some((n,i)=>n<scene.bounds.min[i])||
    !positive(Math.hypot(...scene.bounds.max.map((n,i)=>n-scene.bounds.min[i]))))throw new Error('Invalid measured geometry metadata.');
  // cameraPoint uses the transpose as the inverse; accept six-decimal serialization, never repair the rotation.
  const R=scene.transform.worldFromCamera,rotationTolerance=1e-5;
  const [a,b,c]=R,determinant=a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]);
  if(R.some((row,i)=>R.some((other,j)=>Math.abs(row.reduce((sum,n,k)=>sum+n*other[k],0)-(i===j?1:0))>rotationTolerance))||
    Math.abs(determinant-1)>rotationTolerance)throw new Error('Invalid measured calibration: expected a proper orthonormal rotation.');
  // Calculation and renderer must use the same camera pose (OpenCV x-right/y-down/z-forward).
  if(R.some((row,i)=>Math.abs(scene.camera.eye[i]+row.reduce((sum,n,j)=>sum+n*cameraOrigin[j],0))>rotationTolerance||
    Math.abs(scene.camera.forward[i]-row[2])>rotationTolerance||Math.abs(scene.camera.up[i]+row[1])>rotationTolerance))
    throw new Error('Invalid measured calibration: camera pose disagrees with its transform.');
  if(indexed) {
    const image=scene.recordedImage;
    if(image?.file!=='recorded.png'||image.mimeType!=='image/png'||!integer(image.bytes)||image.bytes>16*1024*1024||!hash(image.sha256))
      throw new Error('Invalid indexed recorded image descriptor.');
    const r=await fetchResource(base+image.file);if(!r.ok)throw new Error(`Recorded image unavailable (${r.status}).`);
    const bytes=await boundedBytes(r,image.bytes,true);
    if(await digest(bytes)!==image.sha256)throw new Error('Recorded image checksum mismatch.');
    // Bind PNG's declared dimensions before browser decoding; the app also checks actual decoded size.
    const view=new DataView(bytes),signature=[137,80,78,71,13,10,26,10];
    if(bytes.byteLength<33||!signature.every((n,i)=>view.getUint8(i)===n)||view.getUint32(8)!==13||view.getUint32(12)!==0x49484452||
      view.getUint32(16)!==scene.camera.width||view.getUint32(20)!==scene.camera.height)
      throw new Error('Recorded PNG dimensions disagree with calibration.');
    scene.recordedImageBlob=new Blob([bytes],{type:image.mimeType});
  }
  const nativeTypes={sourceIndices:Uint32Array,referencePositions:Float64Array,referenceCameraPositions:Float64Array,referenceUV:Float64Array,referenceEligible:Uint8Array};
  if(indexed) {
    const expected=['positions','colors','triangles',...Object.keys(nativeTypes)],names=[...Object.values(scene.buffers??{}),...Object.values(measurement.buffers??{})],
      hash=n=>typeof n==='string'&&/^[a-f0-9]{64}$/.test(n);
    if(names.length!==expected.length||new Set(names).size!==names.length||Object.keys(scene.buffers).sort().join()!=='colors,positions,triangles'||
      Object.keys(measurement.buffers??{}).sort().join()!==Object.keys(nativeTypes).sort().join()||
      !hash(measurement.provenance?.preparedManifestSha256)||!hash(measurement.provenance?.referencesSha256)||
      typeof measurement.provenance?.logId!=='string'||!/^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/.test(measurement.provenance.logId)||
      !['cameraTimestampNs','lidarTimestampNs','crossingId'].every(k=>typeof measurement.provenance?.[k]==='string'&&/^[0-9]{1,32}$/.test(measurement.provenance[k])))throw new Error('Invalid indexed artifact ownership.');
  }
  const byteLengths={positions:scene.pointCount*3*4,colors:scene.pointCount*3*4,triangles:scene.triangleCount*3*4,depth:scene.camera.width*scene.camera.height*2,
    sourceIndices:measurement?.count*4,referencePositions:measurement?.count*3*8,referenceCameraPositions:measurement?.count*3*8,referenceUV:measurement?.count*2*8,referenceEligible:measurement?.count};
  const loadBuffers=async(target,types)=>Promise.all(Object.entries(types).map(async([key,Type])=>{
    const name=target.buffers?.[key];
    if(!/^[a-z]+\.bin$/.test(name))throw new Error(`Invalid ${key} artifact path.`);
    const length=target===scene||target===measurement?byteLengths[key]:key==='triangles'?target.triangleCount*3*4:target.pointCount*3*4;
    const r=await fetchResource(base+name);if(!r.ok)throw new Error(`${key} unavailable (${r.status}).`);
    const buffer=await boundedBytes(r,length,true);
    const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),n=>n.toString(16).padStart(2,'0')).join('');
    if(digest!==target.artifactHashes?.[key])throw new Error(`${key} checksum mismatch.`);
    target[key]=new Type(buffer);
  }));
  await loadBuffers(scene,indexed?{positions:Float32Array,colors:Float32Array,triangles:Uint32Array}:{positions:Float32Array,colors:Float32Array,triangles:Uint32Array,depth:Uint16Array});
  if(scene.positions.length!==scene.pointCount*3||scene.colors.length!==scene.positions.length||scene.triangles.length!==scene.triangleCount*3||(!indexed&&scene.depth.length!==scene.camera.width*scene.camera.height)||
    !scene.positions.every(Number.isFinite)||!scene.colors.every(n=>Number.isFinite(n)&&n>=0&&n<=1)||!scene.triangles.every(n=>n<scene.pointCount)||
    (!indexed&&!matchingBounds(scene.positions,scene.bounds)))throw new Error('Invalid measured geometry buffers.');
  if(indexed) {
    await loadBuffers(measurement,nativeTypes);
    const {count,sourceIndices,referencePositions,referenceCameraPositions,referenceUV,referenceEligible}=measurement,c=scene.camera;
    if(sourceIndices.length!==count||referencePositions.length!==count*3||referenceCameraPositions.length!==count*3||referenceUV.length!==count*2||referenceEligible.length!==count||
      !referencePositions.every(n=>Number.isFinite(n)||Number.isNaN(n))||!referenceCameraPositions.every(n=>Number.isFinite(n)||Number.isNaN(n))||!sourceIndices.every((n,i)=>n===i)||!referenceEligible.every(n=>n===0||n===1))
      throw new Error('Invalid indexed measurement buffers.');
    // This maps pixels to source rows, never depth. -1 is absent; -2 retains ambiguous ownership in the full source buffers.
    scene.pixelRows=new Int32Array(c.width*c.height).fill(-1);
    let rendered=0,min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
    for(let i=0;i<count;i++) {
      const p=referencePositions.subarray(i*3,i*3+3),
        xyz=referenceCameraPositions.subarray(i*3,i*3+3),
        uv=referenceUV.subarray(i*2,i*2+2),finite=xyz.every(Number.isFinite),forward=finite&&xyz[2]>0,
        eligible=forward&&uv.every(Number.isFinite)&&uv[0]>=0&&uv[0]<=c.width-1&&uv[1]>=0&&uv[1]<=c.height-1;
      if(finite?!p.every(Number.isFinite)||p.some((n,k)=>Math.abs(n-R[k].reduce((sum,value,j)=>sum+value*(xyz[j]-cameraOrigin[j]),0))>1e-9):!p.every(Number.isNaN))
        throw new Error('Indexed camera and world coordinates disagree.');
      if(!forward&&!uv.every(Number.isNaN))throw new Error('Unavailable indexed projection must retain NaN coordinates.');
      if(forward&&uv.some((n,k)=>{
        const projected=(k?c.fy*xyz[1]/xyz[2]+c.cy:c.fx*xyz[0]/xyz[2]+c.cx),
          tolerance=1e-9+256*Number.EPSILON*(Math.abs(n)+Math.abs(projected)+1);
        return !Number.isFinite(n)||!Number.isFinite(projected)||!Number.isFinite(tolerance)||Math.abs(n-projected)>tolerance;
      }))
        throw new Error('Indexed source coordinates disagree with calibrated projection.');
      if(referenceEligible[i]!==Number(eligible))throw new Error('Indexed eligibility disagrees with source projection.');
      if(!eligible)continue;
      if(p.some((n,k)=>scene.positions[rendered*3+k]!==Math.fround(n)||scene.positions[rendered*3+k]<scene.bounds.min[k]||scene.positions[rendered*3+k]>scene.bounds.max[k]))throw new Error('Rendered point lost indexed source identity or bounds.');
      for(let k=0;k<3;k++){const value=scene.positions[rendered*3+k];min[k]=Math.min(min[k],value);max[k]=Math.max(max[k],value);}
      rendered++;
      const pixel=Math.floor(uv[1]+.5)*c.width+Math.floor(uv[0]+.5);
      scene.pixelRows[pixel]=scene.pixelRows[pixel]===-1?i:-2;
    }
    if(rendered!==scene.pointCount||min.some((n,k)=>n!==scene.bounds.min[k])||max.some((n,k)=>n!==scene.bounds.max[k]))throw new Error('Indexed render population or bounds mismatch.');
  }
  if(Object.hasOwn(scene,'reconstruction')) {
    const estimate=scene.reconstruction,hash=s=>typeof s==='string'&&/^[a-f0-9]{64}$/.test(s);
    const provenanceFields=indexed?['baseMeasuredSceneSha256','preparedManifestSha256','calibrationSha256','referencesSha256','producerManifestSha256','planSha256','rawPredictionSha256','rawValiditySha256']:
      ['registeredPredictionSha256','producerManifestSha256','planSha256','observationsSha256','measuredSceneSha256'];
    if(!estimate||estimate.evidence!=='predicted'||estimate.role!=='rendererOnly'||
      (indexed&&estimate.kind!=='nativePoints')||
      estimate.model?.provider!=='nvidia'||typeof estimate.model?.id!=='string'||!estimate.model.id||
      typeof estimate.model?.revision!=='string'||!estimate.model.revision||!hash(estimate.model?.sha256)||
      !provenanceFields.every(k=>hash(estimate.provenance?.[k]))||
      !integer(estimate.pointCount)||estimate.pointCount>1_000_000||(indexed&&estimate.pointCount>scene.camera.width*scene.camera.height)||
      !Number.isSafeInteger(estimate.triangleCount)||estimate.triangleCount>2_000_000||(indexed?estimate.triangleCount!==0:estimate.triangleCount<=0)||!finite3(estimate.bounds?.min)||!finite3(estimate.bounds?.max)||
      estimate.bounds.max.some((n,i)=>n<estimate.bounds.min[i])||
      (!indexed&&(estimate.displayDomain?.maxDepthMeters!==scene.processing.maxDepthMeters||
      estimate.displayDomain?.meshEdgeLimitMeters!==scene.processing.meshEdgeLimitMeters)))
      throw new Error('Invalid estimated reconstruction metadata.');
    if(indexed) {
      const reuse=estimate.reuse,execution=reuse?.originalExecution,domain=estimate.displayDomain,provenance=estimate.provenance;
      const directBinding=provenance.inputBindingKind==='directNative';
      if(directBinding?(!hash(provenance.inputBindingResultSha256)||Object.hasOwn(provenance,'inputEquivalenceResultSha256')):
        (Object.hasOwn(provenance,'inputBindingKind')||Object.hasOwn(provenance,'inputBindingResultSha256')||!hash(provenance.inputEquivalenceResultSha256)))
        throw new Error('Invalid native reconstruction input binding.');
      if(estimate.provenance.preparedManifestSha256!==measurement.provenance.preparedManifestSha256||estimate.provenance.referencesSha256!==measurement.provenance.referencesSha256||
        reuse?.newInferenceRuns!==0||!hash(reuse.originalCacheKey)||typeof reuse.originalRecordBase!=='string'||!reuse.originalRecordBase||
        execution?.status!=='succeeded'||execution.provider!=='local_onnxruntime_cpu'||
        !integer(estimate.model.bytes)||execution.modelArtifact?.sha256!==estimate.model.sha256||execution.modelArtifact.bytes!==estimate.model.bytes||
        typeof execution.createdUtc!=='string'||!Number.isFinite(Date.parse(execution.createdUtc))||
        !Number.isFinite(execution.elapsedSeconds)||execution.elapsedSeconds<0||execution.elapsedSeconds>120||
        execution.usage?.cloudRequests!==0||execution.usage.jobs!==1||execution.usage.retries!==0||!recordedCpuConfiguration(execution)||
        domain?.policy!=='numericDepth_rightInputBounds_existingCameraClips'||domain.nearMeters!==scene.camera.near||domain.farMeters!==scene.camera.far||
        JSON.stringify(domain.rightPixelCenters)!==JSON.stringify([0,scene.camera.width-1]))
        throw new Error('Invalid native reconstruction provenance or rendering domain.');
    }
    if(Object.keys(estimate.buffers??{}).sort().join()!=='colors,positions,triangles'||new Set(Object.values(estimate.buffers??{})).size!==3||
      Object.values(estimate.buffers??{}).some(name=>[...Object.values(scene.buffers),...Object.values(measurement?.buffers??{})].includes(name)))
      throw new Error('Estimated reconstruction must retain separate measured buffers.');
    await loadBuffers(estimate,{positions:Float32Array,colors:Float32Array,triangles:Uint32Array});
    if(estimate.positions.length!==estimate.pointCount*3||estimate.colors.length!==estimate.positions.length||
      estimate.triangles.length!==estimate.triangleCount*3||!estimate.positions.every(Number.isFinite)||
      !estimate.colors.every(n=>Number.isFinite(n)&&n>=0&&n<=1)||!estimate.triangles.every(n=>n<estimate.pointCount)||
      !estimate.positions.every((n,i)=>n>=estimate.bounds.min[i%3]-2e-5&&n<=estimate.bounds.max[i%3]+2e-5)||!matchingBounds(estimate.positions,estimate.bounds))
      throw new Error('Invalid estimated reconstruction buffers.');
  }
  return scene;
}

// Selecting a declared source is explicit; failed estimates never fall back during loading.
export function renderGeometry(scene) {
  return Object.hasOwn(scene,'reconstruction') ? scene.reconstruction : scene;
}
