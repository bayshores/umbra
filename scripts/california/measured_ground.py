"""Build umbra's measured-ground-v1 layer from independent, receipted AV2 sweeps.

The publisher raster remains byte-identical for evaluation reference selection. It is never
a runtime fallback. Calibration sweeps, the scored frame and its temporal neighbours cannot
contribute to the fitted surface or support mask. No downloads or optional dependencies.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pyarrow.feather as feather


ROOT = Path(__file__).resolve().parents[2]
CODE_BYTES = Path(__file__).read_bytes()
CODE_SHA256 = sha256(CODE_BYTES).hexdigest()
EXCLUDE_NS = 250_000_000
PARAMETERS = dict(cellMeters=.5, fitRadiusMeters=1., nearestSupportMeters=.5,
                  minimumReturns=12, minimumSweeps=3, groundGateMeters=.25,
                  drivableMarginMeters=.5, residualFloorMeters=.025, madMultiplier=3.,
                  robustRefits=2, maximumSlope=.25, windowHalfWidthMeters=32., maximumSupportSeconds=1.)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')


def pinned(path, expected):
    data = Path(path).read_bytes()
    require(digest(data) == expected, f'Input checksum mismatch: {path}')
    return data


def table(data, columns):
    t = feather.read_table(BytesIO(data), use_threads=False)
    require(len(set(t.column_names)) == len(t.column_names) and set(columns) <= set(t.column_names), 'Missing/duplicate Feather columns')
    return t.select(columns)


def rotation(row):
    w, x, y, z = [row[k] for k in ('qw', 'qx', 'qy', 'qz')]
    require(all(math.isfinite(v) for v in (w, x, y, z)) and abs(w*w+x*x+y*y+z*z-1) <= 1e-6, 'Invalid pose quaternion')
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def partition_sweeps(records, frame_timestamp, maximum_seconds=None):
    frame = int(frame_timestamp)
    ordered = sorted(records, key=lambda r: int(r['timestampNs']))
    require(len({r['timestampNs'] for r in ordered}) == len(ordered), 'Duplicate support timestamps')
    within_window = lambda r: maximum_seconds is None or abs(int(r['timestampNs'])-frame) <= maximum_seconds*1e9
    allowed = [r for r in ordered if abs(int(r['timestampNs']) - frame) > EXCLUDE_NS and within_window(r)]
    excluded = [{**r, 'reason': 'scoring frame' if int(r['timestampNs']) == frame else 'within temporal exclusion'}
                for r in ordered if abs(int(r['timestampNs']) - frame) <= EXCLUDE_NS]
    excluded += [{**r, 'reason': 'outside maximum support time window'} for r in ordered if not within_window(r)]
    return {'frameTimestampNs': str(frame), 'excludedNearNs': EXCLUDE_NS,
            'fit': [r for i, r in enumerate(allowed) if i % 4 != 0],
            'calibration': [r for i, r in enumerate(allowed) if i % 4 == 0], 'excluded': excluded}


def require_support_timestamp(timestamp, frame_timestamp):
    require(int(timestamp) != int(frame_timestamp), 'The scoring frame sweep is refused as support')
    require(abs(int(timestamp) - int(frame_timestamp)) > EXCLUDE_NS, 'Support sweep lies inside the temporal exclusion')


def inside_polygons(xz, areas, margin=0.):
    p = np.asarray(xz, float).reshape(-1, 2); inside = np.zeros(len(p), bool)
    for area in areas:
        poly = np.asarray(area['outline'], float); hit = np.zeros(len(p), bool)
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            with np.errstate(divide='ignore', invalid='ignore'):
                hit ^= ((a[1] > p[:, 1]) != (b[1] > p[:, 1])) & (p[:, 0] < (b[0]-a[0])*(p[:, 1]-a[1])/(b[1]-a[1])+a[0])
            if margin:
                e = b-a; norm = float(e@e)
                t = np.clip((p-a)@e/norm, 0, 1) if norm else np.zeros(len(p))
                inside |= np.sum((p-a-t[:, None]*e)**2, axis=1) <= margin**2
        inside |= hit
    return inside


def raster_heights(ground, heights, xz):
    t = ground['terrain']; sim = t['sim2']; r = np.asarray(sim['R']).reshape(2, 2); o = ground['frame']['cityOrigin']
    city = np.column_stack([xz[:, 0]+o[0], -xz[:, 1]+o[1]])
    image = sim['s']*(city@r.T+sim['t']); idx = np.floor(image).astype(np.int64)-[t['column0'], t['row0']]
    valid = (idx[:, 0] >= 0) & (idx[:, 0] < t['columns']) & (idx[:, 1] >= 0) & (idx[:, 1] < t['rows'])
    out = np.full(len(xz), np.nan); out[valid] = heights[idx[valid, 1], idx[valid, 0]]-o[2]
    return out


def outside_cuboids(ego, rows):
    keep = np.ones(len(ego), bool)
    for row in rows:
        c = np.array([row[k] for k in ('tx_m', 'ty_m', 'tz_m')]); half = np.array([row[k] for k in ('length_m', 'width_m', 'height_m')])/2
        local = (ego-c)@rotation(row)
        keep &= ~np.all(np.abs(local) <= half, axis=1)
    return keep


def camera_visible(points, boxes):
    keep = np.ones(len(points), bool)
    for box in boxes:
        m = np.asarray(box['worldFromBox']); c = np.asarray(box['center']); half = np.array([box['length'], box['width'], box['height']])/2
        a = -c@m; d = points@m; enter = np.zeros(len(points)); leave = np.ones(len(points)); ok = np.ones(len(points), bool)
        for k in range(3):
            flat = np.abs(d[:, k]) < 1e-9
            ok &= ~(flat & ((a[k] < -half[k]) | (a[k] > half[k])))
            with np.errstate(divide='ignore', invalid='ignore'):
                lo = (-half[k]-a[k])/d[:, k]; hi = (half[k]-a[k])/d[:, k]
            enter = np.where(flat, enter, np.maximum(enter, np.minimum(lo, hi)))
            leave = np.where(flat, leave, np.minimum(leave, np.maximum(lo, hi)))
        keep &= ~(ok & (enter <= leave))
    return keep


def fit_plane(points, sweep_ids, xz, params=PARAMETERS):
    """Robust, distance-weighted local plane; absence of measured support returns None."""
    p = np.asarray(points, float); ids = np.asarray(sweep_ids); delta = p[:, [0, 2]]-xz
    distance = np.sum(delta**2, axis=1); mask = distance < params['fitRadiusMeters']**2
    a = np.column_stack([np.ones(len(p)), delta]); weights = np.maximum(0, 1-distance/params['fitRadiusMeters']**2)**2
    for _ in range(params['robustRefits']+1):
        if mask.sum() < params['minimumReturns'] or len(np.unique(ids[mask])) < params['minimumSweeps']:
            return None
        if distance[mask].min() > params['nearestSupportMeters']**2:
            return None
        w = np.sqrt(weights[mask]); coefficients, _, rank, _ = np.linalg.lstsq(a[mask]*w[:, None], p[mask, 1]*w, rcond=None)
        if rank != 3:
            return None
        residual = p[:, 1]-a@coefficients; centre = np.median(residual[mask]); mad = np.median(np.abs(residual[mask]-centre))
        cutoff = max(params['residualFloorMeters'], params['madMultiplier']*1.4826*mad)
        revised = mask & (np.abs(residual-centre) <= cutoff)
        if np.array_equal(revised, mask):
            break
        mask = revised
    if mask.sum() < params['minimumReturns'] or len(np.unique(ids[mask])) < params['minimumSweeps'] or distance[mask].min() > params['nearestSupportMeters']**2:
        return None
    w = np.sqrt(weights[mask]); coefficients, _, rank, _ = np.linalg.lstsq(a[mask]*w[:, None], p[mask, 1]*w, rcond=None)
    if rank != 3:
        return None
    if np.hypot(coefficients[1], coefficients[2]) > params['maximumSlope']:
        return None
    return float(coefficients[0])


def fit_surface(points, sweep_ids, areas, params=PARAMETERS):
    p = np.asarray(points, float).reshape(-1, 3); ids = np.asarray(sweep_ids)
    half = params['windowHalfWidthMeters']; step = params['cellMeters']; count = int(math.ceil(2*half/step))+1
    origin = [-half, -half]; heights = np.full((count, count), np.nan)
    nodes = np.array([(x, z) for z in np.arange(count)*step-half for x in np.arange(count)*step-half])
    candidates = inside_polygons(nodes, areas, params['drivableMarginMeters'])
    bucket_size = params['fitRadiusMeters']; buckets = defaultdict(list)
    for i, key in enumerate(np.floor(p[:, [0, 2]]/bucket_size).astype(int)):
        buckets[tuple(key)].append(i)
    buckets = {k: np.asarray(v, np.int64) for k, v in buckets.items()}
    for i in np.flatnonzero(candidates):
        node = nodes[i]; key = np.floor(node/bucket_size).astype(int)
        parts = [buckets[(key[0]+dx, key[1]+dz)] for dx in (-1, 0, 1) for dz in (-1, 0, 1) if (key[0]+dx, key[1]+dz) in buckets]
        if not parts:
            continue
        indices = np.concatenate(parts); value = fit_plane(p[indices], ids[indices], node, params)
        if value is not None:
            heights.flat[i] = value
    values = heights[np.isfinite(heights)]
    return {'version': 'measured-ground-v1', 'originXZ': origin, 'cellMeters': step, 'columns': count, 'rows': count,
            'heights': [float(h) if np.isfinite(h) else None for h in heights.ravel()],
            'heightBounds': [float(values.min()), float(values.max())] if len(values) else None}


def build(site, output, params=PARAMETERS):
    site = Path(site).resolve(); output = Path(output).resolve()
    require(not output.exists(), 'Output already exists; use a new development variant directory')
    scene = read_json(site/'grounded/scene.json'); ground = read_json(site/'grounded/ground.json')
    provenance = scene['measurements']['provenance']; log = provenance['logId']; frame = provenance['lidarTimestampNs']
    require(log.startswith(('8c52d911-', 'c780d53a-', '88f47a10-', 'fd5c6932-', 'e1d68dde-')), 'This development build requires an explicitly allowed log')
    receipts = sorted((site/'shape').glob(f'shape-v2-{log}/receipt.json')) if (site/'shape').is_dir() else []
    require(len(receipts) <= 1, 'Ambiguous sweep receipt')
    records = []
    if receipts:
        receipt = read_json(receipts[0]); require(receipt['log'] == log and receipt['schema'] == 'shape-v2-download-receipt', 'Receipt log/schema mismatch')
        plan = pinned(receipts[0].parent/'sweep-plan.json', receipt['planSha256'])
        plan = json.loads(plan); require(plan['log'] == log, 'Sweep plan log mismatch')
        require(plan['inputs']['poses']['sha256'] == ground['sources']['poses']['sha256'] and
                plan['inputs']['annotations']['sha256'] == ground['sources']['annotations']['sha256'], 'Receipt poses/annotations differ from ground inputs')
        planned = {r['timestampNs']: r for r in plan['sweeps']}
        for row in receipt['sweeps']:
            ts = row['timestampNs']; expected = planned.get(ts)
            require(expected and row['file'] == f'sweeps/{ts}.feather' and row['bytes'] == expected['bytes'] and row['sizeMatches'] is True, 'Invalid sweep receipt record')
            records.append({'timestampNs': ts, 'sha256': row['sha256'], 'bytes': row['bytes']})
    partition = partition_sweeps(records, frame, params['maximumSupportSeconds'])
    # The partition is written before loading any sensor bytes or performing a fit.
    output.mkdir(parents=True); write_json(output/'partition.json', partition)
    (output/'measured-ground-source.py').write_bytes(CODE_BYTES)
    data_dir = site/'data'/log
    pose_columns = ['qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m']
    poses_data = pinned(data_dir/'city_SE3_egovehicle.feather', ground['sources']['poses']['sha256'])
    poses = {int(r['timestamp_ns']): r for r in table(poses_data, ['timestamp_ns']+pose_columns).to_pylist()}
    annotations_data = pinned(data_dir/'annotations.feather', ground['sources']['annotations']['sha256'])
    annotations = defaultdict(list)
    for row in table(annotations_data, ['timestamp_ns', 'length_m', 'width_m', 'height_m']+pose_columns).to_pylist():
        annotations[int(row['timestamp_ns'])].append(row)
    raster = np.frombuffer(pinned(site/'grounded/terrain.bin', ground['terrain']['sha256']), '<f4').reshape(ground['terrain']['rows'], ground['terrain']['columns'])
    fit_points, fit_ids, calibration, counts = [], [], [], []
    origin = np.asarray(ground['frame']['cityOrigin'])
    for group in ('fit', 'calibration'):
        for sweep_id, row in enumerate(partition[group]):
            ts = int(row['timestampNs']); require_support_timestamp(ts, frame)
            require(ts in poses and ts in annotations, 'Support sweep needs exact timestamp poses and annotations')
            raw = pinned(receipts[0].parent/'sweeps'/f'{ts}.feather', row['sha256'])
            require(len(raw) == row['bytes'], 'Sweep byte count mismatch')
            sweep = table(raw, ['x', 'y', 'z']); ego = np.column_stack([np.asarray(sweep[k]) for k in ('x', 'y', 'z')])
            valid = np.isfinite(ego).all(1); ego = ego[valid]; outside = outside_cuboids(ego, annotations[ts]); ego = ego[outside]
            pose = poses[ts]; city = ego@rotation(pose).T+np.array([pose[k] for k in ('tx_m', 'ty_m', 'tz_m')])
            delta = city-origin; points = delta[:, [0, 2, 1]]*np.array([1, 1, -1]); xz = points[:, [0, 2]]
            in_window = (np.abs(xz) <= params['windowHalfWidthMeters']).all(1)
            points = points[in_window]; xz = xz[in_window]
            h = raster_heights(ground, raster, xz)
            selected = np.isfinite(h) & (np.abs(points[:, 1]-h) <= params['groundGateMeters']) & inside_polygons(xz, ground['drivableAreas'], params['drivableMarginMeters'])
            selected_points = points[selected]
            counts.append({'timestampNs': str(ts), 'set': group, 'finiteReturns': int(valid.sum()), 'removedInsideAnnotations': int((~outside).sum()), 'groundCandidates': len(selected_points)})
            if group == 'fit':
                fit_points.append(selected_points); fit_ids.append(np.full(len(selected_points), sweep_id, np.int32))
            else:
                keep = inside_polygons(selected_points[:, [0, 2]], ground['drivableAreas']) & camera_visible(selected_points, ground['cuboids']['items'])
                calibration.extend(selected_points[keep].tolist())
    all_points = np.concatenate(fit_points) if fit_points else np.empty((0, 3)); all_ids = np.concatenate(fit_ids) if fit_ids else np.empty(0, np.int32)
    surface = fit_surface(all_points, all_ids, ground['drivableAreas'], params)
    surface.update(partition=partition, parameters=params, fitReturns=len(all_points),
                   sourceGroundSha256=digest((site/'grounded/ground.json').read_bytes()),
                   sourceSuggestionsSha256=digest((site/'grounded/suggestions.json').read_bytes()),
                   receiptSha256=digest(receipts[0].read_bytes()) if receipts else None,
                   processingSha256=CODE_SHA256,
                   sourceSite=os.path.relpath(site, output))
    write_json(output/'support-counts.json', counts)
    write_json(output/'calibration-points.json', calibration)
    shutil.copytree(site/'grounded', output/'grounded'); shutil.copytree(site/'assembled', output/'assembled')
    scene['processing']['groundPolicy'] = 'mappedGround with measured-ground-v1: continuous independently measured triangles; missing support is Unknown; publisher raster retained only as the evaluation selection reference'
    write_json(output/'grounded/scene.json', scene)
    ground.update(schemaVersion=4, measuredGround=surface)
    ground['limitations'] = [
        'Runtime ground is a continuous surface fitted to independent support sweeps; unmeasured regions stay holes.' if s.startswith('Mapped ground is publisher terrain') else
        'Displayed uncertainty is calibrated from independent end-to-end integer-click errors, without clipping; scoring-frame returns are excluded.' if 'along-ray uncertainty' in s else s
        for s in ground['limitations']]
    ground['limitations'].append('Support, calibration and scoring sweeps are from the same log and remain correlated; physical calibration and terrain contact remain provisional.')
    ground['license']['changes'] = 'Mapped geometry and annotations converted to umbra scene coordinates; independent LiDAR sweeps registered into a continuous measured ground surface with explicit support holes and click-error calibration. No endorsement implied.'
    ground['support'] = {'rows': [], 'count': 0, 'rule': 'independent measured surface only; scoring-sweep support rows are not used'}
    ground['alongRayUncertainty'] = {'method': 'supportSweepHoldout', 'heightBandMeters': None, 'rule': 'independent end-to-end integer-click horizontal error by hit range',
                                   'quantile': .975, 'minimumCount': 100, 'bins': []}
    for source in ground['sources'].values():
        if source is not None:
            source['path'] = os.path.relpath((site/'grounded'/source['path']).resolve(), output/'grounded')
    ground['processingCodeHashes']['scripts/california/measured_ground.py'] = CODE_SHA256
    protocol = {'schema': 'measured-ground-v1-development', 'parameters': params, 'excludeWithinNs': EXCLUDE_NS,
                'calibration': {'split': 'every fourth eligible sweep, excluded from fit and support mask', 'quantile': .975, 'minimumCount': 100,
                                'rangeEdgesM': [0, 10, 12, 15, 18, 20, 22, 25, 30], 'overLimit': 'display without clipping'},
                'legacyPlacementProtocolSha256': ground['protocol']['sha256']}
    write_json(output/'grounded/measured-ground-protocol.json', protocol)
    raw_protocol = (output/'grounded/measured-ground-protocol.json').read_bytes()
    ground['protocol'] = {'path': 'measured-ground-protocol.json', 'bytes': len(raw_protocol), 'sha256': digest(raw_protocol), 'readOnlyExternal': True}
    write_json(output/'grounded/ground.json', ground)
    result = subprocess.run(['node', '--input-type=module', '-', str(output)], input=FINALIZE_JS, text=True, capture_output=True, cwd=ROOT)
    (output/'finalize-stdout.txt').write_text(result.stdout); (output/'finalize-stderr.txt').write_text(result.stderr)
    require(result.returncode == 0, f'JavaScript calibration/finalization failed: {result.stderr}')
    return {'site': scene['id'], 'fitSweeps': len(partition['fit']), 'calibrationSweeps': len(partition['calibration']),
            'fitReturns': len(all_points), 'supportedVertices': sum(h is not None for h in surface['heights']), 'output': str(output)}


FINALIZE_JS = r'''
import fs from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {mappedGroundHeight,mappedGroundHit,measuredSurfaceHit} from './src/geometry.mjs';
const dir=process.argv[2],folder=path.join(dir,'grounded'),read=name=>JSON.parse(fs.readFileSync(path.join(folder,name))),
  sha=b=>createHash('sha256').update(b).digest('hex'),write=(name,value)=>{const b=JSON.stringify(value)+'\n';fs.writeFileSync(path.join(folder,name),b);return {file:name,bytes:Buffer.byteLength(b),sha256:sha(b)};};
const scene=read('scene.json'),g=read('ground.json'),suggestions=read('suggestions.json'),entry=read('catalog-entry.json');
scene.ground=g;scene.groundSupportXZ=new Float64Array();
const c=scene.camera,M=scene.transform.worldFromCamera,project=p=>{const q=[0,1,2].map(k=>p[0]*M[0][k]+p[1]*M[1][k]+p[2]*M[2][k]);return q[2]>0?[c.fx*q[0]/q[2]+c.cx,c.fy*q[1]/q[2]+c.cy]:null;},
  inCrop=uv=>uv&&uv[0]>=0&&uv[0]<=c.width-1&&uv[1]>=0&&uv[1]<=c.height-1,
  horizontal=(a,b)=>Math.hypot(a[0]-b[0],a[2]-b[2]),edges=[0,10,12,15,18,20,22,25,30],errors=edges.slice(1).map(()=>[]),unknown={};
let eligible=0,placed=0;
for(const p of JSON.parse(fs.readFileSync(path.join(dir,'calibration-points.json')))) {
  const uv=project(p);if(!inCrop(uv))continue;
  const pixel=uv.map(n=>Math.floor(n+.5));if(!inCrop(pixel))continue;eligible++;
  try {const hit=measuredSurfaceHit(scene,pixel),range=Math.hypot(hit.point[0],hit.point[2]),index=edges.findIndex((n,i)=>i<edges.length-1&&range>=n&&range<edges[i+1]);
    if(index>=0){errors[index].push(horizontal(hit.point,p));placed++;}
  }catch(e){unknown[e.message]=(unknown[e.message]??0)+1;}
}
const calibrationBins=[];
for(let i=0;i<errors.length;i++) {
  let lo=i,hi=i,pool=[...errors[i]];
  while(pool.length<g.alongRayUncertainty.minimumCount&&(lo>0||hi<errors.length-1)) {
    if(lo>0)pool.push(...errors[--lo]);if(hi<errors.length-1)pool.push(...errors[++hi]);
  }
  if(pool.length<g.alongRayUncertainty.minimumCount)continue;
  pool.sort((a,b)=>a-b);const meters=Math.ceil(pool[Math.ceil(.975*pool.length)-1]*1000)/1000;
  calibrationBins.push({minRangeM:edges[i],maxRangeM:edges[i+1],meters,count:pool.length,ownCount:errors[i].length,poolMinRangeM:edges[lo],poolMaxRangeM:edges[hi+1]});
}
g.alongRayUncertainty.bins=calibrationBins;g.alongRayUncertainty.coverage={eligible,placed,unknown};
fs.writeFileSync(path.join(dir,'calibration-errors.json'),JSON.stringify({edges,errors,eligible,placed,unknown})+'\n');
const unknownElement=(e,reason)=>({status:'unknown',supportClass:'unknown',reason,label:e.label,...(e.crossingError&&{crossingError:e.crossingError}),...(e.mappedPoint&&{mappedPoint:e.mappedPoint}),...(e.pixelUV&&{pixelUV:e.pixelUV,pixel:e.pixel})});
for(const role of ['crossing','vehicle','person']) {
  const e=suggestions[role];if(e.status==='unknown')continue;
  const y=mappedGroundHeight(scene,e.mappedPoint[0],e.mappedPoint[2]);
  if(y===null){suggestions[role]=unknownElement(e,'No independently measured ground support at the source anchor.');continue;}
  e.mappedPoint[1]=y;const uv=project(e.mappedPoint);
  if(!inCrop(uv)){suggestions[role]=unknownElement(e,'Measured source anchor projects outside the image.');continue;}
  e.pixelUV=uv;e.pixel=uv.map(n=>Math.floor(n+.5));
  try {
    const hit=mappedGroundHit(scene,uv);Object.assign(e,{groundPoint:hit.point,supportClass:hit.supportClass,supportLabel:hit.supportLabel,hidden:hit.hidden,
      hitKind:hit.hitKind,terrainCityZ:hit.terrainCityZ,terrainCell:hit.terrainCell,alongRayUncertaintyM:hit.alongRayUncertaintyM,rangeM:hit.rangeM,
      horizontalRangeM:Math.hypot(hit.point[0],hit.point[2]),pixelOffsetFromMappedM:horizontal(hit.point,e.mappedPoint),
      supportEvidence:{independentFitSweeps:g.measuredGround.partition.fit.length}});
    try{const integer=mappedGroundHit(scene,e.pixel);e.integerPixelCheck={groundPoint:integer.point,residualM:horizontal(integer.point,e.mappedPoint)};}
    catch(error){e.integerPixelCheck={reason:error.message};}
  }catch(error){suggestions[role]=unknownElement(e,error.message);}
}
if(suggestions.vehicle.status!=='annotated')g.carShape=null;
suggestions.tolerances.alongRayHeightBandMeters=null;
suggestions.license=g.license;
suggestions.protocolSha256=g.protocol.sha256;suggestions.inputs['protocol.json']=g.protocol.sha256;
g.measuredGround.suggestionAdapter={horizontalAnchors:'unchanged from source suggestions; heights, projections and ray diagnostics refreshed',sourceSha256:g.measuredGround.sourceSuggestionsSha256};
const gd=write('ground.json',g);delete scene.ground;delete scene.groundSupportXZ;
scene.measurements.ground=gd;const sd=write('scene.json',scene);
suggestions.inputs['scene.json']=sd.sha256;suggestions.inputs['ground.json']=gd.sha256;
const vd=write('suggestions.json',suggestions);entry.metadataSha256=sd.sha256;entry.ground={file:gd.file,sha256:gd.sha256};entry.suggestions={file:vd.file,sha256:vd.sha256};
write('catalog-entry.json',entry);
console.log(JSON.stringify({calibration:g.alongRayUncertainty,groundBytes:gd.bytes,suggestionStatus:Object.fromEntries(['crossing','vehicle','person'].map(k=>[k,suggestions[k].status]))}));
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cell-meters', type=float, default=PARAMETERS['cellMeters'])
    parser.add_argument('--fit-radius-meters', type=float, default=PARAMETERS['fitRadiusMeters'])
    args = parser.parse_args(); params = {**PARAMETERS, 'cellMeters': args.cell_meters, 'fitRadiusMeters': args.fit_radius_meters}
    require(.1 <= args.cell_meters <= 1 and .5 <= args.fit_radius_meters <= 2, 'Unsupported grid/radius')
    print(json.dumps(build(args.site, args.out, params)))


if __name__ == '__main__':
    main()
