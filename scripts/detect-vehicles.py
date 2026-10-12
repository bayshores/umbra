#!/usr/bin/env python3
"""Detect vehicles and people in one prepared scene with RF-DETR Medium and place each on measured depth.

Separate offline preparation environment (scripts/requirements-detection.txt); never imported by the app.

Run:
  .local/venv-detect/bin/python -I scripts/detect-vehicles.py public/scenes/<id> [--output <json>] [--replace]

Weights must already be cached as <--weights-dir>/rf-detr-medium.pth (default .local/venv-detect/rf_home). Their
sha256 is checked against the pinned value before the model is built, so rfdetr never downloads anything.

Each detection is placed with the scene's own measured depth, with the same math as src/geometry.mjs
measuredPoint (raw 0 = no depth; nothing is filled). A car/truck/bus is a curb-lane parked candidate when it
scores >= 0.5, has measured depth in the bottom of its box, and that point lies ahead of the camera within the
right curb lane (iteration-1 analysis). Whether it is between the camera and the crossing is decided by
scripts/suggest-placements.py, which knows the measured crossing.
"""
import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = 'scripts/detect-vehicles.py'
WEIGHTS_FILE = 'rf-detr-medium.pth'
WEIGHTS_SHA256 = '749ff6071828aaffac63e204c4f4135ed3d6cdae4d702e086c360edc3b5768c8'
WEIGHTS_URL = 'https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth'
RULES = {
    'threshold': 0.25,                 # detections kept in the file
    'candidateMinScore': 0.5,
    'candidateClasses': ['car', 'truck', 'bus'],
    'lowerBoxFraction': 0.3,           # bottom 30% of box rows: where the vehicle meets the road
    'minMeasuredPixels': 8,            # below this a statistic is null (Unknown)
    'bodyMaxHeightMeters': 3.0,        # body points: above 2 x road tolerance, below this
    'curbBandsForwardMeters': [8, 29, 3],
    'curbRaisedMaxHeightMeters': 0.6,
    'curbLaneMeters': [-3.0, 0.5],     # lateral range relative to the measured right curb
}


def fail(message):
    raise SystemExit(f'detect-vehicles: {message}')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_scene(scene_dir):
    meta = json.loads((scene_dir / 'scene.json').read_text())
    if meta.get('buffers', {}).get('depth') != 'depth.bin' or 'measurements' in meta:
        fail('scene has no depth.bin buffer')
    if sha256(scene_dir / 'depth.bin') != meta['artifactHashes']['depth']:
        fail('depth.bin does not match scene.json artifactHashes.depth')
    c = meta['camera']
    W, H = c['width'], c['height']
    raw = np.fromfile(scene_dir / 'depth.bin', dtype='<u2')
    if raw.size != W * H:
        fail('depth.bin size does not match camera width*height')
    image = Image.open(scene_dir / 'recorded.jpg').convert('RGB')
    if image.size != (W, H):
        fail('recorded.jpg size does not match camera')
    raw = raw.reshape(H, W)
    d = raw / meta['processing']['depthDivisor']
    vv, uu = np.mgrid[0:H, 0:W]
    P = np.stack([(uu - c['cx']) * d / c['fx'], (vv - c['cy']) * d / c['fy'], d], -1)
    world = (P - np.array(meta['transform']['cameraOriginGround'])) @ np.array(meta['transform']['worldFromCamera']).T
    return meta, image, raw > 0, world, 2 * meta['processing']['roadToleranceMeters']


def right_curb(valid, world, tol):
    """Median over forward bands of the 5th-percentile x of raised points (kerb face / top) right of the camera."""
    X, Y, F = world[..., 0], world[..., 1], -world[..., 2]
    f0, f1, step = RULES['curbBandsForwardMeters']
    xs = []
    for f in range(f0, f1, step):
        up = valid & (F >= f) & (F < f + step) & (X > 0) & (Y > tol) & (Y < RULES['curbRaisedMaxHeightMeters'])
        if up.sum() >= 50:
            xs.append(float(np.percentile(X[up], 5)))
    return (float(np.median(xs)), len(xs)) if xs else (None, 0)


def place(det, valid, world, tol, W, H):
    X, Y, F = world[..., 0], world[..., 1], -world[..., 2]
    x1, y1, x2, y2 = det['box']
    u0, u1 = max(0, int(np.floor(x1))), min(W - 1, int(np.ceil(x2)))
    v0, v1 = max(0, int(np.floor(y1))), min(H - 1, int(np.ceil(y2)))
    lv0 = max(v0, int(np.floor(y2 - RULES['lowerBoxFraction'] * (y2 - y1))))
    box = np.zeros((H, W), bool)
    box[v0:v1 + 1, u0:u1 + 1] = True
    lower = np.zeros((H, W), bool)
    lower[lv0:v1 + 1, u0:u1 + 1] = True
    m_box, m_low = box & valid, lower & valid
    body = m_box & (Y > tol) & (Y < RULES['bodyMaxHeightMeters'])
    n = RULES['minMeasuredPixels']
    r = {**det, 'measuredPixels': int(m_box.sum()), 'lowerMeasuredPixels': int(m_low.sum()),
         'lowerMedianWorld': None, 'bodyPoints': int(body.sum()), 'bodyForwardRangeM': None, 'bodyLateralRangeM': None}
    if m_low.sum() >= n:
        r['lowerMedianWorld'] = [round(float(np.median(world[..., k][m_low])), 3) for k in range(3)]
    if body.sum() >= n:
        r['bodyForwardRangeM'] = [round(float(np.percentile(F[body], q)), 3) for q in (10, 90)]
        r['bodyLateralRangeM'] = [round(float(np.percentile(X[body], q)), 3) for q in (10, 90)]
    return r


def candidacy(r, curb_x):
    if r['class'] not in RULES['candidateClasses']:
        return None
    reasons = []
    if r['score'] < RULES['candidateMinScore']:
        reasons.append(f"score {r['score']} below {RULES['candidateMinScore']}")
    if r['lowerMedianWorld'] is None:
        reasons.append('too little measured depth in the lower box (position Unknown)')
    elif curb_x is None:
        reasons.append('right curb not measured')
    else:
        x, z = r['lowerMedianWorld'][0], r['lowerMedianWorld'][2]
        lo, hi = RULES['curbLaneMeters']
        if z >= 0:
            reasons.append('not ahead of the camera')
        if not curb_x + lo <= x <= curb_x + hi:
            reasons.append(f'not in the right curb lane (x {x:.2f} m, curb {curb_x:.2f} m)')
    return {'eligible': not reasons, 'reasons': reasons}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('scene', type=Path, help='prepared scene directory (scene.json, recorded.jpg, depth.bin)')
    ap.add_argument('--output', type=Path, help='detections JSON (default <scene>/detections.json)')
    ap.add_argument('--replace', action='store_true', help='overwrite an existing output')
    ap.add_argument('--weights-dir', type=Path, default=ROOT / '.local/venv-detect/rf_home')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--warm', type=int, default=3, help='extra timed predictions after the first')
    args = ap.parse_args()
    scene_dir = args.scene.resolve()
    output = (args.output or scene_dir / 'detections.json').resolve()
    if output.exists() and not args.replace:
        fail(f'{output} exists; pass --replace to overwrite')
    weights = args.weights_dir.resolve() / WEIGHTS_FILE
    if not weights.is_file():
        fail(f'{weights} not found; weights are never downloaded by this script')
    t = time.perf_counter()
    weights_sha = sha256(weights)
    if weights_sha != WEIGHTS_SHA256:
        fail(f'{weights} sha256 {weights_sha} does not match the pinned {WEIGHTS_SHA256}')
    t_hash = time.perf_counter() - t
    meta, image, valid, world, tol = load_scene(scene_dir)

    os.environ['RF_HOME'] = str(weights.parent)
    os.environ['HF_HUB_OFFLINE'] = '1'
    t = time.perf_counter()
    import torch
    from rfdetr import RFDETRMedium
    from rfdetr.assets.coco_classes import COCO_CLASSES
    t_import = time.perf_counter() - t
    torch.manual_seed(0)
    t = time.perf_counter()
    model = RFDETRMedium(device=args.device)
    t_build = time.perf_counter() - t

    def predict():
        s = time.perf_counter()
        det = model.predict(image, threshold=RULES['threshold'], include_source_image=False)
        if args.device == 'mps':
            torch.mps.synchronize()
        return det, time.perf_counter() - s
    det, t_first = predict()
    warm = [predict()[1] for _ in range(args.warm)]

    # predict() returns COCO category ids (1 = person, 3 = car, ...), named through COCO_CLASSES.
    rows = [{'class': COCO_CLASSES.get(int(c), str(int(c))), 'cocoCategoryId': int(c), 'score': round(float(s), 4),
             'box': [round(float(v), 1) for v in b]}
            for b, s, c in zip(det.xyxy.tolist(), det.confidence.tolist(), det.class_id.tolist())]
    rows.sort(key=lambda r: (-r['score'], r['box']))
    W, H = image.size
    curb_x, curb_bands = right_curb(valid, world, tol)
    placed = []
    for i, r in enumerate(rows):
        p = place(r, valid, world, tol, W, H)
        p = {'id': i, **p, 'curbLaneCandidate': candidacy(p, curb_x)}
        placed.append(p)

    doc = {
        'schemaVersion': 1, 'sceneId': meta['id'],
        'generatedUtc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'generator': {'script': SCRIPT, 'sha256': sha256(__file__), 'python': platform.python_version(),
                      'torch': torch.__version__, 'numpy': np.__version__, 'pillow': metadata.version('pillow')},
        'inputs': {name: sha256(scene_dir / name) for name in ('scene.json', 'recorded.jpg', 'depth.bin')},
        'model': {'name': 'RF-DETR Medium', 'package': f"rfdetr {metadata.version('rfdetr')}", 'license': 'Apache-2.0',
                  'weightsFile': WEIGHTS_FILE, 'weightsUrl': WEIGHTS_URL, 'weightsSha256': weights_sha,
                  'weightsBytes': weights.stat().st_size, 'weightsLicense': 'Apache-2.0',
                  'labelSpace': 'COCO category ids'},
        'run': {'device': args.device, 'platform': f'{platform.system()} {platform.machine()}',
                'timingSeconds': {'weightsHash': round(t_hash, 3), 'import': round(t_import, 3),
                                  'modelBuild': round(t_build, 3), 'firstPredict': round(t_first, 3),
                                  'warmMedian': round(statistics.median(warm), 4) if warm else None},
                'timingNote': 'Wall clock in one process; functional record, not a serialized benchmark.'},
        'rules': {**RULES, 'bodyMinHeightMeters': tol,
                  'placement': 'measured depth only (src/geometry.mjs measuredPoint); world x right, y up, -z forward'},
        'rightCurb': {'status': 'measured' if curb_x is not None else 'unknown',
                      'lateralX': None if curb_x is None else round(curb_x, 3), 'bands': curb_bands},
        'detections': placed,
        'curbLaneCandidates': [p['id'] for p in placed if (p['curbLaneCandidate'] or {}).get('eligible')],
    }
    tmp = output.with_name(output.name + '.tmp')
    tmp.write_text(json.dumps(doc, indent=1) + '\n')
    os.replace(tmp, output)
    print(json.dumps(doc['run']['timingSeconds']), file=sys.stderr)
    print(f"{len(placed)} detections; curb-lane candidates {doc['curbLaneCandidates']}; wrote {output}", file=sys.stderr)
    print(sha256(output))


if __name__ == '__main__':
    main()
