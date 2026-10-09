#!/usr/bin/env python3
"""Fixed CPU image-only epipolar study; automatic matches are not ground truth.

Run from project root: python3 scripts/evaluate-epipolar-alignment.py
No image warp, runtime adoption, parameter search, or metric-scale inference.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import cv2
import numpy as np

CURRENT = '2018-10-19-09-30-39_2018-10-19-09-40-40-716'
PARAMS = dict(sift=dict(nfeatures=4000, nOctaveLayers=3, contrastThreshold=.04,
                       edgeThreshold=10, sigma=1.6), ratio=.75,
              fundamental=dict(method='USAC_MAGSAC', threshold_px=1., confidence=.999,
                               max_iterations=10000),
              homography=dict(method='RANSAC', threshold_px=1., confidence=.999,
                              max_iterations=10000), seed=42, uniform_pairs=20)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    return dict(count=int(v.size), mean=float(v.mean()), p05=float(np.percentile(v, 5)),
                p50=float(np.median(v)), p90=float(np.percentile(v, 90)),
                p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)),
                min=float(v.min()), max=float(v.max())) if v.size else dict(count=0)


def match_arrays(rows, count):
    indices = np.full((count, 2), -1, np.int32)
    distances = np.full((count, 2), np.inf, np.float64)
    for row in rows:
        for column, match in enumerate(row[:2]):
            indices[match.queryIdx, column] = match.trainIdx
            distances[match.queryIdx, column] = match.distance
    return indices, distances


def accept_matches(forward, fd, reverse, rd):
    # Strict ratio rejects 0/0 duplicate descriptors and exact boundary .75.
    left_ratio = (forward[:, 1] >= 0) & (fd[:, 0] < PARAMS['ratio'] * fd[:, 1])
    mutual = np.zeros(len(forward), bool)
    right_ratio = np.zeros(len(forward), bool)
    valid = forward[:, 0] >= 0
    rows = np.flatnonzero(valid)
    right = forward[rows, 0]
    mutual[rows] = reverse[right, 0] == rows
    right_ratio[rows] = (reverse[right, 1] >= 0) & (rd[right, 0] < PARAMS['ratio'] * rd[right, 1])
    return left_ratio & mutual & right_ratio, left_ratio, mutual, right_ratio


def epipolar(F, left, right):
    lh = np.column_stack([left, np.ones(len(left))])
    rh = np.column_stack([right, np.ones(len(right))])
    right_line, left_line = lh @ F.T, rh @ F
    error = np.sum(rh * right_line, axis=1)
    nr, nl = np.sum(right_line[:, :2] ** 2, axis=1), np.sum(left_line[:, :2] ** 2, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        symmetric = np.abs(error) * np.sqrt(.5 * (1/nr + 1/nl))
        sampson = np.abs(error) / np.sqrt(nr + nl)
        angle = np.degrees(np.arctan2(np.abs(right_line[:, 0]), np.abs(right_line[:, 1])))
    return symmetric, sampson, angle


def fundamental_diagnostics(F, left, right):
    def normalize(p):
        centered = p - p.mean(axis=0)
        return centered * np.sqrt(2) / np.mean(np.linalg.norm(centered, axis=1))
    a, b = normalize(left), normalize(right)
    x, y, u, v = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    design = np.column_stack([u*x, u*y, u, v*x, v*y, v, x, y, np.ones(len(x))])
    d = np.linalg.svd(design, compute_uv=False)
    f = np.linalg.svd(F, compute_uv=False)
    return dict(F_singular_values=f.tolist(), F_nonzero_condition=float(f[0]/f[1]),
                F_rank2_residual=float(f[2]/f[1]), normalized_design_singular_values=d.tolist(),
                design_null_gap=float(d[-2]/d[-1]),
                design_non_null_condition=float(d[0]/d[-2]))


def stats(left, right, depths, symmetric, sampson, angles, mask):
    dy, dx = left[:, 1]-right[:, 1], left[:, 0]-right[:, 0]
    return dict(count=int(mask.sum()), signed_vertical_px=distribution(dy[mask]),
                absolute_vertical_px=distribution(np.abs(dy[mask])),
                disparity_left_minus_right_px=distribution(dx[mask]),
                nonpositive_disparity_count=int((dx[mask] <= 0).sum()),
                symmetric_epipolar_px=distribution(symmetric[mask]),
                sampson_px=distribution(sampson[mask]),
                right_epipolar_line_angle_to_horizontal_deg=distribution(angles[mask]),
                sampled_source_reference_depth_m=distribution(depths[mask]))


def self_check():
    F = np.array([[0., 0, 0], [0, 0, -1], [0, 1, 0]])
    l, r = np.array([[20., 10], [40, 30]]), np.array([[15., 13], [35, 30]])
    symmetric, sampson, angle = epipolar(F, l, r)
    assert np.allclose(symmetric, [3, 0]) and np.allclose(sampson, [3/np.sqrt(2), 0])
    assert np.allclose(angle, 0)
    S = np.diag([.5, .5, 1.])
    Fs = np.linalg.inv(S).T @ F @ np.linalg.inv(S)
    assert np.allclose(epipolar(Fs, l*.5, r*.5)[0], symmetric*.5)
    i = np.array([[0, 1], [1, 0], [2, 0], [3, -1]])
    d = np.array([[.74, 1], [.75, 1], [0, 0], [0, np.inf]])
    accepted, *_ = accept_matches(i, d, i, d)
    assert accepted.tolist() == [True, False, False, False]
    good_d = np.tile([.5, 1.], (4, 1))
    reverse = i.copy()
    reverse[0, 0] = 1
    assert not accept_matches(i, good_d, reverse, good_d)[0][0]
    reverse_d = good_d.copy()
    reverse_d[0, 0] = .75
    assert not accept_matches(i, good_d, i, reverse_d)[0][0]
    assert distribution([]) == dict(count=0)


def overlay(left, right, points_l, points_r, accepted, inliers, title):
    h, w = left.shape[:2]
    canvas = np.zeros((h+55, 2*w, 3), np.uint8)
    canvas[55:, :w], canvas[55:, w:] = left, right
    # Bounded, evenly selected display only; all coordinates remain in CSV/NPZ.
    selected = np.flatnonzero(accepted)
    selected = selected[np.linspace(0, len(selected)-1, min(120, len(selected)), dtype=int)] if len(selected) else []
    for i in selected:
        color = (90, 230, 110) if inliers[i] else (50, 70, 240)
        p = tuple(np.rint(points_l[i]+[0, 55]).astype(int))
        q = tuple(np.rint(points_r[i]+[w, 55]).astype(int))
        cv2.line(canvas, p, q, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p, 3, color, 1)
        cv2.circle(canvas, q, 3, color, 1)
    cv2.putText(canvas, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    cv2.putText(canvas, 'Green F inlier / red F rejection; automatic matches, not truth. Up to 120 uniformly displayed.',
                (12, 44), cv2.FONT_HERSHEY_SIMPLEX, .45, (220, 220, 220), 1)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print('Self-check passed: residual coordinates/scaling and strict bidirectional matcher boundaries.')
        return
    data = Path('.local/data/drivingstereo')
    output = Path('.local/evaluation/epipolar-alignment')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'report.json').exists():
        raise ValueError('Existing result retained; move it explicitly before another run')
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    left_dir, right_dir, depth_dir = [data/'extracted'/p for p in
                                     ['left-image-half-size', 'right-image-half-size', 'depth-map-half-size']]
    matched = sorted({p.stem for p in left_dir.glob('*.jpg')} & {p.stem for p in right_dir.glob('*.jpg')})
    eligible = [s for s in matched if s.split('_')[0] == CURRENT.split('_')[0]]
    assert CURRENT in eligible
    uniform = np.linspace(0, len(eligible)-1, PARAMS['uniform_pairs'], dtype=int).tolist()
    selected = list(dict.fromkeys([CURRENT]+[eligible[i] for i in uniform]))
    assert len(selected) <= 21
    predeclared = dict(hypothesis='Supplied rectified image pairs support horizontal epipolar alignment across frames; residual/drift may undermine horizontal stereo preprocessing.',
                      policy='Fixed before scores; no dy/disparity filter, no image resize, no threshold tuning, no score-based sample selection.',
                      selected_indices=[eligible.index(s) for s in selected], uniform_indices=uniform,
                      sequence_pairs=len(eligible), parameters=PARAMS)
    (output/'predeclared.json').write_text(json.dumps(predeclared, indent=2)+'\n')
    shutil.copy2(__file__, output/Path(__file__).name)
    (output/'opencv-build.txt').write_text(cv2.getBuildInformation())
    inputs = []
    for path in [data/'calibration/2018-10-19-09-30-39.txt',
                 data/'calibration/2018-10-19-09-30-39-full-reference.txt',
                 data/'inspection/candidate-094-calibration-check.json',
                 Path('.local/evaluation/stereo-baseline/five-frame/report.json')]:
        inputs.append(dict(path=str(path), sha256=sha(path)))
    calibration = {}
    for kind, path in [('supplied_half', data/'calibration/2018-10-19-09-30-39.txt'),
                       ('full_reference', data/'calibration/2018-10-19-09-30-39-full-reference.txt')]:
        values = {k: np.fromstring(v, sep=' ') for k, v in
                  (line.split(':', 1) for line in path.read_text().splitlines())}
        S = np.diag([879/values['S_rect_101'][0], 400/values['S_rect_101'][1], 1])
        projections = [S @ values[k].reshape(3, 4) for k in ['P_rect_101', 'P_rect_103']]
        calibration[kind] = dict(resize=S.tolist(), projections=[p.tolist() for p in projections],
                                source_baseline_from_projection_m=float(-values['P_rect_103'][3]/values['P_rect_103'][0]),
                                translation_103_m=values['T_103'].tolist())
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), predeclared=predeclared,
                  environment=dict(python=sys.version, numpy=np.__version__, opencv=cv2.__version__,
                                   platform=platform.platform(), threads=cv2.getNumThreads(), opencl=cv2.ocl.useOpenCL()),
                  script_sha256=sha(Path(__file__)), command=sys.argv, inputs=inputs,
                  calibration_reference=calibration,
                  assumptions=dict(dimensions=[879, 400], camera_order='101 left, 103 right by original directories; not independently surveyed',
                                   signed_disparity='x_left - x_right; positive expected under source horizontal convention but never filtered',
                                   coordinates='Original JPEG pixel coordinates; keypoints subpixel; nearest rounded left pixel for sparse source depth raw/256, zero unsupported',
                                   symmetric_residual='sqrt((left point-line distance squared + right point-line distance squared)/2)',
                                   sampson='abs(xR^T F xL)/sqrt(norm(F xL)[0:2]^2+norm(F^T xR)[0:2]^2)',
                                   masks='Independent robust F fit on all mutually ratio-accepted matches; inlier mask is model-consistency evidence, not correctness',
                                   homography='Independent H RANSAC on identical accepted matches; support overlap diagnoses planar/low-parallax ambiguity, not proof of degeneracy',
                                   ranges='Sparse reference Z bins; nearest samples can cross surface boundaries and are not depth truth for SIFT features'),
                  frames=[])
    sift = cv2.SIFT_create(**PARAMS['sift'])
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    views, pooled = [], []
    started = time.perf_counter()
    for stem in selected:
        index = eligible.index(stem)
        cv2.setRNGSeed(PARAMS['seed']+index)
        paths = dict(left=left_dir/(stem+'.jpg'), right=right_dir/(stem+'.jpg'), depth=depth_dir/(stem+'.png'))
        left, right = [cv2.imread(str(paths[k])) for k in ['left', 'right']]
        depth = cv2.imread(str(paths['depth']), cv2.IMREAD_UNCHANGED)
        assert left.shape == right.shape == (400, 879, 3) and depth.shape == (400, 879) and depth.dtype == np.uint16
        folder = output/stem
        folder.mkdir()
        key_l, desc_l = sift.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None)
        key_r, desc_r = sift.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None)
        points_l, all_r = np.array([k.pt for k in key_l]), np.array([k.pt for k in key_r])
        forward, fd = match_arrays(matcher.knnMatch(desc_l, desc_r, k=2), len(key_l))
        reverse, rd = match_arrays(matcher.knnMatch(desc_r, desc_l, k=2), len(key_r))
        accepted, ratio_l, mutual, ratio_r = accept_matches(forward, fd, reverse, rd)
        points_r = np.full_like(points_l, np.nan)
        has_neighbor = forward[:, 0] >= 0
        points_r[has_neighbor] = all_r[forward[has_neighbor, 0]]
        a, b = points_l[accepted], points_r[accepted]
        if len(a) < 8:
            raise ValueError(f'{stem}: fewer than eight descriptor-accepted matches; no F evidence')
        F, mask_f = cv2.findFundamentalMat(a, b, cv2.USAC_MAGSAC, 1., .999, 10000)
        H, mask_h = cv2.findHomography(a, b, cv2.RANSAC, 1., maxIters=10000, confidence=.999)
        if F is None or F.shape != (3, 3):
            raise ValueError(f'{stem}: F fit unavailable')
        inliers, h_inliers = np.zeros(len(key_l), bool), np.zeros(len(key_l), bool)
        inliers[accepted] = mask_f.ravel().astype(bool)
        if mask_h is not None:
            h_inliers[accepted] = mask_h.ravel().astype(bool)
        symmetric, sampson, angle = epipolar(F, points_l, points_r)
        xy = np.rint(points_l).astype(int)
        depths = depth[xy[:, 1], xy[:, 0]].astype(float)/256
        depths[depths == 0] = np.nan
        masks = dict(raw_nearest=has_neighbor, mutual_ratio=accepted, F_inliers=inliers,
                     F_rejected=accepted & ~inliers, F_inliers_without_H=inliers & ~h_inliers)
        frame = dict(stem=stem, sequence_index=index, inputs={k: dict(path=str(p), sha256=sha(p)) for k, p in paths.items()},
                     keypoints_left=len(key_l), keypoints_right=len(key_r),
                     rejection_counts=dict(left_ratio=int((~ratio_l).sum()), nonmutual=int((~mutual).sum()),
                                           right_ratio=int((~ratio_r).sum()), policy='Counts overlap; final retained set is intersection'),
                     F=F.tolist(), H=H.tolist() if H is not None else None,
                     diagnostics=fundamental_diagnostics(F, a[mask_f.ravel().astype(bool)], b[mask_f.ravel().astype(bool)]),
                     homography_inliers=int(h_inliers.sum()), F_H_overlap=int((inliers & h_inliers).sum()),
                     masks={name: stats(points_l, points_r, depths, symmetric, sampson, angle, mask) for name, mask in masks.items()},
                     spatial=[], ranges=[])
        for row in range(3):
            for col in range(4):
                mask = inliers & (points_l[:, 0] >= col*879/4) & (points_l[:, 0] < (col+1)*879/4) & (points_l[:, 1] >= row*400/3) & (points_l[:, 1] < (row+1)*400/3)
                frame['spatial'].append(dict(row=row, column=col, **stats(points_l, points_r, depths, symmetric, sampson, angle, mask)))
        for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 80), (80, np.inf)]:
            mask = inliers & (depths >= lo) & (depths < hi)
            frame['ranges'].append(dict(lower_m=lo, upper_m=hi if np.isfinite(hi) else None, **stats(points_l, points_r, depths, symmetric, sampson, angle, mask)))
        if inliers.sum() >= 8:
            x = np.column_stack([np.ones(inliers.sum()), points_l[inliers, 0]/879, points_l[inliers, 1]/400])
            beta = np.linalg.lstsq(x, (points_l-points_r)[inliers, 1], rcond=None)[0]
            frame['vertical_spatial_drift_OLS_px'] = dict(intercept=float(beta[0]), across_width=float(beta[1]), across_height=float(beta[2]), limitation='Descriptive fit to F inliers; correlated features and scene motion preclude causal calibration attribution')
        np.savez_compressed(folder/'matches.npz', left_keypoints=points_l, right_keypoints=all_r,
                            forward_indices=forward, forward_distances=fd, reverse_indices=reverse,
                            reverse_distances=rd, accepted=accepted, F_inliers=inliers, H_inliers=h_inliers,
                            F=F, H=H, symmetric_px=symmetric, sampson_px=sampson, reference_depth_m=depths)
        with (folder/'matches.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['left_index', 'right_index', 'left_x', 'left_y', 'right_x', 'right_y', 'distance_1', 'distance_2',
                             'left_ratio_pass', 'mutual_nearest', 'right_ratio_pass', 'accepted', 'F_inlier', 'H_inlier',
                             'dy_left_minus_right', 'dx_left_minus_right', 'symmetric_px', 'sampson_px', 'reference_depth_m'])
            for i in range(len(key_l)):
                writer.writerow([i, int(forward[i, 0]), *points_l[i], *points_r[i], *fd[i],
                                 int(ratio_l[i]), int(mutual[i]), int(ratio_r[i]), int(accepted[i]), int(inliers[i]), int(h_inliers[i]),
                                 *(points_l[i]-points_r[i])[::-1], symmetric[i], sampson[i], depths[i]])
        title = f'Index {index:03d}: matches {accepted.sum()}, F {inliers.sum()}, |dy| p95 {frame["masks"]["F_inliers"]["absolute_vertical_px"]["p95"]:.3f}px'
        view = overlay(left, right, points_l, points_r, accepted, inliers, title)
        views.append(view)
        pooled.append((points_l, points_r, depths, symmetric, sampson, angle, accepted, inliers))
        report['frames'].append(frame)
        print(title, flush=True)
    columns = [np.concatenate([row[i] for row in pooled]) for i in range(8)]
    l, r, z, symmetric, sampson, angle, accepted, inliers = columns
    report['aggregate'] = {name: stats(l, r, z, symmetric, sampson, angle, mask) for name, mask in
                           [('raw_nearest', np.isfinite(r[:, 0])), ('mutual_ratio', accepted), ('F_inliers', inliers), ('F_rejected', accepted & ~inliers)]}
    metrics = [f['masks']['F_inliers']['absolute_vertical_px']['p95'] for f in report['frames']]
    medians = [f['masks']['F_inliers']['signed_vertical_px']['p50'] for f in report['frames']]
    report['between_frame'] = dict(inlier_abs_dy_p95=distribution(metrics), inlier_signed_dy_median=distribution(medians))
    strongest, difficult = int(np.argmin(metrics)), int(np.argmax(metrics))
    weak = int(np.argmin([f['masks']['F_inliers']['count'] for f in report['frames']]))
    report['visuals'] = dict(current=selected[0], strongest=selected[strongest], difficult=selected[difficult],
                            weak_support=selected[weak],
                            selection='Visualization only: min/max F-inlier |dy| p95 and minimum F support, never sample selection or exclusion')
    for name, i in [('current', 0), ('strongest', strongest), ('difficult', difficult), ('weak-support', weak)]:
        cv2.imwrite(str(output/(name+'-matches.png')), views[i])
    thumbs = [cv2.resize(v, (704, 182)) for v in views]
    sheet = np.zeros((7*182, 3*704, 3), np.uint8)
    for i, thumb in enumerate(thumbs):
        sheet[(i//3)*182:(i//3+1)*182, (i%3)*704:(i%3+1)*704] = thumb
    cv2.imwrite(str(output/'contact-sheet.png'), sheet)
    report['elapsed_seconds'] = time.perf_counter()-started
    report['limitations'] = ['Automatic descriptor/F matches are not correspondence truth; F inliers and low fitted residual are conditional, partly circular model evidence.',
                             'One sunny development sequence, correlated 21 frames; no independent sites, California/held-out generalization or confidence interval.',
                             'F determines epipolar geometry up to scale and cannot establish metric baseline or resolve the supplied half projection translation discrepancy.',
                             'Homography overlap, uneven coverage, moving objects, repeated texture, and low parallax can weaken F identification.',
                             'No images warped or calibration repaired; low vertical mismatch does not certify source rectification or dense stereo depth accuracy.',
                             'Sparse depth sampled at rounded SIFT coordinates only stratifies evidence; unsupported matches retained, not scored as depth errors.']
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Saved {len(selected)} pairs; CPU {report["elapsed_seconds"]:.2f}s', flush=True)


if __name__ == '__main__':
    main()
