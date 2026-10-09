#!/usr/bin/env python3
"""Fixed CPU StereoSGBM development baseline; no tuning or visibility claims.

Run: python3 scripts/evaluate-stereo-baseline.py [--count 1..5] [--output PATH]
Raw OpenCV disparity is retained as int16, with a separate float32 /16 copy.
"""
import argparse
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
import rectified_calibration

FIRST = '2018-10-19-09-30-39_2018-10-19-09-40-40-716'
PARAMS = dict(minDisparity=0, numDisparities=128, blockSize=5,
              P1=8 * 5 * 5, P2=32 * 5 * 5, disp12MaxDiff=1,
              preFilterCap=31, uniquenessRatio=10, speckleWindowSize=100,
              speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
RANGES = [(0, 10), (10, 20), (20, 40), (40, 80), (80, None)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def distribution(values):
    if not values.size:
        return None
    return dict(mean=float(np.mean(values)),
                p50=float(np.percentile(values, 50)),
                p90=float(np.percentile(values, 90)),
                p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99)), max=float(np.max(values)))


def errors(predicted, expected, valid, bad_rules):
    signed = predicted[valid] - expected[valid]
    absolute = np.abs(signed)
    count = int(valid.sum())
    total = int(expected.size)
    result = dict(reference_points=total, valid_predictions=count,
                  invalid_predictions=total-count,
                  prediction_coverage_on_reference=ratio(count, total),
                  absolute_error=distribution(absolute),
                  signed_error_mean=float(signed.mean()) if count else None,
                  signed_error_median=float(np.median(signed)) if count else None,
                  rmse=float(np.sqrt(np.mean(signed**2))) if count else None,
                  bad_pixel_rates={})
    for name, rule in bad_rules.items():
        bad = int(rule(absolute, expected[valid]).sum())
        result['bad_pixel_rates'][name] = dict(
            bad_among_valid_count=bad, valid_prediction_denominator=count,
            bad_rate_on_valid=ratio(bad, count),
            invalid_or_bad_count=total-count+bad, all_reference_denominator=total,
            invalid_or_bad_rate_on_all_reference=ratio(total-count+bad, total))
    return result


def metrics(predicted_disparity, expected_disparity, depth, offset, numerator):
    # Zero disparity can be an accepted SGBM result, but cannot yield finite Z.
    disparity_valid = predicted_disparity > PARAMS['minDisparity'] - 1
    depth_valid = disparity_valid & ((predicted_disparity-offset) > 0)
    predicted_depth = np.full(depth.shape, np.nan, dtype=np.float64)
    predicted_depth[depth_valid] = numerator / (predicted_disparity[depth_valid]-offset)
    d = errors(predicted_disparity, expected_disparity, disparity_valid, {
        'abs_gt_1px': lambda a, e: a > 1,
        'abs_gt_2px': lambda a, e: a > 2,
        'abs_gt_3px': lambda a, e: a > 3,
        'abs_gt_3px_and_relative_gt_5pct': lambda a, e: (a > 3) & (a > .05*np.abs(e))})
    z = errors(predicted_depth, depth, depth_valid, {
        'abs_gt_1m': lambda a, e: a > 1,
        'abs_gt_2m': lambda a, e: a > 2,
        'relative_gt_10pct': lambda a, e: a > .1*e,
        'relative_gt_25pct': lambda a, e: a > .25*e})
    z['absolute_relative_error'] = distribution(
        np.abs(predicted_depth[depth_valid]-depth[depth_valid])/depth[depth_valid])
    return dict(disparity_px=d, depth_m=z)


def projection(path, width, height):
    left, right, size, scale = rectified_calibration.load_rectified_pair(path, width, height)
    numerator = float(left[0, 3]-right[0, 3])
    offset = float(left[0, 2]-right[0, 2])
    if numerator <= 0:
        raise ValueError('Projection must produce positive stereo baseline')
    return dict(source=str(path.resolve()), source_sha256=sha(path),
                source_rectified_size=size.tolist(), scale=[*scale.tolist(), 1.],
                left_projection=left.tolist(), right_projection=right.tolist(),
                disparity_numerator_px_m=numerator, principal_point_offset_px=offset,
                implied_baseline_m=numerator/float(left[0, 0]), provisional=True)


def self_check():
    # Canonical projection formula includes nonzero principal-point offset.
    offset, numerator = 4., 500.
    depth = np.array([10., 20., 30.])
    expected = offset+numerator/depth
    predicted = np.array([expected[0], expected[1]+4, -1])
    result = metrics(predicted, expected, depth, offset, numerator)
    d = result['disparity_px']
    assert d['valid_predictions'] == 2 and d['invalid_predictions'] == 1
    assert d['absolute_error']['mean'] == 2
    assert d['bad_pixel_rates']['abs_gt_3px']['invalid_or_bad_rate_on_all_reference'] == 2/3
    assert result['depth_m']['valid_predictions'] == 2
    assert metrics(np.array([0.]), np.array([1.]), np.array([10.]), 0, 10)['depth_m']['invalid_predictions'] == 1
    assert errors(np.array([]), np.array([]), np.array([], dtype=bool), {})['rmse'] is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('.local/data/drivingstereo'))
    parser.add_argument('--output', type=Path, default=Path('.local/evaluation/stereo-baseline'))
    parser.add_argument('--count', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print('Self-check passed: offset, invalid predictions, denominators, empty ranges.')
        return
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    started = time.perf_counter()
    left_dir = args.data/'extracted/left-image-half-size'
    right_dir = args.data/'extracted/right-image-half-size'
    depth_dir = args.data/'extracted/depth-map-half-size'
    sets = [{p.stem for p in directory.glob('*'+suffix)} for directory, suffix in
            [(left_dir, '.jpg'), (right_dir, '.jpg'), (depth_dir, '.png')]]
    matched = sorted(set.intersection(*sets))
    eligible = [s for s in matched if s.split('_')[0] == FIRST.split('_')[0]]
    if FIRST not in eligible:
        raise ValueError('Requested first frame lacks matched triplet')
    candidates = [eligible[i] for i in np.linspace(0, len(eligible)-1, 4, dtype=int)]
    selected = list(dict.fromkeys([FIRST]+candidates))[:args.count]
    if len(selected) != args.count:
        raise ValueError('Insufficient deterministic matched frames')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/'evaluate-stereo-baseline.py')
    (args.output/'opencv-build.txt').write_text(cv2.getBuildInformation())
    matcher = cv2.StereoSGBM_create(**PARAMS)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  evidence_class='development data; single sunny sequence; not held-out; not California',
                  selection='Prepared frame first, then sorted matched sequence indices 0,166,332,499; no score-based selection',
                  matched_counts=[len(s) for s in sets], matched_triplets=len(matched),
                  command=sys.argv, script_sha256=sha(Path(__file__)),
                  calibration_helper_sha256=sha(Path(rectified_calibration.__file__)),
                  environment=dict(python=sys.version, opencv=cv2.__version__, numpy=np.__version__,
                                   platform=platform.platform(), machine=platform.machine(),
                                   opencv_threads=cv2.getNumThreads(), opencl=cv2.ocl.useOpenCL()),
                  parameters=PARAMS, parameter_policy='Fixed before observing scores; grayscale, no parameter search or test-set tuning',
                  definitions=dict(reference='Every source depth uint16 >0, depth=raw/256; no range filtering',
                                   prediction_valid='SGBM disparity >-1px including zero; finite depth additionally requires disparity minus principal offset >0',
                                   expected_disparity='cx_left-cx_right + (P_left[0,3]-P_right[0,3])/reference_Z',
                                   depth='(P_left[0,3]-P_right[0,3])/(predicted_disparity-(cx_left-cx_right))',
                                   errors='Only valid predictions at valid reference points; invalid counts/coverage and invalid-or-bad rates retain every valid reference point',
                                   d1='abs error >3px AND >5% of abs expected disparity; not an official benchmark submission',
                                   pooling='All reference pixels from selected frames pooled; correlated pixels and frames are not independent sites',
                                   ranges='Reference depth bins [lower,upper); final [80,infinity)'),
                  limitations=['Sparse model-guided filtered LiDAR reference, not surveyed scale or independent dense truth.',
                               'Two explicit provisional calibration conventions; scores do not resolve surveyed metric scale.',
                               'Image depth/disparity metrics do not measure whole-segment visibility accuracy.',
                               'Only one sunny sequence, five development frames; no California locations or held-out generalization.',
                               'No predicted pixel outside reference support is scored; density there does not establish accuracy.',
                               'Fixed 0..127px search excludes larger disparities; left search border/occlusions/textureless areas can be invalid.'],
                  frames=[], aggregate={})
    pooled = {}
    for stem in selected:
        paths = dict(left=left_dir/(stem+'.jpg'), right=right_dir/(stem+'.jpg'), depth=depth_dir/(stem+'.png'))
        left, right = [cv2.imread(str(paths[k]), cv2.IMREAD_GRAYSCALE) for k in ['left', 'right']]
        raw_depth = cv2.imread(str(paths['depth']), cv2.IMREAD_UNCHANGED)
        if left is None or right is None or raw_depth is None or left.shape != right.shape or left.shape != raw_depth.shape or raw_depth.dtype != np.uint16:
            raise ValueError('Source image/depth dimensions or uint16 encoding mismatch')
        height, width = left.shape
        folder = args.output/stem
        folder.mkdir()
        for name, path in paths.items():
            shutil.copy2(path, folder/(name+path.suffix))
        begin = time.perf_counter()
        raw_disparity = matcher.compute(left, right)
        runtime = time.perf_counter()-begin
        disparity = raw_disparity.astype(np.float32)/16
        np.save(folder/'disparity-int16.npy', raw_disparity)
        np.save(folder/'disparity-px.npy', disparity)
        valid_ref = raw_depth > 0
        depth = raw_depth.astype(np.float64)/256
        frame = dict(stem=stem, calibration_identifier=stem.split('_')[0], width=width, height=height,
                     input_hashes={k:dict(path=str(p.resolve()), sha256=sha(p)) for k,p in paths.items()},
                     compute_seconds=runtime, image_pixels=int(depth.size),
                     reference_points=int(valid_ref.sum()), reference_density=float(valid_ref.mean()),
                     disparity_valid_pixels=int((raw_disparity > -16).sum()),
                     disparity_valid_density=float((raw_disparity > -16).mean()),
                     positive_disparity_density=float((raw_disparity > 0).mean()), conventions={})
        for name, suffix in [('scaled_full', '-full-reference.txt'), ('supplied_half', '.txt')]:
            calibration_path = args.data/'calibration'/(stem.split('_')[0]+suffix)
            calibration = projection(calibration_path, width, height)
            shutil.copy2(calibration_path, folder/(name+'-calibration.txt'))
            offset, numerator = calibration['principal_point_offset_px'], calibration['disparity_numerator_px_m']
            expected = offset+numerator/depth[valid_ref]
            pred = disparity[valid_ref].astype(np.float64)
            valid_depth = (disparity > -1) & ((disparity-offset) > 0)
            predicted_depth = np.full(depth.shape, np.nan, dtype=np.float32)
            predicted_depth[valid_depth] = numerator/(disparity[valid_depth]-offset)
            np.save(folder/(name+'-depth-m.npy'), predicted_depth)
            scored = metrics(pred, expected, depth[valid_ref], offset, numerator)
            scored['calibration'] = calibration
            scored['expected_disparity_outside_search_count'] = int(((expected < 0) | (expected >= 128)).sum())
            scored['by_reference_range_m'] = {}
            for lo,hi in RANGES:
                mask = (depth[valid_ref] >= lo) & (depth[valid_ref] < (hi if hi else np.inf))
                scored['by_reference_range_m'][f'{lo}-{hi if hi else "inf"}'] = metrics(pred[mask], expected[mask], depth[valid_ref][mask], offset, numerator)
            frame['conventions'][name] = scored
            pooled.setdefault(name, []).append((pred, expected, depth[valid_ref], offset, numerator))
        frame['output_hashes'] = {p.name:sha(p) for p in sorted(folder.iterdir())}
        report['frames'].append(frame)
        print(f'{stem}: {runtime:.3f}s; reference {frame["reference_points"]}; disparity density {frame["disparity_valid_density"]:.3%}')
    for name, items in pooled.items():
        if any((item[3], item[4]) != (items[0][3], items[0][4]) for item in items):
            raise ValueError('Pooling requires identical calibration geometry')
        pred, expected, depth = [np.concatenate([item[i] for item in items]) for i in range(3)]
        offset, numerator = items[0][3:]
        report['aggregate'][name] = metrics(pred, expected, depth, offset, numerator)
        report['aggregate'][name]['by_reference_range_m'] = {}
        for lo,hi in RANGES:
            mask = (depth >= lo) & (depth < (hi if hi else np.inf))
            report['aggregate'][name]['by_reference_range_m'][f'{lo}-{hi if hi else "inf"}'] = metrics(pred[mask], expected[mask], depth[mask], offset, numerator)
    report['compute_seconds_total'] = sum(f['compute_seconds'] for f in report['frames'])
    report['wall_seconds_before_report_write'] = time.perf_counter()-started
    report['run_artifact_hashes'] = {p.name:sha(p) for p in [args.output/'evaluate-stereo-baseline.py', args.output/'opencv-build.txt']}
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Report: {args.output}/report.json; compute {report["compute_seconds_total"]:.3f}s')


if __name__ == '__main__':
    main()
