#!/usr/bin/env python3
"""Development-only reciprocal stereo support experiment; saved left + real CPU right.

Run: python3 scripts/evaluate-stereo-consistency.py [--self-check]
An existing output is never overwritten. All thresholds are declared before scores.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import cv2
import numpy as np

BASE_SCRIPT = Path(__file__).resolve().with_name('evaluate-stereo-baseline.py')
spec = importlib.util.spec_from_file_location('baseline', BASE_SCRIPT)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
THRESHOLDS = (0.5, 1.0, 2.0)
MAX_DEPTH_M = 80.0


def reciprocal(left, right, left_min, right_min, count):
    """Linear right sample at x-dL; require every nonzero-weight endpoint valid."""
    height, width = left.shape
    left_valid = np.isfinite(left) & (left >= left_min) & (left < left_min + count)
    right_valid = np.isfinite(right) & (right >= right_min) & (right < right_min + count)
    x_right = np.arange(width)[None, :] - np.where(left_valid, left, 0)
    in_bounds = left_valid & (x_right >= 0) & (x_right <= width - 1)
    low = np.clip(np.floor(x_right).astype(np.int32), 0, width - 1)
    high = np.minimum(low + 1, width - 1)
    weight = x_right - low
    rows = np.arange(height)[:, None]
    usable = in_bounds & right_valid[rows, low] & ((weight == 0) | right_valid[rows, high])
    # Avoid 0*NaN at exact integer samples without inventing a missing endpoint.
    sampled = right[rows, low].copy()
    fractional = usable & (weight != 0)
    sampled[fractional] += weight[fractional] * (right[rows, high][fractional] - sampled[fractional])
    residual = np.full(left.shape, np.nan, dtype=np.float32)
    residual[usable] = np.abs(left[usable] + sampled[usable])
    return left_valid, in_bounds, usable, residual


def score(pred_disparity, expected_disparity, predicted_depth, reference, dmask, zmask, base_zmask):
    d = baseline.errors(pred_disparity, expected_disparity, dmask, {
        'abs_gt_1px': lambda a, e: a > 1,
        'abs_gt_2px': lambda a, e: a > 2,
        'abs_gt_3px': lambda a, e: a > 3,
        'abs_gt_3px_and_relative_gt_5pct': lambda a, e: (a > 3) & (a > .05*np.abs(e))})
    z = baseline.errors(predicted_depth, reference, zmask, {
        'abs_gt_1m': lambda a, e: a > 1,
        'abs_gt_2m': lambda a, e: a > 2,
        'relative_gt_10pct': lambda a, e: a > .1*e,
        'relative_gt_25pct': lambda a, e: a > .25*e})
    z['absolute_relative_error'] = baseline.distribution(
        np.abs(predicted_depth[zmask]-reference[zmask])/reference[zmask])
    total, valid = int(reference.size), int(zmask.sum())
    rejected = base_zmask & ~zmask
    signed = predicted_depth - reference
    z['severe_overestimates'] = {}
    for threshold in (10, 100):
        severe = base_zmask & (signed > threshold)
        retained = int((severe & zmask).sum())
        removed = int((severe & rejected).sum())
        z['severe_overestimates'][f'positive_error_gt_{threshold}m'] = dict(
            retained_count=retained, valid_prediction_denominator=valid,
            rate_among_valid=baseline.ratio(retained, valid),
            all_reference_denominator=total, rate_on_all_reference=baseline.ratio(retained, total),
            invalid_or_severe_count=total-valid+retained,
            invalid_or_severe_rate_on_all_reference=baseline.ratio(total-valid+retained, total),
            baseline_severe_count=int(severe.sum()), rejected_baseline_severe_count=removed,
            fraction_baseline_severe_rejected=baseline.ratio(removed, int(severe.sum())))
    z['support_loss_from_saved_left'] = dict(
        saved_left_valid_depth_count=int(base_zmask.sum()), rejected_valid_depth_count=int(rejected.sum()),
        retained_valid_depth_count=valid,
        rejected_fraction_of_saved_left_valid=baseline.ratio(int(rejected.sum()), int(base_zmask.sum())),
        rejected_saved_left_absolute_error_le_2m_count=int((rejected & (np.abs(signed) <= 2)).sum()))
    assert total == valid + z['invalid_predictions']
    return dict(disparity_px=d, depth_m=z)


def self_check():
    # Raw OpenCV fixed-point scaling, sign, fractional interpolation, integer
    # endpoints, search upper bound, invalid sentinel, nonfinite and image edge.
    left = np.array([[0, 16, 24, 16, 16, -16, 2048, 16]], np.int16).astype(np.float32)/16
    right = np.array([[-16, -32, -16, -2048, -16, -16, -16, -16]], np.int16).astype(np.float32)/16
    lv, bounds, usable, residual = reciprocal(left, right, 0, -127, 128)
    assert np.array_equal(lv, [[True, True, True, True, True, False, False, True]])
    assert np.isclose(residual[0, 1], 0) and np.isclose(residual[0, 2], 0)
    assert usable[0, 3] and not usable[0, 4]  # exact integer needs no invalid next neighbor
    assert np.array_equal(usable & (residual <= 0.5), [[False, True, True, True, False, False, False, True]])
    _, bounds, usable, _ = reciprocal(np.array([[1., np.nan, 0.]]), np.array([[0., -128., np.nan]]), 0, -127, 128)
    assert not bounds[0, 0] and not bounds[0, 1] and not usable.any()
    # Invalid depth remains invalid even when zero disparity is consistent.
    reference = np.array([10., 20., 30.])
    pred = np.array([12., 130., np.nan])
    result = score(np.array([1., 2., 0.]), np.array([1., 2., 3.]), pred, reference,
                   np.array([True, False, True]), np.array([True, False, False]), np.array([True, True, False]))
    z = result['depth_m']
    assert z['valid_predictions'] == 1 and z['invalid_predictions'] == 2
    assert z['bad_pixel_rates']['abs_gt_1m']['invalid_or_bad_count'] == 3
    assert z['severe_overestimates']['positive_error_gt_100m']['rejected_baseline_severe_count'] == 1
    assert score(np.array([0.]), np.array([1.]), np.array([np.nan]), np.array([10.]),
                 np.array([True]), np.array([False]), np.array([False]))['depth_m']['absolute_error'] is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('.local/evaluation/stereo-baseline/five-frame'))
    parser.add_argument('--output', type=Path, default=Path('.local/evaluation/stereo-consistency'))
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print('Self-check passed: scaling, signs, interpolation, bounds, invalids and denominators.')
        return
    started = time.perf_counter()
    old_path = args.baseline/'report.json'
    old = json.loads(old_path.read_text())
    if len(old['frames']) != 5 or old['parameters'] != baseline.PARAMS or old['script_sha256'] != baseline.sha(BASE_SCRIPT):
        raise ValueError('Requires the retained five-frame baseline with identical fixed source/parameters')
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    left_matcher = cv2.StereoSGBM_create(**baseline.PARAMS)  # configuration only; never compute left
    right_min = -baseline.PARAMS['minDisparity'] - baseline.PARAMS['numDisparities'] + 1
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'createRightMatcher'):
        matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        engine = 'cv2.ximgproc.createRightMatcher; configuration cloned from fixed StereoSGBM'
    else:
        matcher = cv2.StereoSGBM_create(**(baseline.PARAMS | {'minDisparity': right_min}))
        engine = 'StereoSGBM mirrored right search: -left_min-numDisparities+1'
    if matcher.getMinDisparity() != right_min or matcher.getNumDisparities() != baseline.PARAMS['numDisparities']:
        raise ValueError('Unexpected right matcher search domain')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/'evaluate-stereo-consistency.py')
    matcher.save(str(args.output/'right-matcher.yml'))
    (args.output/'opencv-build.txt').write_text(cv2.getBuildInformation())
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Reciprocal left-right consistency can reduce severe positive depth errors from bad correspondences, at measured loss of usable support.',
        declared_before_scoring=True, primary_consistency_threshold_px=1.0,
        development_sensitivity_thresholds_px=[0.5, 2.0], maximum_depth_support_m=MAX_DEPTH_M,
        threshold_policy='Conventional 1px primary; sensitivity is exploratory, not tuned or release gates. Existing 80m support limit applied separately; no threshold chosen from scores.',
        command=sys.argv, baseline_report=dict(path=str(old_path.resolve()), sha256=baseline.sha(old_path)),
        baseline_script=dict(path=str(BASE_SCRIPT), sha256=baseline.sha(BASE_SCRIPT)),
        script_sha256=baseline.sha(Path(__file__)), fixed_left_parameters=baseline.PARAMS,
        right_engine=engine, right_min_disparity=right_min,
        definitions=dict(
            left_disparity='Retained int16 disparity divided by 16; no left inference rerun.',
            right_disparity='Actual matcher.compute(right_gray,left_gray), raw int16/16. Right disparity xR-xL has opposite sign.',
            search_support='Finite disparities in [minDisparity,minDisparity+numDisparities). Invalid sentinel (min-1) excluded; zero may be valid disparity.',
            reciprocal='xR=xL-dL; abs(dL+dR(xR)) <= threshold. Linear horizontal sampling at subpixel xR requires both endpoints valid for nonzero weights; exact integers need one endpoint; no nearest rounding or extrapolation.',
            finite_depth='Z=numerator/(dL-cx_offset), requiring search support, finite Z and dL-cx_offset>0. Zero disparity cannot yield depth when offset=0.',
            max_depth='Retain finite depth <=80m; reject larger values without clipping/filling. All source reference points remain in denominator, including reference beyond 80m.',
            reference=old['definitions']['reference'], calibration='Both provisional conventions from retained source calibration files via baseline.projection.',
            pooling='Pixel-weighted pooling across the original five frames of one development sequence; neither independent sites nor held-out data.',
            severe='Signed predicted depth minus reference >10m or >100m. Report retained, rejected, valid denominators and all-reference invalid-or-severe.',
            masks='Compressed NPZ contains full-image boolean disparity/depth support for every variant; NaN residual denotes unavailable reciprocal evidence.'),
        environment=dict(python=sys.version, opencv=cv2.__version__, numpy=np.__version__, platform=platform.platform(),
                         machine=platform.machine(), opencv_threads=cv2.getNumThreads(), opencl=cv2.ocl.useOpenCL()),
        limitations=old['limitations']+[
            'Reciprocal support is a correspondence heuristic; shared systematic failures may be consistent.',
            'Dropping support can reduce retained error while worsening all-reference invalid-or-bad rate; all-Unknown is never improved accuracy.',
            'No ray, free-space, geometric visibility, parking-safety or release-gate claim; no application data replaced.',
            'Right-only runtime excludes saved-left inference; combined inference shown only as arithmetic sum with historical left runtime, not a joint benchmark.'])
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2, allow_nan=False)+'\n')
    report = dict(protocol=protocol, frames=[], aggregate={})
    pooled = {}
    for old_frame in old['frames']:
        stem = old_frame['stem']
        source, folder = args.baseline/stem, args.output/stem
        inputs = [source/name for name in ['left.jpg', 'right.jpg', 'depth.png', 'disparity-int16.npy', 'disparity-px.npy', 'scaled_full-calibration.txt', 'supplied_half-calibration.txt']]
        for path in inputs:
            if baseline.sha(path) != old_frame['output_hashes'][path.name]:
                raise ValueError(f'Retained input hash mismatch: {path}')
        left, right = [cv2.imread(str(source/(side+'.jpg')), cv2.IMREAD_GRAYSCALE) for side in ['left', 'right']]
        raw_depth = cv2.imread(str(source/'depth.png'), cv2.IMREAD_UNCHANGED)
        raw_left = np.load(source/'disparity-int16.npy', allow_pickle=False)
        if left is None or right is None or raw_depth is None or left.shape != right.shape or left.shape != raw_depth.shape or left.shape != raw_left.shape or raw_depth.dtype != np.uint16 or raw_left.dtype != np.int16:
            raise ValueError('Retained image/disparity/reference shape or dtype mismatch')
        dl = raw_left.astype(np.float32)/16
        if not np.array_equal(dl, np.load(source/'disparity-px.npy', allow_pickle=False)):
            raise ValueError('Saved left fixed-point scaling mismatch')
        begin = time.perf_counter()
        raw_right = matcher.compute(right, left)
        seconds = time.perf_counter()-begin
        dr = raw_right.astype(np.float32)/16
        lv, bounds, usable, residual = reciprocal(dl, dr, baseline.PARAMS['minDisparity'], right_min, baseline.PARAMS['numDisparities'])
        masks = {'saved_left': lv}
        for t in THRESHOLDS:
            masks[f'consistency_{t:g}px'] = usable & (residual <= t)
        folder.mkdir()
        np.save(folder/'right-disparity-int16.npy', raw_right)
        np.save(folder/'right-disparity-px.npy', dr)
        np.save(folder/'reciprocal-residual-px.npy', residual)
        valid_ref = raw_depth > 0
        reference = raw_depth[valid_ref].astype(np.float64)/256
        frame = dict(stem=stem, prepared_frame=stem==baseline.FIRST,
            width=left.shape[1], height=left.shape[0], reference_points=int(valid_ref.sum()),
            input_hashes={p.name:dict(path=str(p.resolve()), sha256=baseline.sha(p)) for p in inputs},
            right_compute_seconds=seconds, saved_left_historical_compute_seconds=old_frame['compute_seconds'],
            reciprocity_pixel_partition=dict(image_pixels=int(dl.size), invalid_left=int((~lv).sum()),
                left_valid_outside_right_image=int((lv & ~bounds).sum()),
                right_sample_invalid=int((bounds & ~usable).sum()), reciprocal_sample_available=int(usable.sum())),
            conventions={})
        assert sum(frame['reciprocity_pixel_partition'][k] for k in ['invalid_left','left_valid_outside_right_image','right_sample_invalid','reciprocal_sample_available']) == dl.size
        saved_masks = masks | {'reciprocal_sample_available': usable}
        for name in ['scaled_full', 'supplied_half']:
            cal = baseline.projection(source/(name+'-calibration.txt'), left.shape[1], left.shape[0])
            offset, numerator = cal['principal_point_offset_px'], cal['disparity_numerator_px_m']
            predicted = np.full(dl.shape, np.nan, dtype=np.float64)
            zbase = lv & ((dl-offset) > 0)
            predicted[zbase] = numerator/(dl[zbase].astype(np.float64)-offset)
            zbase &= np.isfinite(predicted) & (predicted > 0)
            expected = offset+numerator/reference
            pred_ref, depth_ref, zbase_ref = dl[valid_ref].astype(np.float64), predicted[valid_ref], zbase[valid_ref]
            variants = {}
            for key, support in masks.items():
                for bounded in [False, True]:
                    variant = key + ('_max80m' if bounded else '')
                    zm = zbase & support
                    if bounded:
                        zm &= predicted <= MAX_DEPTH_M
                    # An 80m depth limit also limits the scored disparity support.
                    dm = (zm if bounded else support)
                    saved_masks[f'{name}_{variant}_depth'] = zm
                    saved_masks[f'{name}_{variant}_disparity'] = dm
                    variants[variant] = score(pred_ref, expected, depth_ref, reference, dm[valid_ref], zm[valid_ref], zbase_ref)
                    variants[variant]['supported_image_depth_pixels'] = int(zm.sum())
                    variants[variant]['supported_image_depth_density'] = float(zm.mean())
                    pooled.setdefault((name, variant), []).append((pred_ref, expected, depth_ref, reference, dm[valid_ref], zm[valid_ref], zbase_ref))
            # The retained baseline must replay exactly under the stricter explicit search validity.
            for kind in ['disparity_px', 'depth_m']:
                if variants['saved_left'][kind]['valid_predictions'] != old_frame['conventions'][name][kind]['valid_predictions']:
                    raise ValueError('Explicit search validation changes historical support; investigate before comparison')
            frame['conventions'][name] = dict(calibration=cal, variants=variants)
        np.savez_compressed(folder/'support-masks.npz', **saved_masks)
        frame['output_hashes'] = {p.name:baseline.sha(p) for p in sorted(folder.iterdir())}
        report['frames'].append(frame)
        print(f'{stem}: right inference {seconds:.3f}s; reciprocal sample available {usable.mean():.3%}')
    for (name, variant), items in pooled.items():
        values = [np.concatenate([item[i] for item in items]) for i in range(7)]
        report['aggregate'].setdefault(name, {})[variant] = score(*values)
    report['right_compute_seconds_total'] = sum(f['right_compute_seconds'] for f in report['frames'])
    report['historical_left_compute_seconds_total'] = old['compute_seconds_total']
    report['combined_inference_arithmetic_seconds'] = report['right_compute_seconds_total'] + old['compute_seconds_total']
    report['right_compute_seconds_distribution'] = baseline.distribution(np.array([f['right_compute_seconds'] for f in report['frames']]))
    report['wall_seconds_before_report_write'] = time.perf_counter()-started
    report['run_artifact_hashes'] = {p.name:baseline.sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Report: {args.output}/report.json; right inference {report["right_compute_seconds_total"]:.3f}s')


if __name__ == '__main__':
    main()
