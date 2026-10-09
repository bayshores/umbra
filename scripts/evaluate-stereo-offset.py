#!/usr/bin/env python3
"""One fixed local vertical-offset sensitivity run; never overwrite evidence.

python3 scripts/evaluate-stereo-offset.py [--self-check]
No runtime adoption, calibration change, tuning, or independent validation.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import cv2
import numpy as np

BASE_SCRIPT = Path(__file__).resolve().with_name('evaluate-stereo-baseline.py')
spec = importlib.util.spec_from_file_location('offset_baseline', BASE_SCRIPT)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
REGIONS = {'overall': None, 'existing_bottom_center': (293, 586, 300, 400),
           'wider_bottom_center_diagnostic': (250, 601, 300, 400)}
INPUTS = ('left.jpg', 'right.jpg', 'depth.png', 'disparity-int16.npy',
          'disparity-px.npy', 'scaled_full-calibration.txt', 'supplied_half-calibration.txt',
          'scaled_full-depth-m.npy', 'supplied_half-depth-m.npy')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def translate(image, delta):
    """Forward source y -> destination y+delta, retaining left coordinate system."""
    height, width = image.shape
    transform = np.array([[1., 0., 0.], [0., 1., delta]], np.float64)
    return cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def support_masks(shape, delta, radius):
    # Warp ones with the same interpolation to retain the actual border weight.
    weights = translate(np.ones(shape, np.float32), delta)
    complete_rows = np.all(weights == 1, axis=1)
    # Exclude every output row whose local block window intersects a padding row.
    # Ordinary image-edge handling shared by the baseline is not new padding.
    contaminated = np.convolve((~complete_rows).astype(np.int32),
                              np.ones(2*radius+1, np.int32), mode='same') > 0
    block_support = np.broadcast_to((~contaminated)[:, None], shape).copy()
    return weights, weights == 1, block_support


def region_mask(shape, bounds):
    mask = np.ones(shape, bool) if bounds is None else np.zeros(shape, bool)
    if bounds is not None:
        x0, x1, y0, y1 = bounds
        mask[y0:y1, x0:x1] = True
    return mask


def score(pred, reference, supported, offset, numerator):
    """Reuse the baseline evaluator; mask exclusion remains invalid in full view."""
    masked = np.where(supported, pred, np.nan)
    result = baseline.metrics(masked, offset + numerator/reference, reference, offset, numerator)
    valid = (masked > baseline.PARAMS['minDisparity'] - 1) & ((masked-offset) > 0)
    predicted_depth = np.full(reference.shape, np.nan, np.float64)
    predicted_depth[valid] = numerator/(masked[valid]-offset)
    assert np.all(np.isfinite(predicted_depth[valid]))
    z = result['depth_m']
    assert z['valid_predictions'] == int(valid.sum())
    total = int(reference.size)
    excluded = int((~supported).sum())
    z['comparison_excluded_reference_count'] = excluded
    z['eligible_reference_count'] = total-excluded
    z['invalid_within_eligible_count'] = int((supported & ~valid).sum())
    z['positive_error_gt_100m'] = severe = {}
    count = int((valid & ((predicted_depth-reference) > 100)).sum())
    severe.update(count=count, valid_prediction_denominator=int(valid.sum()),
                  rate_on_valid=baseline.ratio(count, int(valid.sum())),
                  all_reference_denominator=total, rate_on_all_reference=baseline.ratio(count,total),
                  invalid_or_severe_count=total-int(valid.sum())+count,
                  invalid_or_severe_rate_on_all_reference=baseline.ratio(total-int(valid.sum())+count,total))
    assert total == excluded + z['invalid_within_eligible_count'] + z['valid_predictions']
    return result


def self_check():
    baseline.self_check()
    shape, delta, radius = (12, 7), .6114578247070312, 2
    yy, xx = np.indices(shape)
    ramp = yy.astype(np.float32)
    down, up = translate(ramp, delta), translate(ramp, -delta)
    # Independent analytic inverse mapping; INTER_LINEAR has a 1/32 pixel table.
    assert np.allclose(down[2:-2], ramp[2:-2]-delta, atol=1/64)
    assert np.allclose(up[2:-2], ramp[2:-2]+delta, atol=1/64)
    assert np.array_equal(translate(xx.astype(np.float32), delta)[1:], xx[1:])
    impulse = np.zeros(shape, np.float32)
    impulse[5, 3] = 1
    assert np.argmax(translate(impulse, delta)) == 6*shape[1]+3
    assert np.argmax(translate(impulse, -delta)) == 4*shape[1]+3
    wd, direct_d, block_d = support_masks(shape, delta, radius)
    wu, direct_u, block_u = support_masks(shape, -delta, radius)
    assert 0 < wd[0, 0] < 1 and 0 < wu[-1, 0] < 1
    assert direct_d[1:].all() and not direct_d[0].any()
    assert direct_u[:-1].all() and not direct_u[-1].any()
    common = block_d & block_u
    assert common[3:-3].all() and common.sum() == 6*shape[1]
    assert not common[:3].any() and not common[-3:].any()
    raw = np.array([-16, 0, 16, 24], np.int16)
    assert np.array_equal(raw.astype(np.float32)/16, [-1, 0, 1, 1.5])
    result = score(np.array([1., 0., -1., 2.]), np.array([10., 10., 10., 10.]),
                   np.array([True, True, True, False]), 0, 10)['depth_m']
    assert result['valid_predictions'] == 1 and result['invalid_predictions'] == 3
    assert result['comparison_excluded_reference_count'] == 1
    assert result['bad_pixel_rates']['abs_gt_2m']['invalid_or_bad_count'] == 3
    assert score(np.array([1.]), np.array([10.]), np.array([False]), 0, 10)['depth_m']['absolute_error'] is None
    assert score(np.array([]), np.array([]), np.array([], bool), 0, 10)['depth_m']['prediction_coverage_on_reference'] is None


def summary(report):
    lines = ['# Fixed vertical-offset sensitivity experiment', '',
        'Development sensitivity on the original five frames of one sunny sequence. No runtime or calibration adoption.', '',
        f"Exact predeclared correction: right image DOWN {report['protocol']['delta_px']!r} px; control UP by the same value.",
        'Saved unshifted left disparities are reused; ten serial CPU matches use the unchanged left image and translated right image.', '',
        '| Convention / region / variant | Paired references | Finite depths | Coverage | MAE m | p95 m | >100 m positive errors | Invalid or >2 m |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for convention, regions in report['aggregate'].items():
        for region in ('overall', 'existing_bottom_center'):
            for variant, views in regions[region].items():
                z = views['paired']['depth_m']
                e = z['absolute_error'] or {}
                bad = z['bad_pixel_rates']['abs_gt_2m']
                fmt = lambda v: 'None' if v is None else f'{v:.4f}'
                coverage = z['prediction_coverage_on_reference']
                lines.append(f"| {convention} / {region} / {variant} | {z['reference_points']} | {z['valid_predictions']} | {fmt(coverage)} | {fmt(e.get('mean'))} | {fmt(e.get('p95'))} | {z['positive_error_gt_100m']['count']} | {bad['invalid_or_bad_count']} ({fmt(bad['invalid_or_bad_rate_on_all_reference'])}) |")
    lines += ['', 'Full reference views retain every source depth >0 in the denominator and count excluded rows as unavailable. The report also retains raw full-reference diagnostics, per-frame scores, the wider region, support-zero cases, and all masks.', '',
        'The primary existing bottom-center region is x[293,586), y[300,400). The separately declared wider diagnostic is x[250,601), y[300,400). All bounds are half open.', '',
        f"Requested OpenCV threads: 1; getter after request: {report['protocol']['environment']['opencv_threads_observed']}. OpenCL is off. This does not establish single-thread execution.", '',
        'Interpretation must consider finite coverage and invalid-or-absolute-error>2 m alongside conditional MAE. Lower conditional MAE alone does not establish a repair. The opposite-sign control tests direction sensitivity; both interpolated images confound alignment with smoothing. INTER_LINEAR uses quantized interpolation weights even though the exact delta is passed.', '',
        'Common boundary support removes direct interpolation and block-radius padding influence only. SGBM path aggregation, disparity checks and speckle filtering can propagate nonlocal boundary effects beyond this mask. Horizontal image coordinates, disparity /16 scaling and both provisional metric-calibration numerators are unchanged. Sparse source references and same-sequence dependence prevent independent accuracy or release claims.', '',
        'Artifacts and sources are SHA-256 recorded. The report replay and artifact manifest validate counts/hashes without rerunning inference.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('.local/evaluation/stereo-baseline/five-frame'))
    parser.add_argument('--epipolar', type=Path, default=Path('.local/evaluation/epipolar-alignment/report.json'))
    parser.add_argument('--output', type=Path, default=Path('.local/evaluation/stereo-offset'))
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print('Self-check passed: analytic shift/sign/x mapping, interpolation support, radius mask, /16 scaling and zero-support denominators.')
        return
    started = time.perf_counter()
    old_path = args.baseline/'report.json'
    old = json.loads(old_path.read_text())
    epi = json.loads(args.epipolar.read_text())
    delta = epi['aggregate']['F_inliers']['signed_vertical_px']['p50']
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError('Expected positive finite saved y_left-y_right pooled median')
    if len(old['frames']) != 5 or old['parameters'] != baseline.PARAMS or old['script_sha256'] != baseline.sha(BASE_SCRIPT):
        raise ValueError('Five-frame retained baseline source/parameter contract changed')
    variants = {'saved_unshifted': 0., 'correction_down': delta, 'opposite_up': -delta}
    sources = {}
    for frame in old['frames']:
        for name in INPUTS:
            path = args.baseline/frame['stem']/name
            if baseline.sha(path) != frame['output_hashes'][name]:
                raise ValueError(f'Retained artifact changed: {path}')
            sources[str(path.resolve())] = baseline.sha(path)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    matcher = cv2.StereoSGBM_create(**baseline.PARAMS)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/Path(__file__).name)
    shutil.copy2(BASE_SCRIPT, args.output/BASE_SCRIPT.name)
    (args.output/'opencv-build.txt').write_text(cv2.getBuildInformation())
    matcher.save(str(args.output/'matcher.yml'))
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), declared_before_execution_and_scores=True,
        hypothesis='The pooled signed vertical offset materially contributes to severe dense depth errors; a fixed right-image downward correction improves coverage/error denominators relative to both saved baseline and opposite-sign control.',
        delta_px=delta, delta_source='aggregate.F_inliers.signed_vertical_px.p50', variants=variants,
        epipolar_report=dict(path=str(args.epipolar.resolve()), sha256=baseline.sha(args.epipolar)),
        baseline_report=dict(path=str(old_path.resolve()), sha256=baseline.sha(old_path)),
        script_sha256=baseline.sha(Path(__file__)), evaluator_sha256=baseline.sha(BASE_SCRIPT), source_hashes=sources,
        command=sys.argv, fixed_parameters=baseline.PARAMS, regions_half_open=REGIONS,
        pairing='Every variant shares intersection of correction/control direct interpolation plus blockSize//2 row support. Paired denominator includes every reference in that mask; full denominator retains excluded references as invalid.',
        inference='Exactly two changed-right input matches per frame: matcher.compute(unchanged_left_gray, shifted_right_gray); reuse saved unshifted int16 disparity without matching it again.',
        interpolation='cv2.INTER_LINEAR, cv2.BORDER_CONSTANT value 0; exact float64 affine delta passed; retain warp-of-ones interpolation weights and complete support masks.',
        metrics='Reuse baseline.metrics/errors/projection unchanged; finite positive depth, conditional MAE/p95, positive error >100m, invalid-or-absolute-error>2m; no clipping or reference range filtering.',
        scaling='Left coordinates and x positions unchanged, raw int16/16 disparity; both saved provisional horizontal projection conventions unchanged. No inferred metric calibration change.',
        environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, opencv=cv2.__version__,
            platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
            opencv_threads_requested=1, opencv_threads_observed=cv2.getNumThreads(), opencl=cv2.ocl.useOpenCL(),
            thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}),
        limitations=old['limitations']+['Same-sequence development sensitivity; offset estimated using overlapping source sequence, not independent validation.',
            'Interpolation smoothing confounds alignment; opposite-sign control is not an interpolation-free comparison.',
            'Boundary mask excludes local direct/block padding only; nonlocal SGBM path/check/speckle effects remain.',
            'Lower conditional MAE without coverage/error-denominator improvement is not a repair. No runtime, scale, safety, or release adoption.'])
    write_json(args.output/'protocol.json', protocol)
    report = dict(protocol=protocol, frames=[], aggregate={}, validation={})
    pooled = {}
    for original in old['frames']:
        stem = original['stem']
        source, folder = args.baseline/stem, args.output/stem
        folder.mkdir()
        left, right = [cv2.imread(str(source/(name+'.jpg')), cv2.IMREAD_GRAYSCALE) for name in ('left','right')]
        raw_depth = cv2.imread(str(source/'depth.png'), cv2.IMREAD_UNCHANGED)
        raw_saved = np.load(source/'disparity-int16.npy', allow_pickle=False)
        assert left.shape == right.shape == raw_depth.shape == raw_saved.shape
        assert raw_depth.dtype == np.uint16 and raw_saved.dtype == np.int16
        assert np.array_equal(raw_saved.astype(np.float32)/16, np.load(source/'disparity-px.npy', allow_pickle=False))
        common = np.ones(left.shape, bool)
        masks, disparities, runtimes = {}, {}, {}
        for variant, shift in variants.items():
            if variant == 'saved_unshifted':
                raw = raw_saved
            else:
                shifted = translate(right, shift)
                np.save(folder/(variant+'-right-gray.npy'), shifted)
                weights, direct, local = support_masks(left.shape, shift, baseline.PARAMS['blockSize']//2)
                masks[variant+'_interpolation_weights'] = weights
                masks[variant+'_direct_support'] = direct
                masks[variant+'_block_support'] = local
                common &= local
                begin = time.perf_counter()
                raw = matcher.compute(left, shifted)
                runtimes[variant] = time.perf_counter()-begin
            assert raw.dtype == np.int16
            np.save(folder/(variant+'-disparity-int16.npy'), raw)
            disparities[variant] = raw.astype(np.float32)/16
            np.save(folder/(variant+'-disparity-px.npy'), disparities[variant])
        masks['paired_common_support'] = common
        valid_ref = raw_depth > 0
        reference = raw_depth[valid_ref].astype(np.float64)/256
        frame = dict(stem=stem, shape=list(left.shape), reference_points=int(valid_ref.sum()),
            image_pixels=int(left.size), new_compute_seconds=runtimes, conventions={}, reference_partitions={})
        for region, bounds in REGIONS.items():
            selected = region_mask(left.shape, bounds) & valid_ref
            masks[region+'_reference'] = selected
            eligible = selected & common
            frame['reference_partitions'][region] = dict(full=int(selected.sum()), paired=int(eligible.sum()), excluded=int((selected & ~common).sum()))
            assert frame['reference_partitions'][region]['full'] == frame['reference_partitions'][region]['paired'] + frame['reference_partitions'][region]['excluded']
        for convention in ('scaled_full','supplied_half'):
            cal = baseline.projection(source/(convention+'-calibration.txt'), left.shape[1],left.shape[0])
            offset, numerator = cal['principal_point_offset_px'],cal['disparity_numerator_px_m']
            frame['conventions'][convention] = dict(calibration=cal,regions={})
            for region, bounds in REGIONS.items():
                selected = masks[region+'_reference'][valid_ref]
                support = common[valid_ref][selected]
                ref = reference[selected]
                variants_scored = {}
                for variant, disparity in disparities.items():
                    pred = disparity[valid_ref].astype(np.float64)[selected]
                    views = {'full_reference_exclusions_invalid':score(pred,ref,support,offset,numerator),
                             'paired':score(pred[support],ref[support],np.ones(int(support.sum()),bool),offset,numerator),
                             'full_reference_raw_diagnostic':score(pred,ref,np.ones(ref.shape,bool),offset,numerator)}
                    variants_scored[variant] = views
                    pooled.setdefault((convention,region,variant),[]).append((pred,ref,support,offset,numerator))
                    if region == 'overall' and variant == 'saved_unshifted':
                        for kind in ('disparity_px','depth_m'):
                            assert views['full_reference_raw_diagnostic'][kind]['valid_predictions'] == original['conventions'][convention][kind]['valid_predictions']
                            assert views['full_reference_raw_diagnostic'][kind]['absolute_error'] == original['conventions'][convention][kind]['absolute_error']
                frame['conventions'][convention]['regions'][region] = variants_scored
        np.savez_compressed(folder/'support-masks.npz',**masks)
        frame['output_hashes'] = {p.name:baseline.sha(p) for p in sorted(folder.iterdir())}
        report['frames'].append(frame)
        print(f'{stem}: two changed-input matches complete; paired reference {frame["reference_partitions"]["overall"]["paired"]}')
    for (convention,region,variant),items in pooled.items():
        assert all(item[3:] == items[0][3:] for item in items)
        pred,ref,support = [np.concatenate([item[i] for item in items]) for i in range(3)]
        offset,numerator = items[0][3:]
        views = {'full_reference_exclusions_invalid':score(pred,ref,support,offset,numerator),
                 'paired':score(pred[support],ref[support],np.ones(int(support.sum()),bool),offset,numerator),
                 'full_reference_raw_diagnostic':score(pred,ref,np.ones(ref.shape,bool),offset,numerator)}
        report['aggregate'].setdefault(convention,{}).setdefault(region,{})[variant] = views
    report['new_match_count'] = sum(len(f['new_compute_seconds']) for f in report['frames'])
    assert report['new_match_count'] == 10
    report['new_compute_seconds_total'] = sum(sum(f['new_compute_seconds'].values()) for f in report['frames'])
    report['wall_seconds_before_write'] = time.perf_counter()-started
    report['validation'] = dict(self_check=True, saved_baseline_full_metrics_replayed=True,
        original_source_hashes_unchanged=all(baseline.sha(Path(p))==h for p,h in sources.items()),
        opencv_threads_after_execution=cv2.getNumThreads(), inference_count=10)
    assert report['validation']['original_source_hashes_unchanged']
    write_json(args.output/'report.json',report)
    (args.output/'SUMMARY.md').write_text(summary(report))
    manifest = {str(p.relative_to(args.output)):baseline.sha(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
    write_json(args.output/'artifact-hashes.json',manifest)
    # Read back without inference: preserve original predictions and verify saved bytes.
    assert json.loads((args.output/'report.json').read_text()) == json.loads(json.dumps(report))
    assert all(baseline.sha(args.output/p)==h for p,h in manifest.items())
    print(f'Report saved; {report["new_match_count"]} serial new matches, {report["new_compute_seconds_total"]:.3f}s total matching.')


if __name__ == '__main__':
    main()
