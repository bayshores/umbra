#!/usr/bin/env python3
"""Adapt pinned real saved StereoSGBM outputs, without inference or benchmarking."""
import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

import matched


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def pin(path, data, external=False, **fields):
    return dict(path=str(path), bytes=len(data), sha256=matched.digest(data),
                **({'readOnlyExternal': True} if external else {}), **fields)


def read_pin(path, expected_hash):
    data = path.read_bytes()
    if matched.digest(data) != expected_hash:
        raise ValueError(f'Replay input hash mismatch: {path}')
    return data


def replay(source_root, corpus_path, corpus_hash, baseline_path, baseline_hash, output):
    corpus_bytes = read_pin(corpus_path, corpus_hash)
    baseline_bytes = read_pin(baseline_path, baseline_hash)
    corpus, baseline = matched.parse_json(corpus_bytes), matched.parse_json(baseline_bytes)
    if output.exists():
        raise ValueError('Existing replay output refused')
    if corpus['corpusVersion'] != 'development-v2':
        raise ValueError('Replay adapter supports only the explicitly pinned development-v2 corpus')
    records = {record['id']: record for record in corpus['records']}
    if len(records) != len(corpus['records']) or set(records) != {f['stem'] for f in baseline['frames']}:
        raise ValueError('Pinned corpus and saved prediction frame sets must match exactly')
    script_path = baseline_path.parent/'evaluate-stereo-baseline.py'
    read_pin(script_path, baseline['script_sha256'])
    for name, sha in baseline['run_artifact_hashes'].items():
        read_pin(baseline_path.parent/name, sha)
    output.mkdir(parents=True, exist_ok=False)
    observations, receipts, inputs, signed_arrays = [], [], [], []
    for frame in baseline['frames']:
        record = records[frame['stem']]
        if record['evaluationUse'] != 'development-only; no final holdout eligibility' or record['verifiedCalifornia'] is not False:
            raise ValueError('Replay adapter cannot convert this corpus to held-out/California evidence')
        sources = []
        raw = {}
        for role, original in record['rawSources'].items():
            path = (source_root / original['path']).resolve()
            data = read_pin(path, original['sha256'])
            if len(data) != original['bytes']:
                raise ValueError('Corpus input byte count mismatch')
            raw[role] = pin(path, data, True, role=role)
            if role in ('left', 'right'):
                if frame['input_hashes'][role]['sha256'] != original['sha256']:
                    raise ValueError('Saved inference input differs from pinned source image')
                sources.append(raw[role])
        if frame['input_hashes']['depth']['sha256'] != raw['depth']['sha256']:
            raise ValueError('Saved reference differs from pinned source depth')
        saved = {}
        for name, sha in frame['output_hashes'].items():
            path = baseline_path.parent/frame['stem']/name
            data = read_pin(path, sha)
            saved[name] = data
            inputs.append(pin(path.resolve(), data, True))
        for role, name in [('left', 'left.jpg'), ('right', 'right.jpg'), ('depth', 'depth.png'), ('calibration', 'scaled_full-calibration.txt')]:
            if matched.digest(saved[name]) != raw[role]['sha256']:
                raise ValueError('Saved inference/reference/calibration binding mismatch')
        shape = [frame['height'], frame['width']]
        disparity = np.load(io.BytesIO(saved['disparity-px.npy']), allow_pickle=False)
        raw_disparity = np.load(io.BytesIO(saved['disparity-int16.npy']), allow_pickle=False)
        depth = np.load(io.BytesIO(saved['scaled_full-depth-m.npy']), allow_pickle=False)
        if (disparity.shape != tuple(shape) or raw_disparity.shape != tuple(shape) or depth.shape != tuple(shape)
                or disparity.dtype != np.float32 or raw_disparity.dtype != np.int16 or depth.dtype != np.float32
                or not np.array_equal(disparity, raw_disparity.astype(np.float32)/16)):
            raise ValueError('Saved NPY shape/dtype/fixed-point binding mismatch')
        calibration = frame['conventions']['scaled_full']['calibration']
        if calibration['source_sha256'] != raw['calibration']['sha256']:
            raise ValueError('Saved conversion calibration identity mismatch')
        offset, numerator = calibration['principal_point_offset_px'], calibration['disparity_numerator_px_m']
        support = (disparity > baseline['parameters']['minDisparity']-1) & (disparity-offset > 0)
        if not np.array_equal(np.isfinite(depth), support):
            raise ValueError('Saved depth invalidity disagrees with actual matcher support')
        expected_depth = numerator / (disparity[support]-offset)
        if not np.array_equal(depth[support], expected_depth):
            raise ValueError('Saved depth does not match recorded explicit disparity conversion')
        processing = dict(originalDimensions=shape, inputDimensions=shape, outputDimensions=shape,
            crop=None, pad=None, resize=None, normalization='Original CPU OpenCV IMREAD_GRAYSCALE; no normalization',
            coordinateConvention='OpenCV-left-rectified-x-right-y-down-z-forward', depthConvention='axialZ')
        observations.append(dict(id=record['id'], siteGroup=record['leakageGroup'], split='development',
            physicalSite=None, developmentExposed=True, modelExposure='unverified',
            geographyEvidence=dict(verifiedCalifornia=False, source='Pinned development-v2 corpus',
                country='China', evidence=pin(corpus_path.resolve(), corpus_bytes, True)), sources=sources,
            license=dict(sourceUrl='https://github.com/drivingstereo/drivingstereo', version='local MIT bytes',
                acceptanceState='existing acquired dataset; no new terms accepted', artifact=raw['license']),
            calibration=dict(artifact=raw['calibration'], conventions='Provisional scaled-full rectified projection; axial-Z inference'),
            reference=dict(method='Existing sparse filtered/temporally fused/model-guided LiDAR depth export',
                independent=False, dependency='Stereo-model-guided filtering and temporal fusion; surveyed metric accuracy unknown',
                timing='Temporal fusion; per-pixel timing uncertainty unresolved', uncertainty='Not quantified independently',
                limits=['No independent metric reference, free-space, whole-ray or footprint labels.',
                        'Calibration and axial-Z interpretation remain provisional.'], processing=processing,
                raster={**raw['depth'], 'shape': shape, 'encoding': 'uint16png', 'divisor': 256,
                        'quantity': 'axialDepthMeters'}),
            admission=dict(status='usable' if record['preparationAccepted'] else 'rejected',
                reasons=[] if record['preparationAccepted'] else ['Retained existing 70% road-plane admission rejection; depth diagnostic still scored'])))
        folder = output/frame['stem']
        folder.mkdir()
        binary = depth.astype('<f4').tobytes()
        validity = support.astype('u1').tobytes()
        (folder/'depth-float32le.bin').write_bytes(binary)
        (folder/'validity-uint8.bin').write_bytes(validity)
        prediction = dict(schemaVersion=1, observationId=record['id'], sources=sources,
            model=dict(provider='OpenCV', id='StereoSGBM', version=baseline['environment']['opencv'],
                parameters=baseline['parameters'], parametersHash=matched.digest(json.dumps(baseline['parameters'], sort_keys=True).encode())),
            processing={**processing, 'depthConversion': calibration,
                'savedNpyIdentity': pin((baseline_path.parent/frame['stem']/'scaled_full-depth-m.npy').resolve(), saved['scaled_full-depth-m.npy'], True)},
            prediction=pin('depth-float32le.bin', binary, shape=shape, encoding='float32le', quantity='axialDepthMeters'),
            validity=pin('validity-uint8.bin', validity, shape=shape, encoding='uint8'),
            execution=dict(status='succeeded', kind='Reuse saved actual CPU inference; only NPY-to-raw adaptation now',
                originalReceipt=pin(baseline_path.resolve(), baseline_bytes, True),
                codeHashes={'originalBaseline': baseline['script_sha256'], 'replayAdapter': matched.digest(Path(__file__).read_bytes())},
                environment=baseline['environment'], elapsedSeconds=None, usage=dict(cloudRequests=0, newInferenceRuns=0)))
        write_json(folder/'prediction.json', prediction)
        receipts.append((folder, frame))
    observation_path = output/'observations.json'
    write_json(observation_path, dict(schemaVersion=1, observations=observations,
        sourceCorpus=pin(corpus_path.resolve(), corpus_bytes, True)))
    results = []
    for folder, original_frame in receipts:
        result, signed = matched.evaluate(observation_path, folder/'prediction.json')
        # Byte-preserved historical float32 depth differs slightly from the old
        # report's float64 scoring expression. Require agreement, never hide it.
        original = original_frame['conventions']['scaled_full']['depth_m']
        counts = result['counts']
        if counts['validReference'] != original['reference_points'] or counts['matchedPrediction'] != original['valid_predictions']:
            raise ValueError('Full reference/matched denominators disagree with original saved run')
        mae_delta = result['metrics']['absoluteErrorMeters']['mean']-original['absolute_error']['mean']
        if abs(mae_delta) > 1e-4:
            raise ValueError('Saved depth replay MAE disagrees with original beyond numerical representation tolerance')
        result['historicalReadback'] = dict(maeDeltaMeters=mae_delta,
            numericalCheckToleranceMeters=1e-4, explanation='Saved float32 raster vs historical float64 conversion scoring; not an accuracy gate')
        errors = signed.astype('<f8').tobytes()
        (folder/'signed-errors-float64le.bin').write_bytes(errors)
        result['rawErrors'] = pin('signed-errors-float64le.bin', errors, encoding='float64le', count=int(signed.size))
        write_json(folder/'result.json', result)
        results.append(result)
        signed_arrays.append(signed)
    signed = np.concatenate(signed_arrays)
    total = sum(r['counts']['validReference'] for r in results)
    kept = int(signed.size)
    invalid = total-kept
    bad = int((np.abs(signed) > 2).sum())
    pooled_errors = signed.astype('<f8').tobytes()
    (output/'pooled-signed-errors-float64le.bin').write_bytes(pooled_errors)
    report = dict(schemaVersion=1, baselineReport=pin(baseline_path.resolve(), baseline_bytes, True),
        corpus=pin(corpus_path.resolve(), corpus_bytes, True), inputPins=inputs,
        codeHashes={p.name: matched.digest(p.read_bytes()) for p in (Path(__file__), Path(matched.__file__))},
        results=[dict(path=str((folder/'result.json').relative_to(output)), sha256=matched.digest((folder/'result.json').read_bytes())) for folder, _ in receipts],
        split='development', counts=dict(frames=len(results), preparationAccepted=sum(r['preparationAccepted'] for r in records.values()),
            preparationRejected=sum(not r['preparationAccepted'] for r in records.values()), conservativeSequenceGroups=1,
            distinctPhysicalSites=None, verifiedCaliforniaSites=0, finalHoldoutObservations=0,
            independentMetricReferences=0, validReference=total, matchedPrediction=kept, invalidPredictionOnReference=invalid,
            independentResolvableVisibilityLabels=0, falseClearCount=None),
        metrics=dict(predictionCoverageOnFullReference=matched.ratio(kept, total),
            absoluteErrorMeters=matched.distribution(np.abs(signed)),
            invalidOrAbsErrorGt2m=dict(count=invalid+bad, denominator=total, rate=matched.ratio(invalid+bad, total)),
            positiveErrorGt10m=int((signed > 10).sum()), positiveErrorGt100m=int((signed > 100).sum()),
            usefulPhysicalCoverage=None, falseClearConfidenceBound=None),
        rawErrors=pin('pooled-signed-errors-float64le.bin', pooled_errors, encoding='float64le', count=kept),
        thresholdLock=None, failures=[],
        independenceLimits=['Same-sequence development pixels are correlated; no independent-site or generalization claim.',
            'Sparse filtered/fused/model-guided source reference is not independent metric/whole-ray truth.',
            'No visibility labels supplied: false Clear unmeasured, not zero.',
            'No justified task resolution, release thresholds, independent sampling design or heldout corpus.',
            'Read/hash captures are sequential; no atomic multi-file snapshot or loaded-bytecode authentication claim.'],
        execution=dict(status='succeeded', newInferenceRuns=0, cloudRequests=0, elapsedSeconds=None,
            performance='not measured during concurrent work'))
    write_json(output/'report.json', report)
    print(json.dumps(dict(output=str(output), counts=report['counts'], metrics=report['metrics']), allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--corpus-sha256', required=True)
    parser.add_argument('--baseline-report', required=True, type=Path)
    parser.add_argument('--baseline-report-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        replay(args.source_root, args.corpus, args.corpus_sha256, args.baseline_report, args.baseline_report_sha256, args.output)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f'Replay rejected: {exc}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
