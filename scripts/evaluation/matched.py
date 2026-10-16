#!/usr/bin/env python3
"""Source-pinned full-frame development evaluator. Never declares a release pass."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import sys

import numpy as np
from PIL import Image


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def artifact(record, parent):
    """Read once; verification and decoding use the same captured bytes."""
    path = Path(record['path'])
    resolved = (parent / path).resolve()
    if (path.is_absolute() or not resolved.is_relative_to(parent.resolve())) and record.get('readOnlyExternal') is not True:
        raise ValueError(f'External artifact needs readOnlyExternal:true: {path}')
    data = resolved.read_bytes()
    if type(record.get('bytes')) is not int or len(data) != record['bytes'] or digest(data) != record['sha256']:
        raise ValueError(f'Artifact bytes/hash mismatch: {path}')
    return data


def shape_of(record):
    shape = record['shape']
    if not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError('Raster shape must be [positive integer height,width]')
    return tuple(shape)


def raster(record, parent):
    data = artifact(record, parent)
    shape = shape_of(record)
    encoding = record['encoding']
    if encoding in ('float32le', 'uint8'):
        dtype = np.dtype('<f4' if encoding == 'float32le' else 'u1')
        if len(data) != int(np.prod(shape)) * dtype.itemsize:
            raise ValueError('Raster byte count does not match shape/encoding')
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    if encoding == 'uint16png':
        with Image.open(io.BytesIO(data)) as image:
            if image.format != 'PNG' or image.mode != 'I;16':
                raise ValueError('Reference must be a single-channel uint16 PNG')
            array = np.array(image)
        if array.shape != shape or array.dtype != np.uint16:
            raise ValueError('Reference PNG shape/encoding mismatch')
        divisor = record['divisor']
        if isinstance(divisor, bool) or not isinstance(divisor, (int, float)) or not np.isfinite(divisor) or divisor <= 0:
            raise ValueError('PNG unit divisor must be explicit, finite and positive')
        return array.astype(np.float64) / divisor
    raise ValueError(f'Unsupported encoding: {encoding}')


def full_frame(processing, shape):
    for key in ('originalDimensions', 'inputDimensions', 'outputDimensions'):
        if (not isinstance(processing[key], list) or any(type(n) is not int for n in processing[key])
                or processing[key] != list(shape)):
            raise ValueError('Only exact full-frame identity raster dimensions are supported')
    if any(processing[key] is not None for key in ('crop', 'pad', 'resize')):
        raise ValueError('Crop/pad/resize require an explicit reviewed adapter; no implicit resampling')
    if processing['coordinateConvention'] != 'OpenCV-left-rectified-x-right-y-down-z-forward':
        raise ValueError('Unsupported coordinate convention')
    if processing['depthConvention'] != 'axialZ':
        raise ValueError('Only explicit axial-Z is supported')
    if not processing['normalization']:
        raise ValueError('Missing input normalization declaration')


def ratio(n, d):
    return n / d if d else None


def distribution(values):
    if not values.size:
        return None
    return dict(mean=float(values.mean()), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), p99=float(np.percentile(values, 99)),
                maximum=float(values.max()))


def depth_scores(predicted, reference, mask):
    valid_reference = np.isfinite(reference) & (reference > 0)
    valid_prediction = mask & np.isfinite(predicted) & (predicted > 0)
    matched = valid_reference & valid_prediction
    total, kept = int(valid_reference.sum()), int(matched.sum())
    signed = predicted[matched].astype(np.float64) - reference[matched].astype(np.float64)
    absolute = np.abs(signed)
    # These fixed diagnostic bins reproduce prior development studies. They are
    # neither release tolerances nor the physical decision resolution.
    diagnostics = {}
    for tolerance in (1, 2):
        bad = int((absolute > tolerance).sum())
        diagnostics[f'absErrorGt{tolerance}m'] = dict(
            badAmongValid=bad, validDenominator=kept, riskAmongValid=ratio(bad, kept),
            invalidOrBad=total-kept+bad, fullReferenceDenominator=total,
            invalidOrBadRate=ratio(total-kept+bad, total),
            withinToleranceCoverage=ratio(kept-bad, total))
    severe = {f'positiveErrorGt{t}m': int((signed > t).sum()) for t in (10, 100)}
    return dict(imagePixels=int(reference.size), validReference=total,
                invalidReference=int(reference.size)-total, sourceSupport=total,
                validPrediction=int(valid_prediction.sum()), matchedPrediction=kept,
                invalidPredictionOnReference=total-kept,
                predictionOutsideReference=int((valid_prediction & ~valid_reference).sum()),
                maskSupportedNonfinite=int((mask & ~np.isfinite(predicted)).sum()),
                maskSupportedNonpositive=int((mask & np.isfinite(predicted) & (predicted <= 0)).sum())), dict(
                predictionCoverageOnFullReference=ratio(kept, total),
                absoluteErrorMeters=distribution(absolute),
                signedErrorMeanMeters=float(signed.mean()) if kept else None,
                rmseMeters=float(np.sqrt(np.mean(signed**2))) if kept else None,
                diagnostics=diagnostics, severeOverestimates=severe,
                usefulPhysicalCoverage=None), signed


def split_audit(observations):
    by_id, groups, images, sites = {}, {}, {}, {}
    failures, limits = [], []
    for observation in observations:
        identity, split = observation['id'], observation['split']
        if identity in by_id:
            raise ValueError(f'Duplicate observation ID: {identity}')
        by_id[identity] = observation
        if split not in ('development', 'heldoutCandidate', 'unassigned'):
            raise ValueError(f'Unsupported split: {split}')
        group = observation['siteGroup']
        if not isinstance(group, str) or not group.strip():
            limits.append(f'{identity}: unknown siteGroup; no independent holdout eligibility')
        else:
            groups.setdefault(group, set()).add(split)
        source_roles = [s['role'] for s in observation['sources']]
        if len(source_roles) != len(set(source_roles)):
            raise ValueError('Duplicate observation source roles')
        for source in observation['sources']:
            if source['role'] in ('left', 'right', 'image'):
                images.setdefault(source['sha256'], set()).add(split)
        site = observation.get('physicalSite')
        if site is not None:
            if not isinstance(site, dict) or not site.get('id') or not site.get('evidence'):
                raise ValueError('Physical site needs an ID and evidence, or null')
            sites.setdefault(site['id'], set()).add(split)
        if split == 'heldoutCandidate':
            if site is None:
                limits.append(f'{identity}: physical site unresolved; holdout candidate is not a final holdout')
            if observation.get('developmentExposed') is not False:
                failures.append(f'{identity}: heldout candidate development exposure not ruled out')
            if observation.get('modelExposure') != 'verifiedAbsent':
                limits.append(f'{identity}: model/pretraining exposure unverified')
    for kind, collection in [('siteGroup', groups), ('imageHash', images), ('physicalSite', sites)]:
        for key, splits in collection.items():
            if 'heldoutCandidate' in splits and len(splits) > 1:
                failures.append(f'Split leakage {kind}={key}: {sorted(splits)}')
    return dict(groupCount=len(groups), declaredPhysicalSiteCount=len(sites),
                observationCount=len(by_id), failures=failures, independenceLimits=limits)


def visibility_scores(cases, stratify=True):
    """Only evidence-backed independent Blocked/Clear labels are resolvable."""
    matrix = {label: {prediction: 0 for prediction in ('Blocked', 'Clear', 'Unknown')}
              for label in ('Blocked', 'Clear')}
    seen, groups, unresolved = set(), set(), 0
    unknown_total = 0
    for case in cases:
        if case['id'] in seen:
            raise ValueError('Duplicate visibility case ID')
        seen.add(case['id'])
        prediction = case['prediction']
        if prediction not in ('Blocked', 'Clear', 'Unknown'):
            raise ValueError('Unsupported visibility prediction label')
        unknown_total += prediction == 'Unknown'
        label = case['reference']
        if label not in ('Blocked', 'Clear', 'unresolved'):
            raise ValueError('Unsupported visibility reference label')
        if label == 'unresolved' or case.get('independent') is not True or not case.get('referenceEvidence'):
            unresolved += 1
            continue
        if not case.get('independenceGroup'):
            raise ValueError('Resolvable case needs an independence group')
        groups.add(case['independenceGroup'])
        matrix[label][prediction] += 1
    resolved = len(cases)-unresolved
    unknown = sum(row['Unknown'] for row in matrix.values())
    correct = matrix['Blocked']['Blocked']+matrix['Clear']['Clear']
    wrong = matrix['Blocked']['Clear']+matrix['Clear']['Blocked']
    result = dict(totalCases=len(cases), resolvableCases=resolved, unresolvedCases=unresolved,
                independentGroups=len(groups), unknownTotal=unknown_total,
                unknownOnResolvable=unknown, confusion=matrix,
                falseClearCount=matrix['Blocked']['Clear'] if sum(matrix['Blocked'].values()) else None,
                blockedReferenceDenominator=sum(matrix['Blocked'].values()),
                correctUsefulCoverage=ratio(correct, resolved),
                decisionCoverage=ratio(resolved-unknown, resolved),
                riskAmongDecisions=ratio(wrong, resolved-unknown),
                falseClearConfidenceBound=None,
                confidenceLimit='No bound: case/site correlation and a justified locked sampling design are required.')
    if stratify:
        conditions = {c.get('condition', 'unassigned') for c in cases}
        if any(not isinstance(c, str) or not c for c in conditions):
            raise ValueError('Visibility conditions must be nonempty strings')
        result['byCondition'] = {condition: visibility_scores([c for c in cases if c.get('condition', 'unassigned') == condition], False)
                                 for condition in sorted(conditions)}
    return result


def evaluate(observation_path, prediction_path, visibility_path=None):
    ob_bytes, pred_bytes = observation_path.read_bytes(), prediction_path.read_bytes()
    manifest, prediction = parse_json(ob_bytes), parse_json(pred_bytes)
    if any(type(m['schemaVersion']) is not int or m['schemaVersion'] != 1 for m in (manifest, prediction)):
        raise ValueError('Unsupported manifest schema version')
    observations = manifest['observations']
    audit = split_audit(observations)
    if audit['failures']:
        raise ValueError('; '.join(audit['failures']))
    matching = [o for o in observations if o['id'] == prediction['observationId']]
    if len(matching) != 1:
        raise ValueError('Prediction must match exactly one observation')
    observation = matching[0]
    if observation['split'] != 'development':
        raise ValueError('Final/heldout scoring is refused until a justified threshold lock and reference design exist')
    for candidate in observations:
        for source in candidate['sources']:
            artifact(source, observation_path.parent)
    if prediction['execution']['status'] != 'succeeded':
        raise ValueError('Successful actual execution receipt required')
    model = prediction['model']
    if not model.get('provider') or not model.get('id') or not (model.get('version') or model.get('revision')):
        raise ValueError('Provider/model and version or revision identities required')
    if not prediction['execution'].get('codeHashes') or not prediction['execution'].get('environment'):
        raise ValueError('Actual prediction execution code/environment identities required')
    sources = observation['sources']
    if not sources or not prediction['sources']:
        raise ValueError('Explicit observation and prediction inputs required')
    roles = [s['role'] for s in sources]
    if len(roles) != len(set(roles)):
        raise ValueError('Duplicate observation source roles')
    source_pins = {(s['role'], s['sha256']) for s in sources}
    pred_roles = [s['role'] for s in prediction['sources']]
    if len(pred_roles) != len(set(pred_roles)):
        raise ValueError('Duplicate prediction source roles')
    for source in prediction['sources']:
        if (source['role'], source['sha256']) not in source_pins:
            raise ValueError('Prediction input hashes/roles do not match observation')
        artifact(source, prediction_path.parent)
    reference = observation['reference']
    ref_record = reference['raster']
    if ref_record['quantity'] != 'axialDepthMeters':
        raise ValueError('Reference must explicitly declare axialDepthMeters')
    shape = shape_of(ref_record)
    full_frame(reference['processing'], shape)
    full_frame(prediction['processing'], shape)
    for record in (prediction['prediction'], prediction['validity']):
        if shape_of(record) != shape:
            raise ValueError('Prediction/reference/mask shape mismatch')
    if prediction['prediction']['encoding'] != 'float32le' or prediction['validity']['encoding'] != 'uint8':
        raise ValueError('Prediction must be float32le and validity uint8')
    predicted = raster(prediction['prediction'], prediction_path.parent).astype(np.float64)
    mask = raster(prediction['validity'], prediction_path.parent)
    if not np.isin(mask, (0, 1)).all():
        raise ValueError('Validity mask must contain only 0 and 1')
    reference_depth = raster(ref_record, observation_path.parent)
    reference_hashes = {'raster': ref_record['sha256']}
    if prediction['prediction']['quantity'] != 'axialDepthMeters':
        raise ValueError('Convert disparity explicitly in an upstream pinned adapter; evaluator never guesses calibration')
    calibration = observation['calibration']['artifact']
    artifact(calibration, observation_path.parent)
    reference_hashes['calibration'] = calibration['sha256']
    counts, metrics, signed = depth_scores(predicted, reference_depth, mask.astype(bool))
    metrics['byReferenceRangeMeters'] = {}
    for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 80), (80, np.inf)):
        selected = np.isfinite(reference_depth) & (reference_depth > 0) & (reference_depth >= lo) & (reference_depth < hi)
        stratum_counts, stratum_metrics, _ = depth_scores(predicted[selected], reference_depth[selected], mask[selected].astype(bool))
        metrics['byReferenceRangeMeters'][f'{lo}-{hi}'] = dict(counts=stratum_counts, metrics=stratum_metrics)
    visibility = None
    if visibility_path is not None:
        case_bytes = visibility_path.read_bytes()
        case_manifest = parse_json(case_bytes)
        if case_manifest['schemaVersion'] != 1 or case_manifest['observationId'] != observation['id']:
            raise ValueError('Visibility manifest observation/schema mismatch')
        if case_manifest['predictionManifestHash'] != digest(pred_bytes):
            raise ValueError('Visibility receipt must bind the evaluated prediction manifest')
        artifact(case_manifest['predictionEvidence'], visibility_path.parent)
        for case in case_manifest['cases']:
            if not isinstance(case.get('referenceEvidence', []), list):
                raise ValueError('Visibility referenceEvidence must be artifact records')
            for evidence in case.get('referenceEvidence', []):
                artifact(evidence, visibility_path.parent)
        visibility = visibility_scores(case_manifest['cases'])
        reference_hashes['visibilityCases'] = digest(case_bytes)
    limits = list(audit['independenceLimits']) + reference['limits']
    if reference['independent'] is not True:
        limits.append('Depth reference is not independent; diagnostics cannot establish physical accuracy.')
    limits.extend(['Pixel/frame counts are correlated, not independent physical sites.',
                   'Depth error does not label whole-ray visibility or ground/footprint support.',
                   'Diagnostic bins are not scientifically justified release gates; useful physical depth coverage remains unmeasured.',
                   'Successful prediction manifests are provenance assertions; independent execution review remains necessary.'])
    if visibility is None:
        limits.append('No independent visibility labels supplied; false-Clear count is unmeasured, never zero.')
    metrics['visibility'] = visibility
    counts['unresolvedPhysicalLabels'] = visibility['unresolvedCases'] if visibility else None
    return dict(schemaVersion=1, observationManifestHash=digest(ob_bytes),
                predictionManifestHash=digest(pred_bytes), referenceHashes=reference_hashes,
                codeHashes={Path(__file__).name: digest(Path(__file__).read_bytes())},
                split=observation['split'], observationId=observation['id'], counts=counts,
                siteGroup=observation['siteGroup'], metrics=metrics, splitAudit=audit,
                referenceAdequacy=dict(independentMetricReference=reference['independent'],
                    method=reference['method'], timing=reference['timing'], uncertainty=reference['uncertainty'],
                    releaseEligible=False, reason='No justified threshold lock or independent sampling design admitted'),
                excludedOrUnresolved=dict(preparationAdmission=observation['admission'],
                    physicalSite=observation.get('physicalSite'), visibilityLabels='absent' if visibility is None else 'supplied'),
                independenceLimits=limits, thresholdLock=None,
                execution=dict(status='succeeded', kind='saved-output functional evaluation; no inference',
                    createdAtUtc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                    numpy=np.__version__, platform=platform.platform(), elapsedSeconds=None,
                    performance='not measured during concurrent work'), failures=[]), signed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', required=True, type=Path)
    parser.add_argument('--prediction', required=True, type=Path)
    parser.add_argument('--visibility-cases', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Existing output refused')
    try:
        result, signed = evaluate(args.observations, args.prediction, args.visibility_cases)
        args.output.mkdir(parents=True, exist_ok=False)
        raw = signed.astype('<f8').tobytes()
        (args.output/'signed-errors-float64le.bin').write_bytes(raw)
        result['rawErrors'] = dict(path='signed-errors-float64le.bin', bytes=len(raw),
                                  sha256=digest(raw), encoding='float64le', count=int(signed.size))
        (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(args.output), counts=result['counts'], thresholdLock=None)))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f'Evaluation rejected: {exc}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
