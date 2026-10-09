#!/usr/bin/env python3
"""Source-pinned local preparation probes. No model/service execution or timing study."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from common import canonical, code_hashes, digest, parse, record, snapshot
import nvidia_stereo as nvidia
import token_factory as tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('crop', 'ledger', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    raw = snapshot(args.crop)
    crop = parse(raw)
    evidence = {'schemaVersion': 1, 'observationId': 'drivingstereo-development-frame-716',
                'evidence': [{'id': 'prepared-common-crop', 'state': 'measured',
                              'summary': 'Recorded development pair has a common integer crop x=71,y=40,width=736,height=320. '
                                         'The saved crop manifest declares no resize/padding/hole fill. This is input preparation, not model prediction or physical precision.',
                              'artifact': record(args.crop, raw, args.output)},
                             {'id': 'independent-accuracy-unavailable', 'state': 'Unknown',
                              'summary': 'The crop manifest explicitly excludes an accuracy claim. No independent physical metric reference is supplied to this audit.',
                              'artifact': record(args.crop, raw, args.output)}],
                'requiredMissing': ['independentPhysicalReference', 'verifiedCaliforniaSite', 'actualNvidiaModelExecution']}
    evidence_path = args.output / 'evidence.json'
    evidence_path.write_bytes(canonical(evidence))
    tf_plan = tf.prepare(evidence_path, args.output / 'nemotron-prepared')
    # Conditional broad envelope from the dated model receipt, deliberately unverified.
    bound = {'model': tf.MODEL, 'endpoint': tf.ENDPOINT, 'maxPromptTokens': 1_048_576,
             'contextBoundVerified': False, 'outputBoundIncludesAllBillableGeneration': False,
             'extraChargesBoundVerified': False, 'extraChargesUpperBoundUsd': '0',
             'verifiedAtUtc': datetime.now(timezone.utc).isoformat(), 'expiresAtUtc': None,
             'note': 'Candidate maximum from dated 1,024K display. Not a verified context/billing bound; no execution permitted.'}
    bound_path = args.output / 'unverified-bound.json'
    bound_path.write_bytes(canonical(bound))
    flight = tf.preflight(args.output / 'nemotron-prepared' / 'plan.json', args.ledger, bound_path)
    (args.output / 'token-factory-preflight.json').write_bytes(canonical(flight))
    stereo_plan = nvidia.prepare(args.crop, evidence['observationId'], args.output / 'nvidia-prepared')
    rejected = None
    try:
        nvidia.run(args.output / 'nvidia-prepared' / 'plan.json', args.output / 'absent-model.onnx', args.output / 'forbidden-prediction')
    except ValueError as exc:
        rejected = str(exc)
    check = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/sponsors', '-v'],
                           cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=60)
    (args.output / 'focused-checks.stdout.txt').write_text(check.stdout)
    (args.output / 'focused-checks.stderr.txt').write_text(check.stderr)
    report = {'schemaVersion': 1, 'createdUtc': datetime.now(timezone.utc).isoformat(),
              'scope': 'real crop preparation and fail-closed local checks only; no sponsor execution',
              'codeHashes': code_hashes(), 'cropManifestSha256': digest(raw),
              'cropArtifactPins': crop['outputs'], 'tfRequestSha256': digest(canonical(tf_plan['request'])),
              'tfCacheKey': tf_plan['cacheKey'], 'nvidiaCacheKey': stereo_plan['cacheKey'],
              'tokenFactoryPreflight': flight, 'nvidiaRejectedReason': rejected,
              'predictionOutputAbsent': not (args.output / 'forbidden-prediction').exists(),
              'nemotronAuditAbsent': not (args.output / 'nemotron-prepared' / 'audit.json').exists(),
              'focusedChecksExit': check.returncode,
              'cloudUsage': 0, 'modelInferenceExecuted': False, 'fixtureLimit': 'Protocol fixtures are not sponsor evidence.'}
    (args.output / 'report.json').write_bytes(canonical(report))
    print(canonical({'focusedChecksExit': check.returncode, 'cloudStatus': flight['status'],
                     'nvidiaRejectedReason': rejected, 'predictionOutputAbsent': report['predictionOutputAbsent']}).decode())
    return 0 if check.returncode == 0 and flight['status'] == 'blocked' and rejected and report['predictionOutputAbsent'] else 1


if __name__ == '__main__':
    sys.exit(main())
