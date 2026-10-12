#!/usr/bin/env python3
"""Validate a returned NV-Grounding-DINO job result offline and write nvdino-boxes.json.

Standard library only; no network, credentials or cloud calls. Two sources are accepted:
  --bundle DIR      the retrieved /output/<submission-id> directory (exact allowlist, <= 8 MiB, hashes)
  --log-block FILE  saved job logs holding the wrapper's numbered base64 block (boxes, labels, job result)
Both are bound to the locally prepared plan (sha256) and submission id. Labels are parsed again and the
duplicate-box rule re-applied; the result must equal the job's boxes.json. --compare-log-block checks that
a bundle and the logs carry byte-identical files. Output carries no timings (NVIDIA terms bar publishing
performance data); only boxes, scores and provenance. Test-only plans are refused without --allow-test.

  python3 -I -B scripts/cloud/consume_nvdino_result.py --plan <prepared>/plan.json --plan-sha256 <sha> \
      --submission-id <sid> (--bundle <dir> | --log-block <logs.txt>) --output <nvdino-boxes.json> \
      [--compare-log-block <logs.txt>] [--allow-test]
"""
import argparse
import json
import os
from pathlib import Path
import stat
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_nvdino_job as job  # noqa: E402

KIND = 'sightline-external-boxes'
MAX_LOG_BYTES = 64 << 20
FILE_LIMITS = {'plan.json': job.MAX_PLAN_BYTES, 'infer.yaml': 65536, 'labels.txt': job.MAX_LABEL_BYTES,
               'annotated.jpg': job.MAX_ANNOTATED_BYTES, 'boxes.json': 2 << 20, 'env.json': 1 << 20,
               'tao-stdout.log': (1 << 20), 'job-result.json': 1 << 20}
require = job.require


def read_bundle(root):
    """Exactly the eight allowlisted regular files, no symlinks or directories, at most 8 MiB in total."""
    root = Path(os.path.abspath(root))
    info = os.lstat(root)
    require(stat.S_ISDIR(info.st_mode), 'bundle must be a real directory')
    names = sorted(os.listdir(root))
    require(names == sorted(job.OUTPUT_NAMES), 'bundle files differ from the allowlist: %s' % names)
    files, total = {}, 0
    for name in job.OUTPUT_NAMES:
        raw = job.read_regular(str(root / name), FILE_LIMITS[name])
        total += len(raw)
        files[name] = raw
    require(total <= job.MAX_BUNDLE_BYTES, 'bundle exceeds 8 MiB')
    return files


def check_core(files, plan, plan_sha, sid):
    """boxes.json, labels.txt and job-result.json agree with each other and with the prepared plan."""
    result = job.parse_json(files['job-result.json'], FILE_LIMITS['job-result.json'])
    boxes = job.parse_json(files['boxes.json'], FILE_LIMITS['boxes.json'])
    wrapper = next(i for i in plan['injected'] if i['role'] == 'wrapper')
    require(result.get('kind') == job.KIND_RESULT and result.get('claim') == 'application_output_binding_only'
            and result.get('status') == 'succeeded' and result.get('taoExitCode') == 0, 'job result status')
    require(result.get('submissionId') == sid and result.get('planSha256') == plan_sha
            and result.get('testOnly') is plan['testOnly'], 'job result binding')
    require(result.get('wrapperSha256') == wrapper['sha256'] and result.get('requestedImage') == plan['image']
            and result.get('imageInputSha256') == plan['imageInput']['sha256']
            and result.get('specSha256') == plan['spec']['sha256']
            and result.get('weightsSha256') == plan['model']['weightsSha256'], 'job result pins')
    require(result.get('downloads') == {os.path.basename(d['path']): d['sha256'] for d in plan['downloads']},
            'downloaded file hashes differ from the plan')
    outputs = result.get('outputs')
    require(isinstance(outputs, dict) and set(outputs) == set(job.OUTPUT_NAMES) - {'job-result.json'},
            'job result output list')
    for name, raw in files.items():
        if name != 'job-result.json':
            require(outputs[name] == {'bytes': len(raw), 'sha256': job.digest(raw)}, 'hash differs: ' + name)
    require(boxes.get('kind') == job.KIND_BOXES and boxes.get('schemaVersion') == 1
            and boxes.get('submissionId') == sid and boxes.get('planSha256') == plan_sha
            and boxes.get('testOnly') is plan['testOnly'] and boxes.get('image') == plan['imageInput']
            and boxes.get('model') == plan['model'] and boxes.get('prompt') == plan['prompt']
            and boxes.get('labelsSha256') == job.digest(files['labels.txt']), 'boxes.json binding')
    caps = plan['prompt']['captions']
    rows = job.parse_labels(files['labels.txt'].decode('ascii'), caps, plan['prompt']['confThreshold'])
    again = job.dedupe(rows, caps)
    require(boxes.get('boxes') == again and boxes.get('rawLineCount') == len(rows) == result.get('rawLineCount')
            and result.get('boxCount') == len(again), 'boxes.json is not the labels with the duplicate rule')
    require(boxes.get('rules', {}).get('dedupe') == job.DEDUPE_RULE == plan['dedupeRule'], 'dedupe rule')
    return result, boxes


def check_bundle_extras(files, plan_raw, plan):
    require(files['plan.json'] == plan_raw, 'bundle plan.json differs from the prepared plan')
    require(job.digest(files['infer.yaml']) == plan['spec']['sha256'], 'bundle infer.yaml differs')
    job.check_spec(files['infer.yaml'].decode('ascii'), plan)
    require(files['annotated.jpg'][:2] == b'\xff\xd8' and files['annotated.jpg'][-2:] == b'\xff\xd9',
            'annotated.jpg is not a complete JPEG')
    job.parse_json(files['env.json'], FILE_LIMITS['env.json'])


def consume(plan_path, plan_sha, sid, *, bundle=None, log_block=None, compare_log_block=None, allow_test=False):
    require((bundle is None) != (log_block is None), 'give exactly one of --bundle or --log-block')
    require(compare_log_block is None or bundle is not None, '--compare-log-block needs --bundle')
    plan_raw = job.read_regular(str(plan_path), job.MAX_PLAN_BYTES)
    require(job.digest(plan_raw) == plan_sha, 'prepared plan sha256 differs')
    plan = job.validate_plan(job.parse_json(plan_raw), sid)
    require(allow_test or plan['testOnly'] is False, 'test-only plan refused (pass --allow-test for self-tests)')
    source = {'submissionId': sid, 'planSha256': plan_sha, 'testOnly': plan['testOnly']}
    if bundle is not None:
        files = read_bundle(bundle)
        check_bundle_extras(files, plan_raw, plan)
        source['channel'] = 'bundle'
    else:
        text = job.read_regular(str(log_block), MAX_LOG_BYTES).decode('utf-8', 'replace')
        files, block_sha = job.decode_block(text, sid)
        source.update(channel='log-block', logBlockSha256=block_sha)
    result, boxes = check_core(files, plan, plan_sha, sid)
    if compare_log_block is not None:
        text = job.read_regular(str(compare_log_block), MAX_LOG_BYTES).decode('utf-8', 'replace')
        logged, block_sha = job.decode_block(text, sid)
        require(all(logged[name] == files[name] for name in job.BLOCK_NAMES), 'log block differs from the bundle')
        source.update(logBlockSha256=block_sha, logBlockMatchesBundle=True)
    source.update(jobResultSha256=job.digest(files['job-result.json']), boxesJsonSha256=job.digest(files['boxes.json']),
                  labelsSha256=job.digest(files['labels.txt']))
    return {'schemaVersion': 1, 'kind': KIND, 'testOnly': plan['testOnly'], 'image': plan['imageInput'],
            'model': plan['model'], 'prompt': plan['prompt'], 'threshold': plan['prompt']['confThreshold'],
            'rules': boxes['rules'], 'source': source,
            'boxes': [{'class': b['class'], 'score': b['score'], 'box': b['box'],
                       'mergedLabels': [{'class': m['class'], 'score': m['score']} for m in b['mergedLabels']]}
                      for b in boxes['boxes']],
            'limitations': ['Detector boxes from one frame with no ground truth; not an accuracy claim.',
                            'Scores are this model\'s own and are not calibrated against other detectors.']}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--plan-sha256', required=True)
    ap.add_argument('--submission-id', required=True)
    ap.add_argument('--bundle', type=Path)
    ap.add_argument('--log-block', type=Path)
    ap.add_argument('--compare-log-block', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--allow-test', action='store_true')
    args = ap.parse_args(argv)
    try:
        require(not os.path.lexists(args.output), 'output exists; refusing to overwrite')
        doc = consume(args.plan, args.plan_sha256, args.submission_id, bundle=args.bundle, log_block=args.log_block,
                      compare_log_block=args.compare_log_block, allow_test=args.allow_test)
        raw = (json.dumps(doc, indent=1) + '\n').encode('ascii')
        job.write_new(str(args.output), raw, 0o644)
    except (Exception, KeyboardInterrupt) as failure:
        print('consume_nvdino_result: refused: %s: %s' % (type(failure).__name__, failure), file=sys.stderr)
        return 1
    print('%d boxes from %s; wrote %s' % (len(doc['boxes']), doc['source']['channel'], args.output), file=sys.stderr)
    print(job.digest(raw))
    return 0


if __name__ == '__main__':
    sys.exit(main())
