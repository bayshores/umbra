#!/usr/bin/env python3
"""Prepare (never launch) the single NV-Grounding-DINO Nebius Serverless Job.

Writes one new directory (refuses to overwrite) holding the injected files, plan.json, infer.yaml,
create-argv.json and prepared-manifest.json. It makes no network or cloud calls and spends nothing.
Private identifiers (project id, SSH public key) are arguments, never constants in this tracked file.

  python3 -I -B scripts/cloud/prepare_nvdino_job.py --output-dir <new dir> --project-id <project> \
      --ssh-key-from-argv <earlier create-argv.json> --chunks-dir <dir with recorded.jpg.part0..2> \
      --experiment-yaml <NGC experiment.yaml> --license-dir <dir with NVIDIA-Open-Model-License.txt, Notice> \
      [--submission-id <32 hex>] [--pins <nvdino-pins.json>] [--retrieval-window {0,300}]

Retrieval window (`--retrieval-window`, default 300): how long the wrapper waits after printing the
result for the coordinator to copy the bundle over SSH and write the acknowledgement. 0 means the log
block printed to stdout is the only retrieval channel: use it when port 22 is not reachable, so the job
does not wait 300 s for an acknowledgement that cannot come and then exit as FAILED. The chosen value is
recorded in plan.json (`retrieval`), in the job arguments, in create-argv.json and in the manifest. The
plan's `limits.retrievalWindowSeconds` stays 300: it is the wrapper's maximum, not the chosen value.

Container command (docs: `--container-command` is the container entrypoint and `--args` are the
arguments passed to it): the TAO image's own ENTRYPOINT is /opt/nvidia/nvidia_entrypoint.sh, which runs
NVIDIA's container start-up (CUDA compatibility and environment set-up) and then execs its arguments.
Naming that same script explicitly as the container command, with `python3 -B <wrapper> ...` as the
arguments, keeps NVIDIA's start-up whichever way the service combines the two: as a replaced entrypoint it
is the same script; if the command were instead prepended to the image entrypoint, the script would exec
itself once more and then python3. A bare `python3` command would skip that start-up, and an empty
command would depend on undocumented defaults (the image CMD starts a TAO microservice). The arguments
contain only whitespace-separated tokens without quotes, so any whitespace splitting yields the same argv.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_nvdino_job as wrapper  # noqa: E402  (same directory; standard library only)

INJECT_LIMIT = 65536
IMAGE = 'nvcr.io/nvidia/tao/tao-toolkit@sha256:3b8c6240bfc29842b761481d7a5df7e3c821cd40b9df950b4b21377c4a42e563'
ENTRYPOINT = '/opt/nvidia/nvidia_entrypoint.sh'
EXPERIMENT_YAML = {'bytes': 2360, 'sha256': 'b6c1f211cb8da80e4ae3e882b87358087befd6e69c097a12b909de52f447e4da'}
LICENSE = {'bytes': 10476, 'sha256': '48b1f47664f5e7f54eb5104b62cdf183fb8cff4c9022b534a0fbe6b0f033e962'}
NOTICE_TEXT = b'Licensed by NVIDIA Corporation under the NVIDIA Open Model License'
IMAGE_INPUT = {'name': 'recorded.jpg', 'bytes': 195846,
               'sha256': '238d845f9519144ceb8b1d262b211e1a5b5989e5a192ac19caf4502811e33d97'}
CHUNKS = [('recorded.jpg.part0', 65282, 'a7af826fbd66fda5727d4e7c51a8302ae58c9ff396fb6d7dbbd9740fe3b772a1'),
          ('recorded.jpg.part1', 65282, 'b1e3dff9802ae23fa16b22b0ccceab4f2ecfb5c233f5d6c7cb450a65573cc79a'),
          ('recorded.jpg.part2', 65282, '5fee13cb51281aa5ea04d52b7b468ce82f99a3149d52d2cf7aed731da8e4a066')]
NGC_VERSION = 'grounding_dino_swin_tiny_commercial_trainable_v1.1'
WEIGHTS = {'file': 'grounding_dino_swin_tiny_commercial_trainable.pth', 'bytes': 2070860191,
           'sha256': '8ea7e089e174e72a7fe57ff63cdba5e1e4994b159e41cf72122a7e0d841beaa6',
           'url': 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/grounding_dino/'
                  + NGC_VERSION + '/files?redirect=true&path=grounding_dino_swin_tiny_commercial_trainable.pth'}
BERT_REPO, BERT_REVISION = 'google-bert/bert-base-uncased', '86b5e0934494bd15c9632b12f734a8a67f723594'
BERT = [('config.json', 570, '7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20'),
        ('tokenizer_config.json', 48, 'a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10'),
        ('vocab.txt', 231508, '07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3'),
        ('tokenizer.json', 466062, 'ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98'),
        ('model.safetensors', 440449768, '68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3')]
CAPTIONS = ['car', 'truck', 'bus', 'person']
RETRIEVAL_WINDOWS = (0, 300)  # the only values the wrapper accepts (0 or limits.retrievalWindowSeconds)
COLORS = {'car': 'green', 'truck': 'blue', 'bus': 'yellow', 'person': 'red'}
MODEL = {'name': 'NV-Grounding-DINO', 'ngc': 'nvidia/tao/grounding_dino', 'version': NGC_VERSION,
         'package': 'NGC nvidia/tao/grounding_dino ' + NGC_VERSION + ', run once with NVIDIA TAO Toolkit 7.2.0',
         'license': 'NVIDIA Open Model License',
         'attribution': 'Licensed by NVIDIA Corporation under the NVIDIA Open Model License',
         'runtime': 'NVIDIA TAO Toolkit 7.2.0 container (NVIDIA software, not redistributed), ' + IMAGE,
         'textEncoder': BERT_REPO + '@' + BERT_REVISION + ' (Apache-2.0)',
         'weightsFile': WEIGHTS['file'], 'weightsBytes': WEIGHTS['bytes'], 'weightsSha256': WEIGHTS['sha256'],
         'labelSpace': 'text prompt ' + ' . '.join(CAPTIONS) + ' .'}
MODEL_KEYS = ['pretrained_backbone_path', 'backbone', 'num_feature_levels', 'dec_layers', 'enc_layers',
              'num_queries', 'dropout_ratio', 'dim_feedforward', 'log_scale', 'class_embed_bias']
EXPECTED_INJECTS = ['run_nvdino_job.py', 'plan.json', 'infer.yaml'] + [c[0] for c in CHUNKS] + [
    'NVIDIA-Open-Model-License.txt', 'Notice']


def fail(message):
    raise SystemExit('prepare_nvdino_job: ' + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def real_paths(sid):
    return {'work': '/workload', 'images': '/workload/images', 'weights': '/workload/weights',
            'bert': '/workload/bert', 'results': '/workload/tao-results', 'output': '/output/' + sid,
            'ack': '/output/.' + sid + '-retrieved'}


def real_downloads(paths):
    out = [{'group': 'weights', 'url': WEIGHTS['url'], 'path': paths['weights'] + '/' + WEIGHTS['file'],
            'bytes': WEIGHTS['bytes'], 'sha256': WEIGHTS['sha256']}]
    for name, size, digest in BERT:
        out.append({'group': 'bert', 'path': paths['bert'] + '/' + name, 'bytes': size, 'sha256': digest,
                    'url': 'https://huggingface.co/%s/resolve/%s/%s' % (BERT_REPO, BERT_REVISION, name)})
    return out


def infer_yaml(experiment_raw, paths):
    """NGC experiment.yaml `model:` block copied with exactly two edits (skeptic section 6)."""
    if len(experiment_raw) != EXPERIMENT_YAML['bytes'] or sha(experiment_raw) != EXPERIMENT_YAML['sha256']:
        fail('experiment.yaml does not match its NGC pin')
    lines = experiment_raw.decode('ascii').split('\n')
    start = lines.index('model:')
    block = []
    for line in lines[start + 1:]:
        if not line.startswith('  '):
            break
        block.append(line.rstrip())
    keys = [line.split(':')[0].strip() for line in block]
    if keys != MODEL_KEYS or '  log_scale: auto' not in block or '  class_embed_bias: True' not in block:
        fail('experiment.yaml model block is not the reviewed one')
    edited = []
    for line in block:
        if line.startswith('  pretrained_backbone_path:'):
            was = line.split(': ', 1)[1]
            line = '  pretrained_backbone_path: null      # EDIT: was ' + was
        edited.append(line)
    edited.append('  text_encoder_type: %s   # EDIT: nested, not a top-level dotted key' % paths['bert'])
    weights = paths['weights'] + '/' + WEIGHTS['file']
    colors = ', '.join('%s: %s' % (c, COLORS[c]) for c in CAPTIONS)
    text = '\n'.join(
        ['results_dir: ' + paths['results'],
         'model:                      # NGC experiment.yaml %s..., two edits marked' % EXPERIMENT_YAML['sha256'][:8]]
        + edited
        + ['inference: {num_gpus: 1, checkpoint: %s, conf_threshold: 0.25, color_map: {%s}}' % (weights, colors),
           'dataset: {infer_data_sources: {image_dir: [%s], captions: [%s]}, batch_size: 1, workers: 1}'
           % (paths['images'], ', '.join(CAPTIONS))]) + '\n'
    return text.encode('ascii')


def retrieval_block(window):
    if window not in RETRIEVAL_WINDOWS:
        fail('retrieval window must be 0 or 300 seconds')
    if window:
        return {'windowSeconds': window, 'channels': ['ssh-bundle', 'log-block'],
                'rule': 'The wrapper waits up to %d s for the acknowledgement after printing the log block; '
                        'the bundle is copied over SSH and the log block is kept as the second channel.' % window}
    return {'windowSeconds': 0, 'channels': ['log-block'],
            'rule': 'No acknowledgement wait: the log block in the job logs is the only retrieval channel '
                    '(prepared this way when port 22 is not reachable).'}


def build_plan(sid, injected, *, paths=None, downloads=None, test_only=False, min_free_bytes=None,
               image=IMAGE, model=None, retrieval_window=300):
    """Plan dict (validated by the wrapper's own validator). `injected`: dicts with path/role/bytes/sha256."""
    paths = paths or real_paths(sid)
    downloads = downloads or real_downloads(paths)
    total = sum(d['bytes'] for d in downloads)
    spec = next(i for i in injected if i['role'] == 'spec')
    plan = {
        'schemaVersion': 1, 'kind': wrapper.KIND_PLAN, 'testOnly': test_only, 'submissionId': sid,
        'image': image, 'paths': paths, 'injected': injected, 'imageInput': dict(IMAGE_INPUT),
        'spec': {'path': spec['path'], 'sha256': spec['sha256']}, 'model': dict(model or MODEL),
        'downloads': downloads, 'downloadTimeoutSeconds': {'weights': 900, 'bert': 300},
        'preflight': {'minFreeBytes': min_free_bytes or total + (2 << 30), 'python': wrapper.PREFLIGHT_PYTHON,
                      'nvidiaSmi': list(wrapper.NVIDIA_SMI_ARGV), 'timeoutSeconds': 180},
        'tao': {'argv': [*wrapper.TAO_COMMAND, spec['path']],
                'env': {'TELEMETRY_OPT_OUT': 'yes', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
                'timeoutSeconds': 900, 'labels': 'recorded.txt', 'annotated': 'recorded.jpg', 'logCapBytes': 1 << 20},
        'prompt': {'captions': list(CAPTIONS), 'caption': ' . '.join(CAPTIONS) + ' .', 'confThreshold': 0.25},
        'dedupeRule': wrapper.DEDUPE_RULE,
        'limits': {'jobs': 1, 'retries': 0, 'wrapperSeconds': wrapper.WRAPPER_SECONDS, 'retrievalWindowSeconds': 300,
                   'bundleMaxBytes': wrapper.MAX_BUNDLE_BYTES, 'blockMaxBytes': wrapper.MAX_BLOCK_BYTES},
        'requiredEnv': {'TELEMETRY_OPT_OUT': 'yes'},
        'retrieval': retrieval_block(retrieval_window),
    }
    wrapper.validate_plan(plan, sid, paths['output'])
    return plan


def plan_bytes(plan):
    raw = (json.dumps(plan, indent=1, sort_keys=True) + '\n').encode('ascii')
    if len(raw) > INJECT_LIMIT:
        fail('plan.json exceeds the 65,536-byte inject limit')
    return raw


def run_args(sid, plan_sha, paths=None, retrieval_window=300):
    paths = paths or real_paths(sid)
    retrieval_block(retrieval_window)
    return ' '.join(['python3', '-B', '/workload/input/run_nvdino_job.py', '--submission-id', sid,
                     '--plan', '/workload/input/plan.json', '--plan-sha256', plan_sha, '--output', paths['output'],
                     '--execute', '--retrieval-window-seconds', str(retrieval_window)])


def check_inject(path):
    size = path.stat().st_size
    if not 1 <= size <= INJECT_LIMIT:
        fail('%s is %d bytes; injected files must be 1..65,536 bytes' % (path.name, size))
    return size


def ssh_key_from_argv(path):
    argv = json.loads(Path(path).read_text())['argv']
    key = argv[argv.index('--ssh-key') + 1]
    if not re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/]{68}', key):
        fail('the earlier create argv does not hold one plain ssh-ed25519 public key')
    return key


def check_pins(path):
    pins = json.loads(Path(path).read_text())
    w = pins['model']['weights']
    same = (w['bytes'] == WEIGHTS['bytes'] and w['sha256'] == WEIGHTS['sha256'] and w['guestUrl'] == WEIGHTS['url']
            and pins['model']['version'] == NGC_VERSION and pins['model']['experimentYaml'] == EXPERIMENT_YAML
            and pins['container']['amd64ManifestDigest'] == IMAGE.split('@')[1]
            and pins['textEncoder']['revision'] == BERT_REVISION
            and {k: (v['bytes'], v['sha256']) for k, v in pins['textEncoder']['files'].items()}
            == {n: (b, h) for n, b, h in BERT}
            and pins['input']['recorded.jpg']['sha256'] == IMAGE_INPUT['sha256']
            and [(c['name'], c['bytes'], c['sha256']) for c in pins['input']['chunks']] == CHUNKS)
    if not same:
        fail('pins file disagrees with the constants in this script')
    return sha(Path(path).read_bytes())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--project-id', required=True)
    ap.add_argument('--ssh-key-from-argv', type=Path, required=True)
    ap.add_argument('--chunks-dir', type=Path, required=True)
    ap.add_argument('--experiment-yaml', type=Path, required=True)
    ap.add_argument('--license-dir', type=Path, required=True)
    ap.add_argument('--submission-id')
    ap.add_argument('--pins', type=Path)
    ap.add_argument('--retrieval-window', type=int, choices=RETRIEVAL_WINDOWS, default=300,
                    help='seconds the wrapper waits for the SSH retrieval acknowledgement; 0 = log block only')
    args = ap.parse_args(argv)
    if not re.fullmatch(r'project-[a-z0-9]{10,40}', args.project_id):
        fail('project id format')
    sid = args.submission_id or uuid.uuid4().hex
    if not wrapper.SID.fullmatch(sid):
        fail('submission id must be 32 lower-case hex characters')
    out = args.output_dir.resolve()
    if os.path.lexists(out):
        fail('%s exists; refusing to overwrite a prepared job' % out)
    pins_sha = check_pins(args.pins) if args.pins else None
    key = ssh_key_from_argv(args.ssh_key_from_argv)

    sources = {}
    for name, size, digest in CHUNKS:
        raw = (args.chunks_dir / name).read_bytes()
        if (len(raw), sha(raw)) != (size, digest):
            fail(name + ' does not match its pin')
        sources[name] = raw
    if sha(b''.join(sources[c[0]] for c in CHUNKS)) != IMAGE_INPUT['sha256']:
        fail('chunks do not reassemble to the scene recorded.jpg')
    license_raw = (args.license_dir / 'NVIDIA-Open-Model-License.txt').read_bytes()
    notice_raw = (args.license_dir / 'Notice').read_bytes()
    if (len(license_raw), sha(license_raw)) != (LICENSE['bytes'], LICENSE['sha256']):
        fail('license copy differs from the October 10 accepted text')
    if notice_raw.strip() != NOTICE_TEXT:
        fail('Notice text differs')
    paths = real_paths(sid)
    spec_raw = infer_yaml(args.experiment_yaml.read_bytes(), paths)
    wrapper_raw = Path(wrapper.__file__).read_bytes()
    files = {'run_nvdino_job.py': wrapper_raw, 'infer.yaml': spec_raw, **sources,
             'NVIDIA-Open-Model-License.txt': license_raw, 'Notice': notice_raw}
    container = {'run_nvdino_job.py': ('/workload/input/run_nvdino_job.py', 'wrapper'),
                 'infer.yaml': ('/workload/input/infer.yaml', 'spec'),
                 'NVIDIA-Open-Model-License.txt': ('/model/NVIDIA-Open-Model-License.txt', 'license'),
                 'Notice': ('/model/Notice', 'notice')}
    for index, (name, _, _) in enumerate(CHUNKS):
        container[name] = ('/workload/input/' + name, 'chunk', index)
    injected = []
    for name, raw in files.items():
        if not 1 <= len(raw) <= INJECT_LIMIT:
            fail('%s is %d bytes; injected files must be 1..65,536 bytes' % (name, len(raw)))
        item = {'path': container[name][0], 'role': container[name][1], 'bytes': len(raw), 'sha256': sha(raw)}
        if item['role'] == 'chunk':
            item['index'] = container[name][2]
        injected.append(item)
    plan = build_plan(sid, injected, paths=paths, retrieval_window=args.retrieval_window)
    plan_raw = plan_bytes(plan)
    plan_sha = sha(plan_raw)
    files['plan.json'] = plan_raw
    wrapper.check_spec(spec_raw.decode('ascii'), plan)

    out.mkdir(mode=0o700, parents=False)
    for name, raw in files.items():
        target = out / name
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
        check_inject(target)
    container['plan.json'] = ('/workload/input/plan.json', 'plan')
    inject_args = []
    for name in EXPECTED_INJECTS:
        inject_args += ['--inject-file', '%s:%s' % (out / name, container[name][0])]
    args_text = run_args(sid, plan_sha, paths, args.retrieval_window)
    argv_create = ['ai', 'job', 'create', '--parent-id', args.project_id, '--name', 'sightline-nvdino-' + sid[:8],
                   '--image', IMAGE, '--container-command', ENTRYPOINT, '--args', args_text,
                   '--working-dir', '/workload', '--env', 'TELEMETRY_OPT_OUT=yes',
                   '--platform', 'gpu-l40s-a', '--preset', '1gpu-8vcpu-32gb', '--preemptible', '--follows-spot-price',
                   '--disk-size', '200Gi', '--timeout', '1h', '--restart-policy', 'never', '--ssh-key', key,
                   *inject_args, '--async']
    create = {'note': 'Global flags are prepended by the coordinator. Never run with --async except by the '
                      'approved coordinator; the dry run replaces --async with --dry-run.',
              'submissionId': sid, 'planSha256': plan_sha, 'wrapperSha256': sha(wrapper_raw),
              'image': IMAGE, 'containerCommand': ENTRYPOINT, 'args': args_text,
              'retrievalWindowSeconds': args.retrieval_window, 'argv': argv_create}
    create_raw = (json.dumps(create, indent=1) + '\n').encode('ascii')
    (out / 'create-argv.json').write_bytes(create_raw)
    manifest = {
        'schemaVersion': 1, 'submissionId': sid, 'planSha256': plan_sha, 'createArgvSha256': sha(create_raw),
        'files': {name: {'bytes': len(raw), 'sha256': sha(raw), 'containerPath': container[name][0]}
                  for name, raw in sorted(files.items())},
        'sshPublicKeySha256': sha(key.encode('ascii')),
        'retrievalWindowSeconds': args.retrieval_window,
        'sources': {'experimentYamlSha256': EXPERIMENT_YAML['sha256'], 'pinsSha256': pins_sha,
                    'licenseSha256': LICENSE['sha256'], 'sshKeyFromArgv': str(args.ssh_key_from_argv)},
        'containerCommand': {'command': ENTRYPOINT, 'args': args_text, 'why': __doc__.split('Container command')[1]
                             .split('\n\n')[0].replace('\n', ' ').strip()},
        'dryRunArgv': argv_create[:-1] + ['--dry-run'],
        'injectLimitBytes': INJECT_LIMIT,
    }
    (out / 'prepared-manifest.json').write_bytes((json.dumps(manifest, indent=1) + '\n').encode('ascii'))
    print(json.dumps({'outputDir': str(out), 'submissionId': sid, 'planSha256': plan_sha,
                      'createArgvSha256': sha(create_raw), 'retrievalWindowSeconds': args.retrieval_window}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
