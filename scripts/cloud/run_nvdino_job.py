#!/usr/bin/env python3
"""One NV-Grounding-DINO inference inside the official TAO 7.2.0 container (one Serverless Job).

Standard library only; injected into the job, so it must stay under 64 KiB. It refuses to run without
--execute, verifies the plan sha256 and every injected input, reassembles the image from its chunks, runs
pre-flight checks before any download (writable work dir, free disk, nvidia-smi, torch.cuda), checks the
spec's model block, downloads the pinned weights and BERT files once (no retries), runs TAO once, parses
its KITTI-style label lines, merges lines with identical boxes, writes a fixed bundle, prints the results
as a numbered base64 block (second retrieval channel) and waits for a retrieval acknowledgement.

It creates no cloud resources and accepts no credentials. Outputs are application records, not provider
proof. Timings stay in private records: the NVIDIA software terms bar publishing performance data.
"""
import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

KIND_PLAN = 'sightline-nvdino-job-plan'
KIND_RESULT = 'sightline-nvdino-job-result'
KIND_BOXES = 'sightline-nvdino-boxes'
READY = 'NVDINO result ready; awaiting retrieval acknowledgement.'
FAILED = 'NVDINO-JOB-FAILED'
BEGIN, LINE, END = 'SIGHTLINE-NVDINO-BEGIN', 'SIGHTLINE-NVDINO-L', 'SIGHTLINE-NVDINO-END'
OUTPUT_NAMES = ('plan.json', 'infer.yaml', 'labels.txt', 'annotated.jpg', 'boxes.json', 'env.json',
                'tao-stdout.log', 'job-result.json')
BLOCK_NAMES = ('boxes.json', 'labels.txt', 'job-result.json')
LINE_CHARS = 76
MAX_PLAN_BYTES = 65536
MAX_LABEL_BYTES = 1 << 20
MAX_ANNOTATED_BYTES = 4 << 20
MAX_BUNDLE_BYTES = 8 << 20
MAX_BLOCK_BYTES = 256 << 10
MODEL_URL_PREFIXES = {'weights': 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/grounding_dino/',
                      'bert': 'https://huggingface.co/'}
WRAPPER_SECONDS = 1800
SID = re.compile(r'[0-9a-f]{32}\Z')
HEX = re.compile(r'[0-9a-f]{64}\Z')
COORD = re.compile(r'-?\d+\.\d{3}\Z')
SCORE = re.compile(r'[01]\.\d{3}\Z')
YAML_LINE = re.compile(r'  ([a-z_]+): (.+)\Z')
DEDUPE_RULE = ('TAO keeps the top 100 (box, label) pairs without merging, so one object can appear once per '
               'caption with identical coordinates. Lines whose printed x1 y1 x2 y2 are identical are one box: '
               'the highest score keeps its label (ties: earlier caption in the prompt, then earlier line); '
               'the other lines are listed as mergedLabels.')
TORCH_CHECK = r'''
import json, torch
r = {'torch': torch.__version__, 'cudaBuild': torch.version.cuda, 'available': bool(torch.cuda.is_available())}
if r['available']:
    r['count'] = int(torch.cuda.device_count())
    r['name'] = str(torch.cuda.get_device_name(0))
    r['op'] = float((torch.ones(4, device='cuda') * 2).sum().item())
try:
    from torch._weights_only_unpickler import _get_allowed_globals
    r['counterAllowlisted'] = 'collections.Counter' in _get_allowed_globals()
except Exception:
    r['counterAllowlisted'] = None
print('TORCHCHECK ' + json.dumps(r, sort_keys=True))
'''


class Refused(Exception):
    pass


def require(condition, message='refused'):
    if not condition:
        raise Refused(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def pretty(value):
    return (json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + '\n').encode('ascii')


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def parse_json(raw, limit=MAX_PLAN_BYTES):
    """Strict JSON: bounded, UTF-8, no duplicate keys, no NaN/Infinity, object at top level."""
    require(isinstance(raw, bytes) and len(raw) <= limit, 'JSON too large')

    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out

    def finite(text):
        value = float(text)
        require(math.isfinite(value), 'non-finite JSON number')
        return value

    def constant(_):
        raise Refused('non-finite JSON constant')

    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_float=finite, parse_constant=constant)
    require(isinstance(value, dict), 'JSON must be an object')
    return value


def read_regular(path, limit):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_size <= limit, 'not a bounded regular file: ' + str(path))
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, 'file grew past its bound: ' + str(path))
    return raw


def write_new(path, raw, mode=0o600):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def abs_path(value):
    require(isinstance(value, str) and value.startswith('/') and os.path.normpath(value) == value
            and re.fullmatch(r'[A-Za-z0-9_./-]{2,512}', value) is not None and '/..' not in value, 'bad path')
    return value


# ------------------------------------------------------------------ plan
def refuse_av2_url(url):
    require(isinstance(url, str), 'download url')
    decoded = urllib.parse.unquote(url).lower()
    parsed = urllib.parse.urlsplit(decoded)
    require('argoverse' not in (parsed.hostname or '') and parsed.path.lstrip('/').split('/', 1)[0] != 'argoverse'
            and 'datasets/av2' not in decoded, 'AV2 downloads refused')


def validate_plan(plan, submission_id=None, output=None):
    """Structure and internal consistency of a plan; raises Refused. Returns the plan."""
    require(plan.get('schemaVersion') == 1 and plan.get('kind') == KIND_PLAN, 'plan kind')
    test = plan.get('testOnly')
    require(type(test) is bool, 'plan testOnly')
    sid = plan.get('submissionId')
    require(isinstance(sid, str) and SID.fullmatch(sid) is not None, 'plan submissionId')
    if submission_id is not None:
        require(sid == submission_id, 'submission id differs from the plan')
    p = plan['paths']
    for key in ('work', 'images', 'weights', 'bert', 'results', 'output', 'ack'):
        abs_path(p[key])
    require(p['output'] == os.path.dirname(p['output']) + '/' + sid, 'output path must end in the submission id')
    require(p['ack'] == os.path.dirname(p['output']) + '/.' + sid + '-retrieved', 'ack path')
    for key in ('images', 'weights', 'bert', 'results'):
        require(os.path.dirname(p[key]) == p['work'], key + ' must sit directly in the work dir')
    if output is not None:
        require(output == p['output'], 'output argument differs from the plan')
    roles = {}
    for item in plan['injected']:
        abs_path(item['path'])
        require(item['role'] in ('wrapper', 'spec', 'chunk', 'license', 'notice'), 'injected role')
        require(type(item['bytes']) is int and 1 <= item['bytes'] <= 65536 and HEX.fullmatch(item['sha256']),
                'injected size/hash')
        roles.setdefault(item['role'], []).append(item)
    require(len(roles.get('wrapper', [])) == 1 and len(roles.get('spec', [])) == 1, 'one wrapper and one spec')
    require(len({i['path'] for i in plan['injected']}) == len(plan['injected']), 'duplicate injected path')
    chunks = roles.get('chunk', [])
    require([c.get('index') for c in chunks] == list(range(len(chunks))) and chunks, 'chunk order')
    image = plan['imageInput']
    require(image['name'] == 'recorded.jpg' and type(image['bytes']) is int and HEX.fullmatch(image['sha256'])
            and sum(c['bytes'] for c in chunks) == image['bytes'], 'image input')
    spec = plan['spec']
    require(spec['path'] == roles['spec'][0]['path'] and spec['sha256'] == roles['spec'][0]['sha256'], 'spec pin')
    m = plan['model']
    for key in ('name', 'package', 'license', 'weightsFile', 'ngc', 'version', 'attribution', 'runtime',
                'textEncoder', 'weightsSha256'):
        require(isinstance(m.get(key), str) and 0 < len(m[key]) <= 400, 'model.' + key)
    require(HEX.fullmatch(m['weightsSha256']) and type(m['weightsBytes']) is int and m['weightsBytes'] > 0,
            'model weights pin')
    downloads = plan['downloads']
    groups = [d['group'] for d in downloads]
    require(groups.count('weights') == 1 and groups.count('bert') >= 1 and set(groups) <= {'weights', 'bert'},
            'download groups')
    for d in downloads:
        require(HEX.fullmatch(d['sha256']) and type(d['bytes']) is int and d['bytes'] > 0, 'download pin')
        abs_path(d['path'])
        require(os.path.dirname(d['path']) == p[d['group']], 'download path')
        url = d['url']
        require(isinstance(url, str) and len(url) <= 1024 and not any(c in url for c in ' \n\r\t"\'<>\\'),
                'download url')
        refuse_av2_url(url)
        require(url.startswith('https://') or (test and url.startswith('file:///')), 'https only (file:// in test plans)')
        if not test:
            require(url.startswith(MODEL_URL_PREFIXES[d['group']]), 'download outside pinned model resource')
    weights = next(d for d in downloads if d['group'] == 'weights')
    require(weights['sha256'] == m['weightsSha256'] and weights['bytes'] == m['weightsBytes']
            and os.path.basename(weights['path']) == m['weightsFile'], 'weights pin differs from model block')
    timeouts = plan['downloadTimeoutSeconds']
    require(type(timeouts['weights']) is int and type(timeouts['bert']) is int
            and 0 < timeouts['weights'] <= 900 and 0 < timeouts['bert'] <= 300, 'download timeouts')
    pre = plan['preflight']
    require(type(pre['minFreeBytes']) is int and pre['minFreeBytes'] > 0 and isinstance(pre['python'], str)
            and isinstance(pre['nvidiaSmi'], list) and all(isinstance(a, str) for a in pre['nvidiaSmi'])
            and type(pre['timeoutSeconds']) is int and 0 < pre['timeoutSeconds'] <= 300, 'preflight')
    if not test:
        require(pre['minFreeBytes'] >= sum(d['bytes'] for d in downloads), 'free-disk floor below downloads')
    tao = plan['tao']
    require(isinstance(tao['argv'], list) and tao['argv'][-2:] == ['-e', spec['path']]
            and all(isinstance(a, str) for a in tao['argv']), 'tao argv')
    require(tao['env'] == {'TELEMETRY_OPT_OUT': 'yes', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
            'tao env')
    require(type(tao['timeoutSeconds']) is int and 0 < tao['timeoutSeconds'] <= 900, 'tao timeout')
    require(tao['labels'] == 'recorded.txt' and tao['annotated'] == 'recorded.jpg'
            and tao['logCapBytes'] == 1 << 20, 'tao outputs')
    prompt = plan['prompt']
    caps = prompt['captions']
    require(isinstance(caps, list) and caps and all(isinstance(c, str) and re.fullmatch(r'[a-z]{1,20}', c)
            for c in caps) and len(set(caps)) == len(caps), 'captions are single lower-case words')
    require(prompt['caption'] == ' . '.join(caps) + ' .' and prompt['confThreshold'] == 0.25, 'prompt')
    require(plan['dedupeRule'] == DEDUPE_RULE, 'dedupe rule text')
    lim = plan['limits']
    require(lim == {'jobs': 1, 'retries': 0, 'wrapperSeconds': WRAPPER_SECONDS, 'retrievalWindowSeconds': 300,
                    'bundleMaxBytes': MAX_BUNDLE_BYTES, 'blockMaxBytes': MAX_BLOCK_BYTES}, 'limits')
    require(plan['requiredEnv'] == {'TELEMETRY_OPT_OUT': 'yes'}, 'required env')
    require(isinstance(plan['image'], str) and re.fullmatch(r'[a-z0-9./_:-]+@sha256:[0-9a-f]{64}', plan['image']),
            'image must be digest-pinned')
    return plan


def model_block(text):
    """Scalar values of the two-space `model:` block of our controlled spec (comments stripped)."""
    lines = text.split('\n')
    starts = [i for i, line in enumerate(lines) if line.split('#')[0].rstrip() == 'model:']
    require(len(starts) == 1 and sum(line.startswith('model') for line in lines) == 1, 'one model block')
    start = starts[0]
    values = {}
    for line in lines[start + 1:]:
        if not line.startswith('  '):
            break
        match = YAML_LINE.fullmatch(line.split(' #')[0].rstrip())
        require(match is not None, 'unexpected model line')
        require(match[1] not in values, 'duplicate model key')
        values[match[1]] = match[2]
    return values


def check_spec(text, plan):
    """The skeptic's binding checks: log_scale auto, class_embed_bias True, no backbone path, local BERT."""
    model = model_block(text)
    require(model.get('log_scale') == 'auto', 'model.log_scale must be auto')
    require(model.get('class_embed_bias') == 'True', 'model.class_embed_bias must be True')
    require(model.get('pretrained_backbone_path') == 'null', 'model.pretrained_backbone_path must be null')
    require(model.get('text_encoder_type') == plan['paths']['bert'], 'model.text_encoder_type must be local BERT')
    require('use_dn' not in text and 'dn_number' not in text, 'no denoising override')
    weights = next(d for d in plan['downloads'] if d['group'] == 'weights')['path']
    caps = plan['prompt']['captions']
    for needle in ('results_dir: ' + plan['paths']['results'] + '\n', 'checkpoint: ' + weights + ',',
                   'conf_threshold: 0.25,', 'image_dir: [' + plan['paths']['images'] + ']',
                   'captions: [' + ', '.join(caps) + ']', 'num_gpus: 1,'):
        require(text.count(needle) == 1, 'spec must contain ' + needle.strip())
    return model


# ------------------------------------------------------------------ labels
def parse_labels(text, captions, threshold):
    """TAO line: <class> 0.00 0 0.00 x1 y1 x2 y2 0.00 x7 <score>; strict, nothing repaired."""
    require(text == '' or text.endswith('\n'), 'labels must end with a newline')
    rows = []
    for number, line in enumerate(text.split('\n')[:-1], 1):
        t = line.split(' ')
        require(len(t) == 16, 'label line %d: 16 fields expected' % number)
        require(t[0] in captions, 'label line %d: class not in the prompt' % number)
        require(t[1:4] == ['0.00', '0', '0.00'] and t[8:15] == ['0.00'] * 7, 'label line %d: KITTI filler' % number)
        require(all(COORD.fullmatch(v) for v in t[4:8]) and SCORE.fullmatch(t[15]), 'label line %d: number format' % number)
        box = [float(v) for v in t[4:8]]
        score = float(t[15])
        require(box[0] <= box[2] and box[1] <= box[3], 'label line %d: inverted box' % number)
        require(threshold - 0.0005 <= score <= 1.0, 'label line %d: score below the threshold' % number)
        rows.append({'line': number, 'class': t[0], 'coords': tuple(t[4:8]), 'box': box, 'score': score})
    return rows


def dedupe(rows, captions):
    groups = {}
    for row in rows:
        groups.setdefault(row['coords'], []).append(row)
    boxes = []
    for group in groups.values():
        group = sorted(group, key=lambda r: (-r['score'], captions.index(r['class']), r['line']))
        keep = group[0]
        boxes.append({'class': keep['class'], 'score': keep['score'], 'box': keep['box'], 'line': keep['line'],
                      'mergedLabels': [{'class': r['class'], 'score': r['score'], 'line': r['line']} for r in group[1:]]})
    boxes.sort(key=lambda b: (-b['score'], b['box'], b['line']))
    return boxes


# ------------------------------------------------------------------ log block (second retrieval channel)
def encode_block(submission_id, files):
    payload = canonical({'submissionId': submission_id,
                         'files': {name: files[name].decode('ascii') for name in BLOCK_NAMES}})
    require(len(payload) <= MAX_BLOCK_BYTES, 'log block too large')
    text = base64.b64encode(payload).decode('ascii')
    chunks = [text[i:i + LINE_CHARS] for i in range(0, len(text), LINE_CHARS)]
    h = digest(payload)
    out = ['%s v1 sid=%s lines=%d bytes=%d sha256=%s' % (BEGIN, submission_id, len(chunks), len(payload), h)]
    out += ['%s %05d %s' % (LINE, i + 1, c) for i, c in enumerate(chunks)]
    out.append('%s sid=%s sha256=%s' % (END, submission_id, h))
    return out, h


def decode_block(text, submission_id):
    """Rebuild the three files from saved logs in any layout (prefixes, repeats, JSON-wrapped lines or one JSON
    document): markers are found anywhere in the text with boundary checks; every hash and count must match."""
    head = re.compile(BEGIN + r' v1 sid=([0-9a-f]{32}) lines=(\d{1,5}) bytes=(\d{1,7}) sha256=([0-9a-f]{64})(?![0-9a-f])')
    body = re.compile(LINE + r' (\d{5}) ([A-Za-z0-9+/=]{1,%d})(?![A-Za-z0-9+/=])' % LINE_CHARS)
    tail = re.compile(END + r' sid=([0-9a-f]{32}) sha256=([0-9a-f]{64})(?![0-9a-f])')
    text = text.replace('\\/', '/')  # a JSON encoder may escape '/'; base64 and markers never contain '\\'
    heads = {m.groups() for m in head.finditer(text)}
    ends = {m.groups() for m in tail.finditer(text)}
    lines = {}
    for found in body.finditer(text):
        index = int(found[1])
        require(lines.get(index, found[2]) == found[2], 'conflicting block line %d' % index)
        lines[index] = found[2]
    require(len(heads) == 1 and len(ends) == 1, 'exactly one block header and trailer')
    sid, count, size, h = next(iter(heads))
    require(sid == submission_id and next(iter(ends)) == (sid, h), 'block identity')
    require(sorted(lines) == list(range(1, int(count) + 1)), 'block lines missing or extra')
    payload = base64.b64decode(''.join(lines[i] for i in sorted(lines)), validate=True)
    require(len(payload) == int(size) <= MAX_BLOCK_BYTES and digest(payload) == h, 'block hash')
    data = parse_json(payload, MAX_BLOCK_BYTES)
    require(set(data) == {'submissionId', 'files'} and data['submissionId'] == submission_id
            and set(data['files']) == set(BLOCK_NAMES), 'block content')
    return {name: data['files'][name].encode('ascii') for name in BLOCK_NAMES}, h


# ------------------------------------------------------------------ steps
class Clock:
    def __init__(self, seconds):
        self.start = time.monotonic()
        self.deadline = self.start + seconds
        self.marks = {}

    def remaining(self):
        return self.deadline - time.monotonic()

    def bound(self, seconds):
        left = self.remaining()
        require(left > 1, 'wrapper deadline reached')
        return min(seconds, left)

    def mark(self, name, started):
        self.marks[name] = round(time.monotonic() - started, 3)


def run_child(argv, timeout, env=None, stdout=None):
    """One child in its own process group; the whole group is killed on timeout. Returns (code, bytes)."""
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout or subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
    except BaseException:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise
    return proc.returncode, out or b''


def verify_injected(plan, wrapper_path):
    for item in plan['injected']:
        raw = read_regular(item['path'], 65536)
        require(len(raw) == item['bytes'] and digest(raw) == item['sha256'], 'injected file differs: ' + item['path'])
    wrapper = next(i for i in plan['injected'] if i['role'] == 'wrapper')
    require(os.path.realpath(wrapper_path) == wrapper['path'], 'wrapper is not the injected copy')
    return wrapper['sha256']


def preflight(plan, clock):
    p, pre = plan['paths'], plan['preflight']
    env = {}
    require('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD' not in os.environ, 'weights-only loading must stay on')
    for key, value in plan['requiredEnv'].items():
        require(os.environ.get(key) == value, 'environment %s must be %s' % (key, value))
    os.makedirs(p['work'], exist_ok=True)
    probe = os.path.join(p['work'], '.write-probe-' + plan['submissionId'])
    write_new(probe, b'probe')
    os.unlink(probe)
    usage = shutil.disk_usage(p['work'])
    env['freeBytes'] = usage.free
    require(usage.free >= pre['minFreeBytes'], 'free disk %d below %d' % (usage.free, pre['minFreeBytes']))
    code, out = run_child(pre['nvidiaSmi'], clock.bound(60))
    text = out.decode('utf-8', 'replace')[:4096]
    env['nvidiaSmi'] = text.strip()
    require(code == 0 and text.strip(), 'nvidia-smi failed (exit %d): %s' % (code, text.strip()[:300]))
    code, out = run_child([pre['python'], '-B', '-c', TORCH_CHECK], clock.bound(pre['timeoutSeconds']))
    text = out.decode('utf-8', 'replace')
    found = [line[11:] for line in text.splitlines() if line.startswith('TORCHCHECK ')]
    require(code == 0 and len(found) == 1, 'torch check failed (exit %d): %s' % (code, text.strip()[-300:]))
    check = json.loads(found[0])
    env['torch'] = check
    require(check.get('available') is True and check.get('count', 0) >= 1 and check.get('op') == 8.0,
            'torch.cuda is not usable: ' + found[0][:300])
    require(check.get('counterAllowlisted') is not False, 'collections.Counter not allowlisted for weights_only')
    return env


def assemble_image(plan):
    p = plan['paths']
    os.mkdir(p['images'], 0o755)
    chunks = sorted((i for i in plan['injected'] if i['role'] == 'chunk'), key=lambda c: c['index'])
    raw = b''.join(read_regular(c['path'], 65536) for c in chunks)
    image = plan['imageInput']
    require(len(raw) == image['bytes'] and digest(raw) == image['sha256'], 'reassembled image hash differs')
    write_new(os.path.join(p['images'], image['name']), raw, 0o644)
    return raw


class HttpsOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        refuse_av2_url(newurl)
        require(newurl.startswith('https://'), 'redirect leaves https')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(item, deadline):
    """Stream once to a fresh partial file, checking size and sha256; no retry, no resume."""
    refuse_av2_url(item['url'])
    partial = item['path'] + '.partial'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), HttpsOnly)
    request = urllib.request.Request(item['url'], headers={'User-Agent': 'sightline-nvdino-job/1'})
    h, size = hashlib.sha256(), 0
    with opener.open(request, timeout=max(1.0, min(60.0, deadline - time.monotonic()))) as response:
        final = response.geturl()
        refuse_av2_url(final)
        require(final.startswith('https://') or item['url'].startswith('file:///'), 'final url not https')
        status = getattr(response, 'status', None)
        require(status in (None, 200), 'HTTP status %s' % status)
        length = response.headers.get('Content-Length')
        require(length is None or int(length) == item['bytes'], 'Content-Length differs from the pin')
        os.makedirs(os.path.dirname(item['path']), exist_ok=True)
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            while True:
                require(time.monotonic() < deadline, 'download deadline: ' + os.path.basename(item['path']))
                block = response.read(1 << 20)
                if not block:
                    break
                size += len(block)
                require(size <= item['bytes'], 'download larger than its pin')
                h.update(block)
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
    require(size == item['bytes'] and h.hexdigest() == item['sha256'],
            'download hash/size differs: ' + os.path.basename(item['path']))
    os.rename(partial, item['path'])
    host = final.split('/')[2] if final.startswith('https://') else 'file'
    return {'file': os.path.basename(item['path']), 'bytes': size, 'sha256': h.hexdigest(), 'finalHost': host}


def find_one(root, folder, name, limit):
    hits = []
    for base, dirs, files in os.walk(root):
        depth = os.path.relpath(base, root).count(os.sep)
        if depth > 4:
            dirs[:] = []
        if os.path.basename(base) == folder and name in files:
            hits.append(os.path.join(base, name))
    require(len(hits) == 1, 'expected exactly one %s/%s under results, found %d' % (folder, name, len(hits)))
    return os.path.relpath(hits[0], root), read_regular(hits[0], limit)


def capped_log(raw, cap):
    if len(raw) <= cap:
        return raw
    marker = b'\n[... %d bytes omitted by the wrapper ...]\n' % (len(raw) - cap)
    head = cap // 4
    return raw[:head] + marker + raw[len(raw) - (cap - head - len(marker)):]


def emit(line):
    sys.stdout.write(line + '\n')
    sys.stdout.flush()


def run_job(args, clock, state):
    state['step'] = 'arguments'
    require(args.execute is True, 'refusing without --execute')
    require(SID.fullmatch(args.submission_id) and HEX.fullmatch(args.plan_sha256), 'submission id / plan sha format')
    plan_raw = read_regular(args.plan, MAX_PLAN_BYTES)
    require(digest(plan_raw) == args.plan_sha256, 'plan sha256 differs')
    plan = validate_plan(parse_json(plan_raw), args.submission_id, args.output)
    require(args.retrieval_window_seconds in (0, plan['limits']['retrievalWindowSeconds']), 'retrieval window')
    wrapper_sha = verify_injected(plan, os.path.abspath(__file__))
    p = plan['paths']
    require(not os.path.lexists(p['ack']), 'acknowledgement already exists')
    os.makedirs(os.path.dirname(p['output']), mode=0o755, exist_ok=True)
    os.mkdir(p['output'], 0o700)
    out = p['output']
    write_new(os.path.join(out, 'plan.json'), plan_raw)
    created = utc()

    state['step'] = 'preflight'
    started = time.monotonic()
    env = preflight(plan, clock)
    clock.mark('preflight', started)

    state['step'] = 'spec'
    spec_raw = read_regular(plan['spec']['path'], 65536)
    check_spec(spec_raw.decode('ascii'), plan)
    write_new(os.path.join(out, 'infer.yaml'), spec_raw)

    state['step'] = 'image'
    image_raw = assemble_image(plan)

    downloads = []
    for group in ('weights', 'bert'):
        state['step'] = 'download-' + group
        started = time.monotonic()
        deadline = time.monotonic() + clock.bound(plan['downloadTimeoutSeconds'][group])
        for item in plan['downloads']:
            if item['group'] == group:
                downloads.append(download(item, deadline))
        clock.mark('download-' + group, started)

    state['step'] = 'tao'
    started = time.monotonic()
    tao = plan['tao']
    # TAO's launcher switches to its microservice log path when JOB_ID is set and to multi-node mode when
    # WORLD_SIZE is set; neither applies to this single-GPU run, so they are not passed to the child.
    child_env = {k: v for k, v in os.environ.items() if k not in ('JOB_ID', 'WORLD_SIZE')}
    child_env.update(tao['env'])
    os.makedirs(p['results'], exist_ok=False)
    log_path = os.path.join(p['work'], 'tao-stdout.full.log')
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'wb') as log_stream:
        try:
            code, _ = run_child(tao['argv'], clock.bound(tao['timeoutSeconds']), child_env, log_stream)
        except subprocess.TimeoutExpired:
            code = 'timeout'
    clock.mark('tao', started)
    with open(log_path, 'rb') as stream:
        full_log = stream.read()
    if code != 0:
        tail = full_log[-8192:].decode('utf-8', 'replace')
        emit('TAO output tail (last %d bytes):' % min(len(full_log), 8192))
        for line in tail.splitlines()[-80:]:
            emit('  | ' + line[:400])
        raise Refused('grounding_dino inference exit %s' % code)

    state['step'] = 'parse'
    labels_rel, labels_raw = find_one(p['results'], 'labels', tao['labels'], MAX_LABEL_BYTES)
    annotated_rel, annotated_raw = find_one(p['results'], 'images_annotated', tao['annotated'], MAX_ANNOTATED_BYTES)
    require(annotated_raw[:2] == b'\xff\xd8', 'annotated image is not a JPEG')
    caps = plan['prompt']['captions']
    rows = parse_labels(labels_raw.decode('ascii'), caps, plan['prompt']['confThreshold'])
    boxes = dedupe(rows, caps)

    state['step'] = 'write'
    model = plan['model']
    boxes_doc = {'schemaVersion': 1, 'kind': KIND_BOXES, 'testOnly': plan['testOnly'],
                 'submissionId': plan['submissionId'], 'planSha256': args.plan_sha256,
                 'image': plan['imageInput'], 'model': model, 'prompt': plan['prompt'],
                 'labelsSha256': digest(labels_raw), 'rawLineCount': len(rows),
                 'rules': {'dedupe': DEDUPE_RULE,
                           'coordinates': 'Original image pixels x1 y1 x2 y2 as printed by TAO (3 decimals); '
                                          'not clipped to the image.'},
                 'boxes': boxes}
    environment = {'python': sys.version.split()[0], 'platform': sys.platform, 'uname': ' '.join(os.uname()),
                   'containerEnv': {k: os.environ.get(k) for k in (
                       'CUDA_VERSION', 'CUDA_DRIVER_VERSION', 'PYTORCH_VERSION', 'NVIDIA_PYTORCH_VERSION',
                       'TAO_TOOLKIT_VERSION', 'NVIDIA_TAO_TOOLKIT_VERSION', 'TELEMETRY_OPT_OUT')},
                   'taoChildEnv': tao['env'], **env, 'downloads': downloads,
                   'resultsPaths': {'labels': labels_rel, 'annotated': annotated_rel}}
    files = {'labels.txt': labels_raw, 'annotated.jpg': annotated_raw, 'boxes.json': pretty(boxes_doc),
             'env.json': pretty(environment), 'tao-stdout.log': capped_log(full_log, tao['logCapBytes'])}
    for name, raw in files.items():
        write_new(os.path.join(out, name), raw)
    files['plan.json'], files['infer.yaml'] = plan_raw, spec_raw
    clock.marks['total'] = round(time.monotonic() - clock.start, 3)
    result = {'schemaVersion': 1, 'kind': KIND_RESULT, 'claim': 'application_output_binding_only',
              'status': 'succeeded', 'testOnly': plan['testOnly'], 'submissionId': plan['submissionId'],
              'planSha256': args.plan_sha256, 'wrapperSha256': wrapper_sha, 'requestedImage': plan['image'],
              'imageInputSha256': digest(image_raw), 'specSha256': digest(spec_raw),
              'weightsSha256': model['weightsSha256'],
              'downloads': {d['file']: d['sha256'] for d in downloads},
              'taoArgv': tao['argv'], 'taoEnv': tao['env'], 'taoExitCode': 0,
              'rawLineCount': len(rows), 'boxCount': len(boxes),
              'mergedLineCount': sum(len(b['mergedLabels']) for b in boxes),
              'outputs': {name: {'bytes': len(raw), 'sha256': digest(raw)} for name, raw in sorted(files.items())},
              'timingsSeconds': clock.marks, 'createdUtc': created, 'finishedUtc': utc(),
              'limitations': ['Application output binding only; the requested image digest and recorded '
                              'environment do not attest the running binaries or provider execution.',
                              'Timings are private records: the NVIDIA software terms bar publishing '
                              'benchmark or performance data.',
                              'One frame, no ground truth: boxes are detections, not an accuracy claim.']}
    files['job-result.json'] = pretty(result)
    require(sum(len(v) for v in files.values()) <= plan['limits']['bundleMaxBytes'], 'bundle exceeds 8 MiB')
    write_new(os.path.join(out, 'job-result.json'), files['job-result.json'])
    for line in encode_block(plan['submissionId'], files)[0]:
        emit(line)
    return plan


def wait_for_ack(plan, seconds, clock):
    if seconds == 0:
        return
    ack = plan['paths']['ack']
    expires = time.monotonic() + min(seconds, max(0.0, clock.remaining() - 5))
    emit(READY)
    while True:
        try:
            info = os.lstat(ack)
        except FileNotFoundError:
            info = None
        if info is not None:
            require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_size == 0,
                    'acknowledgement must be an empty regular file owned by this user')
            return
        if time.monotonic() >= expires:
            raise TimeoutError('retrieval window expired; the result is only in the log block')
        time.sleep(1.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--submission-id', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--retrieval-window-seconds', type=int, default=0)
    args = parser.parse_args(argv)
    clock = Clock(WRAPPER_SECONDS)

    def expired(_signum, _frame):
        raise TimeoutError('wrapper deadline (%d s) reached' % WRAPPER_SECONDS)

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(WRAPPER_SECONDS)
    state = {'step': 'arguments'}
    try:
        if not args.execute:
            emit('%s step=arguments error=refusing without --execute' % FAILED)
            return 2
        plan = run_job(args, clock, state)
        state['step'] = 'retrieval'
        wait_for_ack(plan, args.retrieval_window_seconds, clock)
    except (Exception, KeyboardInterrupt) as failure:
        message = ('%s: %s' % (type(failure).__name__, failure)).replace('\n', ' ')[:600]
        emit('%s step=%s error=%s' % (FAILED, state['step'], message))
        return 1
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    emit('NVDINO job finished; result retrieved or printed; no retries.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
