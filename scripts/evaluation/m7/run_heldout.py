"""M7 held-out scene build for one site: public listings and metadata -> frozen builder v4.3 (variant v4-18.5) pass 1 ->
the sensor files it names -> pass 2 -> prepare_av2 -> assemble -> per-scene placement protocol v3.4 -> ground (box) ->
suggestions; when the builder says the car shape applies: ground (box, no suggestions) -> car_shape.py plan (hashed and
recorded before any sweep is downloaded) -> car_shape.py download -> ground with the car shape -> suggestions.

  nice -n 10 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -B scripts/evaluation/m7/run_heldout.py \
      --log <uuid> --crossing <id> --out .local/evaluation/pao-accuracy/m7/<log8>-<crossing> \
      (--protocol-lock-sha256 <sha256> | --rehearse [--fixed-band])
  ... run_heldout.py --log <uuid> --crossing <id> --record-interrupted --protocol-lock-sha256 <sha256>

Held-out sites run only after common.preflight (approved lock, pushed anchor, clean checkout, every pin unchanged); the
builder and car_shape.py check the lock again before they open a file of the log. --rehearse runs the same steps on a
development site of the split (no lock); --fixed-band makes the rehearsal use the held-out fixed height band.

Rules (protocol-lock dataAccess): every download is receipted (size, ETag, MD5 for single-part objects, SHA-256), at most 3
retries, each into a new empty folder. A builder status other than plan-written is the site's recorded outcome (the
builder's own rules). A calibration refusal by the builder is "not scored: calibration". Any other failure stops the
site with the step named (site-stopped.json); it is never rerun. Nothing here prints an app result: the step logs hold
the tools' own output (the banned vehicle-status word and the coordinator-only folder masked), and the coordinator does not
open grounded/ before the held-out label hashes are committed. Every tool step runs without the coordinator-only folder's
environment variable.
The receipt records the labels addendum sha256 the run works under (held-out: its committed anchor; null when D and E
are not scored); score.py refuses a held-out site built under another value. Held-out: the committed lost-site list must
exist and not name the site (statistics.lostSiteRule; the receipt records its sha256 and score.py refuses another), and a
run marker allows one build per site. --record-interrupted
(held-out): a marked build that ended (its process is gone) without a site record is recorded once as stopped
(site-stopped.json in its folder, step "interrupted"), so the site is still scored like a site with no feasible frame.
The shape sweeps follow the frozen shape-v2 attempt rule inside car_shape.py (4 attempts; a sweep that never matches is
recorded as dropped).

Declared-default scenes: when the builder chooses no real current vehicle (currentVehicleTrackPrefix null), the
per-scene protocol sets currentVehicle.trackPrefix to NO_TRACK, a string that cannot begin any AV2 track uuid (lowercase
hex and '-'), so the placement protocol's current-vehicle check finds zero rows and the frozen declared-default rule
applies. (A null prefix would stop suggest_av2_placements.py with a TypeError.)
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import ROOT, SiteStop, fetch, now, read_listing, rel, sha, sha_file, write_new  # noqa: E402

VARIANT = 'v4-18.5'
BUILDER = 'scripts/evaluation/m7/build_scene.py'  # unchanged v4.3 builder, fixed lock-2 path
BUILDER_PROTOCOL = '.local/evaluation/pao-accuracy/lock-2/builder-protocol-v4.3.json'
BUILDER_PROTOCOL_SHA = '1d2d5b6d0bbc81aa08cc6c5b6d0c722c78d2396b658255dba36d5189e1a07c53'
V34 = '.local/evaluation/av2-map-first-2026-10-11/protocol-v3.4.json'
V34_SHA = '330f1dcd6725201a624cf3c3e0e9c3bde41319b38a63ed5cf2e74b26c0dcab22'
SHAPE_PROTOCOL_SHA = '70cbb24fabd1509bcb797d651ba2ec6c094c28c869b4a6ecd061f57a5106a95a'
SPLIT_LOCK = '.local/evaluation/pao-accuracy/split.lock.json'
LICENSE = '.local/data/av2/LICENSE-AND-ATTRIBUTION.md'
SOURCE_DATE_EPOCH = '1791808628'  # the fixed epoch of every builder v4.x run (generatedUtc only)
NICE = ['nice', '-n', '10']
SCENE_KEYS = ('logId', 'cameraTimestampNs', 'lidarTimestampNs', 'annotationTimestampNs', 'crossingId', 'sceneId', 'crop')
NO_TRACK = 'declared-default'  # 'l' and 'r' are not hex: no AV2 track uuid starts with it
# protocol-lock resultRules.heldOutHeightBand: the largest development fit, fixed on every held-out scene.
HELD_OUT_BAND = 0.075
HELD_OUT_BAND_FITTED_ON = '8c52d911/2658849 app scene (grounded-v3.4, placement protocol v3.4)'
CALIBRATION_REFUSALS = ('Stereo sensors missing', 'Stereo sizes must match')
METADATA = (('root', 'annotations.feather'), ('root', 'city_SE3_egovehicle.feather'),
            ('calibration', 'egovehicle_SE3_sensor.feather'), ('calibration', 'intrinsics.feather'))
MAP_PATTERNS = (r'log_map_archive_{log}____[A-Z]+_city_[0-9]+\.json', r'{log}_ground_height_surface____[A-Z]+\.npy',
                r'{log}___img_Sim2_city\.json')
MAX_METADATA = 16_000_000  # the builder's MAX_FILE
MAX_SENSOR = 4_000_000
MAX_LISTING = 2_000_000
CODE = ('scripts/evaluation/m7/run_heldout.py', 'scripts/evaluation/m7/common.py', BUILDER,
        'scripts/california/build_av2_scene.py', 'scripts/california/prepare_av2.py',
        'scripts/california/assemble_av2_scene.py', 'scripts/california/prepare_av2_ground.py',
        'scripts/california/suggest_av2_placements.py', 'scripts/california/car_shape.py', 'scripts/sponsors/common.py',
        'scripts/remap_support.py', 'scripts/prepare-stereo-crop.py', 'scripts/rectified_calibration.py')


class Build:
    def __init__(self, args, entry, lock):
        self.args, self.entry, self.lock = args, entry, lock
        self.lock_sha256 = None if args.rehearse else args.protocol_lock_sha256  # common.fetch guards each URL with it
        self.log, self.crossing = args.log, str(args.crossing)
        self.out = Path(args.out).resolve()
        self.addendum = common.labels_addendum(not args.rehearse)  # held-out: equal to its committed anchor (checked in preflight)
        self.lost = None if args.rehearse else {'file': common.LOST_SITES_REL, 'sha256': common.lost_sites()[1]}  # read before any folder
        self.data = self.out / 'data' / self.log  # publisher layout; the builder needs the folder named by the log id
        self.logs = self.out / 'logs'
        self.held = [] if args.rehearse else ['--held-out-run', '--protocol-lock-sha256', args.protocol_lock_sha256]
        self.steps, self.downloads = [], []
        self.py = [sys.executable, '-I', '-B']

    # ---------------------------------------------------------------- helpers
    def run(self, step, argv, env=None):
        start = time.time()
        result = subprocess.run(NICE + argv, cwd=ROOT, capture_output=True, text=True, env=env or common.child_env())
        files = {}
        for kind, text in (('stdout', result.stdout), ('stderr', result.stderr)):
            path = self.logs / f'{step}-{kind}.txt'
            files[kind] = {'file': rel(path), 'sha256': write_new(path, common.masked(text).encode())}  # the tools' own output, masked
        record = {'step': step, 'command': [rel(a) if os.path.isabs(str(a)) else str(a) for a in argv], 'exit': result.returncode,
                  'seconds': round(time.time() - start, 2), 'logs': files}
        self.steps.append(record)
        return result, record

    def must(self, step, argv, env=None):
        result, _ = self.run(step, argv, env)
        if result.returncode != 0:
            raise SiteStop(step, (result.stderr.strip() or result.stdout.strip())[-400:])
        return result

    def place(self, source, relative):
        """Hard-link a verified download into the publisher layout (same bytes, read-only)."""
        target = self.data / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
        return target

    # ---------------------------------------------------------------- steps
    def listings(self):
        self.listing_files, self.listed = {}, {}
        for name, sub in common.LISTINGS.items():
            path, receipt = fetch(common.listing_url(self.log, sub), self.out / 'listings' / name, f'{name}.xml', MAX_LISTING,
                                  lock_sha256=self.lock_sha256)
            self.downloads.append({'kind': 'listing', 'name': name, **receipt})
            self.listing_files[name] = path
            self.listed[name] = read_listing(path, self.log, sub)
        self.steps.append({'step': 'listings', 'files': {k: rel(v) for k, v in self.listing_files.items()}, 'utc': now()})

    def metadata(self):
        wanted = [(group, name) for group, name in METADATA]
        maps = self.listed['map']
        for pattern in MAP_PATTERNS:
            hits = [k for k in maps if re.fullmatch(pattern.format(log=re.escape(self.log)), k)]
            if len(hits) != 1:
                raise SiteStop('metadata', f'map listing has {len(hits)} files matching {pattern}')
            wanted.append(('map', hits[0]))
        for group, name in wanted:
            entry = self.listed[group].get(name)
            if entry is None:
                raise SiteStop('metadata', f'{group}/{name} is not in the public listing')
            if entry['bytes'] > MAX_METADATA:
                raise SiteStop('metadata', f'{group}/{name} is {entry["bytes"]} bytes, over {MAX_METADATA}')
            relative = name if group == 'root' else f'{group}/{name}'
            path, receipt = fetch(common.BUCKET + entry['key'], self.out / 'downloads' / relative.replace('/', '__'), name,
                                  MAX_METADATA, entry, lock_sha256=self.lock_sha256)
            self.place(path, relative)
            self.downloads.append({'kind': 'metadata', 'publisherPath': relative, 'listing': entry, **receipt})
        self.steps.append({'step': 'metadata', 'files': len(wanted), 'utc': now()})

    def builder(self, name):
        argv = self.py + [BUILDER, '--protocol', BUILDER_PROTOCOL, '--protocol-sha256', BUILDER_PROTOCOL_SHA,
                          '--placement-protocol', V34, '--placement-protocol-sha256', V34_SHA,
                          '--split', rel(common.SPLIT), '--split-lock', SPLIT_LOCK, '--log-id', self.log, '--crossing-id', self.crossing,
                          '--log-dir', str(self.data),
                          '--listing-left', str(self.listing_files['stereo_front_left']),
                          '--listing-right', str(self.listing_files['stereo_front_right']),
                          '--listing-lidar', str(self.listing_files['lidar']),
                          '--license', LICENSE, '--protocol-variant', VARIANT, '--output', str(self.out / name), *self.held]
        result, record = self.run(name, argv)
        if result.returncode != 0:
            message = result.stderr.strip()
            if name == 'builder-pass1' and any(text in message for text in CALIBRATION_REFUSALS):
                return None  # the lock's calibration rule
            raise SiteStop(name, message[-400:] or f'exit {result.returncode}')
        selection_path = self.out / name / 'selection.json'
        selection = json.loads(selection_path.read_bytes())
        record['selection'] = {'file': rel(selection_path), 'sha256': sha_file(selection_path), 'status': selection['status']}
        return selection

    def sensors(self, selection):
        for missing in selection['missingSources']:
            listed = missing['listing'] or {}
            key = listed.get('key', '')
            prefix = common.log_prefix(self.log)
            if not key.startswith(prefix + 'sensors/'):
                raise SiteStop('sensors', f'unexpected key {key!r}')
            relative = key[len(prefix):]
            group = {'left': 'stereo_front_left', 'right': 'stereo_front_right', 'lidar': 'lidar'}[missing['role']]
            entry = self.listed[group].get(relative.rsplit('/', 1)[1])
            if entry != {'key': key, 'bytes': listed['bytes'], 'etag': listed['etag']}:
                raise SiteStop('sensors', f'{relative}: the builder listing record differs from the M7 listing')
            if entry['bytes'] > MAX_SENSOR:
                raise SiteStop('sensors', f'{relative} is {entry["bytes"]} bytes, over {MAX_SENSOR}')
            path, receipt = fetch(common.BUCKET + key, self.out / 'downloads' / relative.replace('/', '__'), relative.rsplit('/', 1)[1],
                                  MAX_SENSOR, entry, lock_sha256=self.lock_sha256)
            self.place(path, relative)
            self.downloads.append({'kind': 'sensor', 'role': missing['role'], 'publisherPath': relative, 'listing': entry, **receipt})
        self.steps.append({'step': 'sensors', 'files': len(selection['missingSources']), 'utc': now()})

    def scene_protocol(self, selection, plan_path):
        """protocol-v3.4 with only the per-scene values replaced; on a held-out scene (or --fixed-band) the height band is
        the fixed development band. Every other field is asserted equal to protocol-v3.4.json."""
        raw = (ROOT / V34).read_bytes()
        if sha(raw) != V34_SHA:
            raise SiteStop('scene protocol', 'protocol-v3.4.json differs from its pinned sha256')
        base = json.loads(raw); values = selection['protocolSceneValues']; scene = values['scene']
        doc = copy.deepcopy(base)
        for key in SCENE_KEYS:
            doc['scene'][key] = scene[key]
        site = self.entry; mode = 'M7 rehearsal on a development site' if self.args.rehearse else 'M7 held-out run'
        doc['scene']['plan'] = (f'plan.json written by builder protocol v4.3 (sha256 {BUILDER_PROTOCOL_SHA}, variant {VARIANT}): '
                                f'{rel(plan_path)}, sha256 {sha_file(plan_path)}')
        doc['scene']['split'] = (f"{site['role']}: {self.log[:8]}/{self.crossing} is a {site['role']} site in the frozen Palo Alto "
                                 f'split (.local/evaluation/pao-accuracy/split.json, sha256 {common.SPLIT_SHA})')
        lane = re.search(r'lane segment (\d+)', base['mapElements']['roadAxis'])
        if lane is None:
            raise SiteStop('scene protocol', 'protocol-v3.4.json roadAxis names no lane segment')
        old_lane = lane.group(1)
        doc['mapElements']['roadAxis'] = base['mapElements']['roadAxis'].replace(f'lane segment {old_lane}', f"lane segment {values['roadAxisLaneId']}", 1)
        prefix = values['currentVehicleTrackPrefix']
        doc['currentVehicle']['trackPrefix'] = NO_TRACK if prefix is None else prefix
        changed = ['name', 'frozenUtc', 'frozenBefore', *[f'scene.{k}' for k in SCENE_KEYS], 'scene.plan', 'scene.split',
                   'mapElements.roadAxis (lane id)', 'currentVehicle.trackPrefix', 'sceneDerivation']
        fixed = not self.args.rehearse or self.args.fixed_band
        if fixed:
            band = doc['alongRayUncertainty']['heightBand']
            band.update(method='fixed', heightBandMeters=HELD_OUT_BAND, fittedOn=HELD_OUT_BAND_FITTED_ON)
            changed.append('alongRayUncertainty.heightBand (method fixed, heightBandMeters, fittedOn)')
        stamp = now()
        doc['name'] = f'AV2 map-first placement protocol v3.4 - scene values from builder v4.3 ({mode}, {self.log[:8]}/{self.crossing})'
        doc['frozenUtc'] = stamp
        doc['frozenBefore'] = ('Hashed in the -freeze.json record before this run\'s ground layer, car shape, suggestions or any metric '
                               'were computed. Rules are protocol-v3.4.json unchanged; only the per-scene values come from the builder '
                               'selection' + ('; the height band is the fixed development band (protocol-lock resultRules.heldOutHeightBand).'
                                              if fixed else '.'))
        doc['sceneDerivation'] = {
            'from': {'file': V34, 'sha256': V34_SHA},
            'sceneValues': {'file': rel(self.out / 'builder/selection.json'), 'sha256': sha_file(self.out / 'builder/selection.json'),
                            'field': 'protocolSceneValues'},
            'changedFields': changed,
            'values': {**{k: scene[k] for k in SCENE_KEYS}, 'roadAxisLaneId': values['roadAxisLaneId'],
                       'currentVehicleTrackPrefix': prefix, 'trackPrefixUsed': doc['currentVehicle']['trackPrefix'],
                       **({'heightBandMeters': HELD_OUT_BAND, 'fittedOn': HELD_OUT_BAND_FITTED_ON} if fixed else {})},
            'scope': mode + '; not an app scene.'}
        changed_top = {'name', 'frozenUtc', 'frozenBefore', 'scene', 'mapElements', 'currentVehicle', 'sceneDerivation'}
        changed_top |= {'alongRayUncertainty'} if fixed else set()
        same = lambda a, b: json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)  # noqa: E731
        problems = [k for k in set(base) | set(doc) if k not in changed_top and not same(base.get(k), doc.get(k))]
        problems += [f'scene.{k}' for k in set(base['scene']) | set(doc['scene'])
                     if k not in SCENE_KEYS + ('plan', 'split') and base['scene'].get(k) != doc['scene'].get(k)]
        problems += [f'{part}.{k}' for part, skip in (('mapElements', 'roadAxis'), ('currentVehicle', 'trackPrefix'))
                     for k in set(base[part]) | set(doc[part]) if k != skip and base[part].get(k) != doc[part].get(k)]
        if fixed:
            a, b = base['alongRayUncertainty'], doc['alongRayUncertainty']
            problems += [f'alongRayUncertainty.{k}' for k in set(a) | set(b) if k != 'heightBand' and not same(a.get(k), b.get(k))]
            problems += [f'heightBand.{k}' for k in set(a['heightBand']) | set(b['heightBand'])
                         if k not in ('method', 'heightBandMeters', 'fittedOn') and not same(a['heightBand'].get(k), b['heightBand'].get(k))]
        if doc['carShape']['shapeProtocol']['sha256'] != SHAPE_PROTOCOL_SHA:
            problems.append('carShape.shapeProtocol')
        if problems:
            raise SiteStop('scene protocol', 'fields differ from protocol-v3.4.json: ' + ', '.join(sorted(problems)))
        name = 'protocol-v3.4-scene.json'
        digest = write_new(self.out / name, doc)
        freeze = {'file': name, 'sha256': digest, 'recordedUtc': stamp, 'derivedFrom': {'file': V34, 'sha256': V34_SHA},
                  'writer': {'file': 'scripts/evaluation/m7/run_heldout.py', 'sha256': sha_file(__file__)},
                  'note': 'Recorded before this run\'s ground layer, car shape, suggestions or any metric were computed.'}
        write_new(self.out / 'protocol-v3.4-scene-freeze.json', freeze)
        self.steps.append({'step': 'scene protocol', 'file': rel(self.out / name), 'sha256': digest, 'fixedBand': fixed,
                           'trackPrefixUsed': doc['currentVehicle']['trackPrefix']})
        return self.out / name, digest

    def ground(self, step, folder, protocol, protocol_sha, shape_args, suggest=True):
        """prepare_av2_ground.py, then suggest_av2_placements.py unless suggest is False: the box pass of a car-shape site
        only feeds car_shape.py plan, and the frozen suggester refuses a box ground.json for an annotated stationary
        vehicle (found by the 2026-10-14 rehearsal)."""
        annotations = self.data / 'annotations.feather'
        # Pass the explicit parent lock to the frozen ground builder's in-process shape helper.
        env = common.child_env(**({} if self.args.rehearse else {'PAO_M7_GROUND_PROTOCOL_LOCK_SHA256': self.args.protocol_lock_sha256}))
        self.must(step, self.py + ['scripts/california/prepare_av2_ground.py',
                                   '--scene', str(self.out / 'assembled/scene.json'), '--scene-sha256', sha_file(self.out / 'assembled/scene.json'),
                                   '--prepared', str(self.out / 'prepared/manifest.json'),
                                   '--prepared-sha256', sha_file(self.out / 'prepared/manifest.json'),
                                   '--protocol', str(protocol), '--protocol-sha256', protocol_sha,
                                   '--annotations', str(annotations), '--annotations-sha256', sha_file(annotations),
                                   '--output', str(self.out / folder), *shape_args], env)
        if not suggest:
            return
        env = common.child_env(SOURCE_DATE_EPOCH=SOURCE_DATE_EPOCH)
        self.must(step.replace('ground', 'suggest'),
                  self.py + ['scripts/california/suggest_av2_placements.py', '--grounded', str(self.out / folder),
                             '--metadata-sha256', sha_file(self.out / folder / 'scene.json'), '--buffers', str(self.out / 'assembled'),
                             '--protocol', str(protocol), '--protocol-sha256', protocol_sha], env)

    def shape(self, selection):
        """car_shape.py plan from the box ground.json (its poseAverage box is the one the shape pass checks), the plan
        sha256 recorded in shape-plan-record.json before any sweep is downloaded, then the download."""
        ground = self.out / 'grounded-box/ground.json'
        annotations, poses, listing = self.data / 'annotations.feather', self.data / 'city_SE3_egovehicle.feather', self.listing_files['lidar']
        root = self.out / 'shape'
        result = self.must('shape-plan', self.py + ['scripts/california/car_shape.py', 'plan', '--log', self.log,
                                                     '--ground', str(ground), '--ground-sha256', sha_file(ground),
                                                     '--annotations', str(annotations), '--annotations-sha256', sha_file(annotations),
                                                     '--poses', str(poses), '--poses-sha256', sha_file(poses),
                                                     '--listing-file', str(listing), '--listing-sha256', sha_file(listing),
                                                     '--out-root', str(root), *self.held])
        plan_dir = root / f'shape-v2-{self.log}'
        plan_sha = sha_file(plan_dir / 'sweep-plan.json')
        if json.loads(result.stdout)['planSha256'] != plan_sha:
            raise SiteStop('shape-plan', 'printed plan sha256 differs from the file')
        plan = json.loads((plan_dir / 'sweep-plan.json').read_bytes())
        if plan['track'] != selection['carShape']['track'] or plan['log'] != self.log:
            raise SiteStop('shape-plan', 'plan track or log differs from the builder selection')
        write_new(self.out / 'shape-plan-record.json', {'planFile': rel(plan_dir / 'sweep-plan.json'), 'planSha256': plan_sha,
                                                        'recordedUtc': now(), 'note': 'Recorded before any shape sweep was downloaded.'})
        self.must('shape-download', self.py + ['scripts/california/car_shape.py', 'download', '--log', self.log, '--plan-sha256', plan_sha,
                                                '--out-root', str(root), *self.held])
        return plan_dir, plan_sha

    # ---------------------------------------------------------------- driver
    def build(self):
        self.listings()
        self.metadata()
        pass1 = self.builder('builder-pass1')
        if pass1 is None:
            return 'not-scored-calibration'
        if pass1['status'] == 'plan-written':
            raise SiteStop('builder-pass1', 'plan-written before any sensor file was downloaded')
        if pass1['status'] != 'sources-missing':
            return pass1['status']  # the builder's own outcome (no feasible frame, occupied default spot, ...); nothing more runs
        self.sensors(pass1)
        selection = self.builder('builder')
        if selection is None or selection['status'] != 'plan-written':
            raise SiteStop('builder', f"pass 2 status {selection and selection['status']}")
        if selection['chosen'] != pass1['chosen'] or selection['protocolSceneValues'] != pass1['protocolSceneValues']:
            raise SiteStop('builder', 'pass 2 chose differently from pass 1')
        plan_path = self.out / 'builder/plan.json'
        if selection['plan']['sha256'] != sha_file(plan_path):
            raise SiteStop('builder', 'plan.json differs from its selection record')
        self.must('prepare', self.py + ['scripts/california/prepare_av2.py', '--plan', str(plan_path), '--output', str(self.out / 'prepared')])
        self.must('assemble', self.py + ['scripts/california/assemble_av2_scene.py', '--manifest', str(self.out / 'prepared/manifest.json'),
                                         '--output', str(self.out / 'assembled'), '--id', selection['protocolSceneValues']['scene']['sceneId']])
        protocol, protocol_sha = self.scene_protocol(selection, plan_path)
        shape = selection['carShape']
        if shape['shapeProtocol']['sha256'] != SHAPE_PROTOCOL_SHA:
            raise SiteStop('builder', 'selection names another shape protocol')
        if not shape['applies']:
            self.ground('ground', 'grounded', protocol, protocol_sha, ['--no-car-shape'])
            return 'built-box'
        self.ground('ground-box', 'grounded-box', protocol, protocol_sha, ['--no-car-shape'], suggest=False)
        plan_dir, plan_sha = self.shape(selection)
        self.ground('ground', 'grounded', protocol, protocol_sha, ['--car-shape-dir', str(plan_dir), '--car-shape-plan-sha256', plan_sha])
        return 'built-shape'

    def receipt(self, outcome, stop=None):
        outputs = {}
        for folder in ('builder-pass1', 'builder', 'prepared', 'assembled', 'grounded-box', 'grounded'):
            for path in sorted((self.out / folder).glob('*.json')) if (self.out / folder).is_dir() else ():
                outputs[rel(path)] = sha_file(path)
        doc = {'kind': 'pao-m7-site-build', 'mode': 'rehearsal (development site)' if self.args.rehearse else 'held-out',
               'site': {'logId': self.log, 'crossingId': self.crossing, 'role': self.entry['role']},
               'protocolLockSha256': None if self.args.rehearse else self.args.protocol_lock_sha256,
               'labelsAddendumSha256': self.addendum, 'lostSiteList': self.lost,
               'variant': VARIANT, 'fixedBand': (not self.args.rehearse) or self.args.fixed_band, 'outcome': outcome,
               'stopped': None if stop is None else {'step': stop.step, 'reason': stop.reason,
                                                     'rule': 'protocol-lock dataAccess: the site stops, is not rerun, and counts like '
                                                             'a site with no feasible frame'},
               'finishedUtc': now(), 'sourceDateEpoch': SOURCE_DATE_EPOCH, 'nice': ' '.join(NICE),
               'python': {'executable': sys.executable, 'version': sys.version.split()[0]},
               'code': {p: sha_file(ROOT / p) for p in CODE}, 'steps': self.steps, 'downloads': self.downloads,
               'downloadedBytes': sum(d['bytes'] for d in self.downloads), 'outputs': outputs,
               'license': 'Argoverse 2: CC BY-NC-SA 4.0 plus publisher terms; public bucket over HTTPS, no credentials, $0'}
        return write_new(self.out / ('site-stopped.json' if stop else 'build-receipt.json'), doc)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--log', required=True)
    parser.add_argument('--crossing', required=True)
    parser.add_argument('--out', help='a new folder under .local/ (not with --record-interrupted)')
    parser.add_argument('--record-interrupted', action='store_true', help='held-out: record a marked build that ended without a site record')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--protocol-lock-sha256')
    mode.add_argument('--rehearse', action='store_true')
    parser.add_argument('--fixed-band', action='store_true', help='rehearsal only: use the held-out fixed height band')
    args = parser.parse_args()
    if args.fixed_band and not args.rehearse:
        parser.error('--fixed-band is for rehearsals; held-out runs always use the fixed band')
    if args.record_interrupted and (args.rehearse or args.out):
        parser.error('--record-interrupted takes only --log, --crossing and --protocol-lock-sha256')
    if not args.record_interrupted and not args.out:
        parser.error('--out is required')
    out = None if args.record_interrupted else Path(args.out).resolve()
    if out and not out.is_relative_to(ROOT / '.local'):
        parser.error('--out must be under .local/ (ignored by git): downloads are never committed')
    if out and os.path.lexists(out):
        parser.error(f'{out} exists; every run writes a new folder')
    entry, lock = common.guard(args.log, args.crossing, args.rehearse, args.protocol_lock_sha256)  # before any network or file access
    if not args.rehearse and f'{args.log[:8]}/{args.crossing}' in common.lost_sites()[0]:  # committed before any held-out build
        parser.error('this site is on the committed lost-site list: it is not built')
    if args.record_interrupted:
        folder = common.ended_run('build', args.log, args.crossing)
        if any((folder / n).is_file() for n in ('build-receipt.json', 'site-stopped.json')):
            parser.error('the marked build has its site record: there is nothing to record')
        args.out = str(folder)
        build = Build(args, entry, lock)
        folder.mkdir(parents=True, exist_ok=True)
        build.receipt('stopped', SiteStop('interrupted', 'the marked build ended without a site record (recorded afterwards with '
                                                         '--record-interrupted)'))
        print(json.dumps({'site': f'{args.log[:8]}/{args.crossing}', 'stopped': 'interrupted', 'record': rel(folder / 'site-stopped.json')}))
        sys.exit(3)
    build = Build(args, entry, lock)  # reads the labels addendum sha256 before any folder is made
    if not args.rehearse:
        common.run_marker('build', args.log, args.crossing, out)  # a held-out site is built once
    out.mkdir(parents=True)
    (out / 'logs').mkdir()
    try:
        outcome = build.build()
    except Exception as error:  # anything unexpected still stops the site with a record; its message is not kept (it could hold a value)
        stop = error if isinstance(error, SiteStop) else SiteStop('unexpected', type(error).__name__)
        build.receipt('stopped', stop)
        print(json.dumps({'site': f'{args.log[:8]}/{args.crossing}', 'stopped': stop.step, 'record': rel(out / 'site-stopped.json')}))
        sys.exit(3)
    digest = build.receipt(outcome)
    print(json.dumps({'site': f'{args.log[:8]}/{args.crossing}', 'outcome': outcome, 'steps': len(build.steps),
                      'receipt': rel(out / 'build-receipt.json'), 'sha256': digest}))


if __name__ == '__main__':
    main()
