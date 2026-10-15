"""M7 metric A input for one crossing: the frozen paint v4 estimators (paint protocol v4, code-v4) on the public files
that run_heldout.py already downloaded for the site, plus the 15 LiDAR sweeps that paint v4's own selection names.

  nice -n 10 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -B scripts/evaluation/m7/paint_heldout.py \
      --site .local/evaluation/pao-accuracy/m7/<log8>-<crossing> --out .local/evaluation/pao-accuracy/m7/<log8>-<crossing>-paint \
      (--protocol-lock-sha256 <sha256> | --rehearse)
  ... paint_heldout.py --site <the marked build folder> --record-interrupted --protocol-lock-sha256 <sha256>

Steps: config.json (the dev config format: sources with paths relative to the config, sha256 and bytes) -> code-v4 select
(metadata and the listing only) -> each chosen sweep downloaded into its own new folder (single-part objects only: bytes
== Content-Length == listing Size, MD5 == ETag == listing ETag; at most 3 retries, 4,000,000-byte ceiling) ->
receipts.json -> code-v4 measure. Paint v4's parameters (sweep count 15, thresholds, agreement and precision rules) are
the locked ones: this file passes none. A failed paint step leaves metric A "not scored" at this crossing (paint-stopped.json);
it never stops the scene build. Nothing here prints a measurement; step logs have the banned vehicle-status word and the
coordinator-only folder masked, and every step runs without that folder's environment variable. Held-out: --site must be
the folder the build marker names, the committed lost-site list must exist and not name the site (the paint record keeps
its sha256), and a run marker allows one paint run per crossing.
--record-interrupted (held-out): a marked paint run that ended (its process is gone) without a paint record is recorded
once as stopped (paint-stopped.json in its folder, step "interrupted"): metric A is "not scored" at this crossing. When
the folder already holds paint v4's result.json, the summary lists the crossing by name (stopped after measuring).
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import ROOT, SiteStop, fetch, now, rel, sha_file, write_new  # noqa: E402

PAINT = '.local/evaluation/pao-paint'
PROTOCOL = f'{PAINT}/protocol-v4.json'
PROTOCOL_SHA = 'f9a52b440180e84c67d78dfb963849da73d7131f7bcaaf9ea6d0e94fafdfe273'
CODE = f'{PAINT}/code-v4/measure_paint_edge.py'
CODE_SHA = 'bd2c6996551e03fc39160cbcb718f314467334979264816dca4f14fd0f92617a'
SWEEP_COUNT = 15  # paint protocol v4 sweepSelection.count (checked against the protocol below)
MAX_SWEEP = 4_000_000  # acquire.py MAX_BYTES
PACKAGES = {'numpy': '2.3.1', 'scipy': '1.16.0', 'pyarrow': '20.0.0'}  # protocol-v4.lock.json environment.packages
NICE = ['nice', '-n', '10']
SOURCES = {'poses': 'city_SE3_egovehicle.feather', 'extrinsics': 'calibration/egovehicle_SE3_sensor.feather'}
MAP_SOURCES = {'map': 'log_map_archive_', 'groundRaster': '_ground_height_surface____', 'groundTransform': '___img_Sim2_city.json'}


def site_record(site):
    """The site's build-receipt.json or site-stopped.json; its metadata downloads must be complete."""
    paths = [site / 'build-receipt.json', site / 'site-stopped.json']
    found = [p for p in paths if p.is_file()]
    if len(found) != 1:
        raise SystemExit(f'{site} has no single site record (build-receipt.json or site-stopped.json)')
    return json.loads(found[0].read_bytes()), found[0]


def paint_doc(args, site, record_path, out, outcome, stop, steps, downloads, packages, lost=None):
    """The paint record (paint-receipt.json, or paint-stopped.json when stop is set)."""
    return {'kind': 'pao-m7-paint', 'mode': 'rehearsal (development site)' if args.rehearse else 'held-out',
            'site': site, 'siteRecord': {'file': rel(record_path), 'sha256': sha_file(record_path)},
            'protocolLockSha256': None if args.rehearse else args.protocol_lock_sha256, 'lostSiteList': lost,
            'paint': {'protocol': {'file': PROTOCOL, 'sha256': PROTOCOL_SHA}, 'code': {'file': CODE, 'sha256': CODE_SHA}},
            'outcome': outcome, 'stopped': None if stop is None else {'step': stop.step, 'reason': stop.reason,
                                                                      'rule': 'metric A is "not scored" at this crossing; the scene is unaffected'},
            'steps': steps, 'downloads': downloads, 'downloadedBytes': sum(d['bytes'] for d in downloads),
            'outputs': {name: sha_file(out / name) for name in ('config.json', 'selection.json', 'receipts.json', 'result.json') if (out / name).is_file()},
            'driver': {'file': 'scripts/evaluation/m7/paint_heldout.py', 'sha256': sha_file(__file__)},
            'common': {'file': 'scripts/evaluation/m7/common.py', 'sha256': sha_file(common.__file__)},
            'python': {'executable': sys.executable, 'version': sys.version.split()[0], 'packages': packages}, 'finishedUtc': now()}


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    parser.add_argument('--site', required=True, help='the run_heldout.py output folder of this site')
    parser.add_argument('--out', help='a new folder under .local/ (not with --record-interrupted)')
    parser.add_argument('--record-interrupted', action='store_true', help='held-out: record a marked paint run that ended without a paint record')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--protocol-lock-sha256')
    mode.add_argument('--rehearse', action='store_true')
    args = parser.parse_args()
    if args.record_interrupted and (args.rehearse or args.out):
        parser.error('--record-interrupted takes only --site and --protocol-lock-sha256')
    if not args.record_interrupted and not args.out:
        parser.error('--out is required')
    site_dir = Path(args.site).resolve()
    out = None if args.record_interrupted else Path(args.out).resolve()
    if out and (not out.is_relative_to(ROOT / '.local') or os.path.lexists(out)):
        parser.error('--out must be a new folder under .local/')
    record, record_path = site_record(site_dir)
    log, crossing = record['site']['logId'], record['site']['crossingId']
    if record['mode'].startswith('held-out') == bool(args.rehearse):
        parser.error('the site folder was built in the other mode')
    if not args.rehearse and record['protocolLockSha256'] != args.protocol_lock_sha256:
        parser.error('the site folder was built under another lock')
    entry, _ = common.guard(log, crossing, args.rehearse, args.protocol_lock_sha256)  # before any network access
    site = {'logId': log, 'crossingId': str(crossing), 'role': entry['role']}
    lost = None
    if not args.rehearse:
        if common.marked_folder('build', log, crossing) != rel(site_dir):
            parser.error('--site must be the folder the build marker names')
        names, lost_sha = common.lost_sites()
        if f'{log[:8]}/{crossing}' in names:  # statistics.lostSiteRule: committed before any held-out build
            parser.error('this site is on the committed lost-site list')
        lost = {'file': common.LOST_SITES_REL, 'sha256': lost_sha}
    if args.record_interrupted:
        folder = common.ended_run('paint', log, crossing)
        if any((folder / n).is_file() for n in ('paint-receipt.json', 'paint-stopped.json')):
            parser.error('the marked paint run has its paint record: there is nothing to record')
        folder.mkdir(parents=True, exist_ok=True)
        stop = SiteStop('interrupted', 'the marked paint run ended without a paint record (recorded afterwards with --record-interrupted)')
        digest = write_new(folder / 'paint-stopped.json', paint_doc(args, site, record_path, folder, 'not-scored', stop, [], [], {}, lost))
        print(json.dumps({'site': f'{log[:8]}/{crossing}', 'paint': 'not-scored', 'record': rel(folder / 'paint-stopped.json'), 'sha256': digest}))
        sys.exit(3)
    if not args.rehearse:
        common.run_marker('paint', log, crossing, out)  # painted once: a second run could hide an A fail
    out.mkdir(parents=True)
    steps = []

    def run(step, argv):
        start = time.time()
        result = subprocess.run(NICE + [sys.executable, '-I', '-B', *argv], cwd=ROOT, capture_output=True, text=True, env=common.child_env())
        for kind, text in (('stdout', result.stdout), ('stderr', result.stderr)):
            write_new(out / f'{step}-{kind}.txt', common.masked(text).encode())
        steps.append({'step': step, 'exit': result.returncode, 'seconds': round(time.time() - start, 2)})
        if result.returncode != 0:
            raise SiteStop(step, (result.stderr.strip() or result.stdout.strip())[-400:])

    downloads = []
    packages = {}
    try:
        packages = {name: version(name) for name in PACKAGES}
        if sha_file(ROOT / CODE) != CODE_SHA or sha_file(ROOT / PROTOCOL) != PROTOCOL_SHA:
            raise SiteStop('pins', 'paint code-v4 or protocol v4 differs from its pinned sha256')
        protocol = json.loads((ROOT / PROTOCOL).read_bytes())
        if protocol['parameters']['sweepSelection']['count'] != SWEEP_COUNT:
            raise SiteStop('pins', 'paint protocol v4 sweep count is not 15')
        if packages != PACKAGES:
            raise SiteStop('pins', f'numpy/scipy/pyarrow {packages} differ from paint v4\'s locked {PACKAGES}')
        # 1. config.json from the site's receipted downloads (the same files the scene build used)
        by_path = {d['publisherPath']: d for d in record['downloads'] if d['kind'] == 'metadata'}
        data = site_dir / 'data' / log
        wanted = dict(SOURCES)
        for role, marker in MAP_SOURCES.items():
            hits = [p for p in by_path if p.startswith('map/') and marker in p]
            if len(hits) != 1:
                raise SiteStop('config', f'the site has {len(hits)} receipted map files for {role}')
            wanted[role] = hits[0]
        sources = {}
        for role, publisher in wanted.items():
            receipt = by_path.get(publisher)
            path = data / publisher
            if receipt is None or not path.is_file() or sha_file(path) != receipt['sha256']:
                raise SiteStop('config', f'{publisher} is missing or differs from its site receipt')
            sources[role] = {'path': os.path.relpath(path, out), 'sha256': receipt['sha256'], 'bytes': receipt['bytes']}
        listing = [d for d in record['downloads'] if d['kind'] == 'listing' and d['name'] == 'lidar']
        if len(listing) != 1:
            raise SiteStop('config', 'the site has no single receipted lidar listing')
        listing_path = ROOT / listing[0]['path']
        if sha_file(listing_path) != listing[0]['sha256']:
            raise SiteStop('config', 'the lidar listing differs from its site receipt')
        sources['lidarListing'] = {'path': os.path.relpath(listing_path, out), 'sha256': listing[0]['sha256'], 'bytes': listing[0]['bytes']}
        config = {'schemaVersion': 1, 'logId': log, 'crossingId': str(crossing), 'split': entry['role'], 'sources': sources,
                  'provenance': (f'M7 ({"rehearsal" if args.rehearse else "held-out"}): public AV2 Sensor train objects downloaded by '
                                 f'run_heldout.py ({rel(record_path)}, sha256 {sha_file(record_path)}); each matched the public listing '
                                 'size and ETag (MD5 for single-part objects).')}
        write_new(out / 'config.json', config)
        # 2. code-v4 select (metadata and the listing only)
        run('select', [CODE, 'select', '--protocol', PROTOCOL, '--protocol-sha256', PROTOCOL_SHA, '--config', str(out / 'config.json'),
                       '--output', str(out / 'selection.json')])
        selection = json.loads((out / 'selection.json').read_bytes())
        if not selection.get('ok'):
            raise SiteStop('select', selection.get('reason') or 'paint v4 selection failed')
        if len(selection['chosen']) != SWEEP_COUNT:
            raise SiteStop('select', f"{len(selection['chosen'])} sweeps chosen, not {SWEEP_COUNT}")
        # 3. the chosen sweeps, each into its own new folder, receipted
        listed = common.read_listing(listing_path, log, 'sensors/lidar/')
        files = []
        for chosen in selection['chosen']:
            ts = str(chosen['timestampNs']); entry_ = listed.get(f'{ts}.feather')
            if entry_ is None or '-' in entry_['etag'] or entry_['bytes'] > MAX_SWEEP:
                raise SiteStop('sweeps', f'{ts}: not listed, multi-part, or over {MAX_SWEEP} bytes')
            path, receipt = fetch(common.BUCKET + entry_['key'], out / 'sweeps' / ts, f'{ts}.feather', MAX_SWEEP, entry_)
            downloads.append(receipt)
            files.append({'timestampNs': ts, 'url': receipt['url'], 'path': os.path.relpath(path, out), 'bytes': receipt['bytes'],
                          'etag': receipt['httpETag'], 'md5': receipt['md5'], 'sha256': receipt['sha256'], 'downloadedUtc': receipt['utc']})
        write_new(out / 'receipts.json', {'kind': 'sweep-download-receipts', 'logId': log, 'selection': 'selection.json',
                                          'totalBytes': sum(f['bytes'] for f in files), 'files': files})
        # 4. code-v4 measure
        run('measure', [CODE, 'measure', '--protocol', PROTOCOL, '--protocol-sha256', PROTOCOL_SHA, '--config', str(out / 'config.json'),
                        '--sweeps', str(out / 'receipts.json'), '--output', str(out / 'result.json')])
        outcome, stop = 'measured', None
    except SiteStop as error:
        outcome, stop = 'not-scored', error
    except Exception as error:  # anything unexpected still leaves a stop record; its message is not kept (it could hold a value)
        outcome, stop = 'not-scored', SiteStop('unexpected', type(error).__name__)
    doc = paint_doc(args, site, record_path, out, outcome, stop, steps, downloads, packages, lost)
    name = 'paint-receipt.json' if stop is None else 'paint-stopped.json'
    digest = write_new(out / name, doc)
    print(json.dumps({'site': f'{log[:8]}/{crossing}', 'paint': outcome, 'record': rel(out / name), 'sha256': digest}))
    sys.exit(0 if stop is None else 3)


if __name__ == '__main__':
    main()
