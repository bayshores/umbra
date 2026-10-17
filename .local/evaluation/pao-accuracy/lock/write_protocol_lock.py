"""M6 protocol lock-2 writer (Palo Alto accuracy program), revision 12.

Revision 12 (2026-10-16): candidate r11 failed a fresh review with one major finding: the closed NV-Grounding-DINO job
wrapper ran whatever GPU-query, Python and TAO commands its plan named, so a plan could start a program that requests
AV2 data before the download guards. The wrapper now accepts only the three fixed commands, and its child runner refuses
any other command before starting a process. No site, target, threshold, label, statistics or algorithm changes.

Revision 11 (2026-10-16): candidate r10 failed a fresh review with three major findings: car_shape.py opened caller-given
ground and listing paths before binding them to the checked log, the closed NV-Grounding-DINO job downloader accepted
any https URL, and standalone local readers have no preflight. The first two are fixed in code; the third is handled by
the data boundary (protected AV2 bytes arrive only through guarded network paths; none are on disk before approval).
No site, target, threshold, label, statistics or algorithm changes.

Revision 10 (2026-10-16): candidate r9 failed a fresh review with one major finding: the shared M7 download helpers
(common.curl and common.fetch) did not check the log their URL requests, so a direct call could download held-out data
without preflight. Both now run guard_url first (exact object or listing URL, frozen av2Split, full preflight with an
explicit lock sha256 for held-out/control logs, reserve refused). No site, target, threshold, label, statistics or algorithm changes.

Revision 9 (2026-10-16): candidate r8 failed a fresh review with one major finding: car_shape.py's URL guard read
the log from a query prefix, so a held-out object URL with a development prefix passed unguarded. The guard now
accepts only the exact sweep and listing URL forms. Only the car_shape.py pin, the builder protocol copy, the code
update record and the driver's protocol pin change. No site, target, threshold, label, statistics or algorithm changes.

Revision 8 (2026-10-16): candidate r7 failed a fresh review with one major finding: M7 preflight did not check
sceneBuilder.protocolPath against protocolSha256, so a changed builder protocol could start a held-out build. Preflight
now checks every <name>Path/<name>Sha256 pair. No site, target, threshold, label, statistics or development algorithm changes.

Revision 7 (2026-10-16): candidate r6 failed a fresh review with three major findings. The driver protocol pin is
checked against this writer, shape I/O binds and guards its log in-process, and the adapter preflights independently
of cached shape modules. No site, target, threshold, label, statistics or development algorithm changes.

Revision 6 (2026-10-16): the fresh read-only r5 review found one major finding: direct builder adapter and shape CLI
access could precede the pushed lock-2 anchors. These entry points now run the full M7 preflight without fetching;
drivers still fetch. M7 receipt/score reads are guarded before opening inputs. No site, target, threshold or rule changes.

Revision 5 (2026-10-16): Sean chose "Fix it, new lock" before any held-out access. AV2 URLs use each log's frozen
av2Split, and M7 uses lock-2 paths. The original approved writer is retained in lock/history/. COMMIT must be set by
the coordinator after committing the new pipeline; do not run candidate while it is a placeholder.

Revision 2 (2026-10-14) answers the fresh review of candidate r1 (lock/history/r1-review-2026-10-14.md: FAIL, 1 blocker,
7 major, 5 minor) and moves the pins to shape-v2, builder v4.3, placement protocol v3.4 and metric F shape-v2 (Sean,
2026-10-14: "More scans"). r1 and its writer are kept read-only in lock/history/.

Two steps, both from the repository root:
  candidate:  nice -n 10 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -B .local/evaluation/pao-accuracy/lock/write_protocol_lock.py candidate
      Checks every pinned file on disk and every pinned code file against git at COMMIT, writes
      lock-2/protocol-lock.candidate.json (approval null, writtenUtc from this machine's clock) and proves that builder
      v4.3's own check_protocol_lock accepts its structure (on a scratch copy with a placeholder approval).
  approve:    nice -n 10 /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -I -B .local/evaluation/pao-accuracy/lock/write_protocol_lock.py approve \
                  --candidate-sha256 <sha of the candidate Sean approved> --by Sean --utc <ISO UTC> --how "<how>"
      Refuses if protocol-lock.json, the record or the anchor already exists. Rechecks every pin, requires the candidate
      bytes Sean approved, writes the lock to a temporary file, runs check_protocol_lock on it, and only then links it
      to .local/evaluation/pao-accuracy/lock-2/protocol-lock.json (the adapter's fixed path) with O_EXCL semantics, read-only.
      Writes lock-2/protocol-lock.record.json (with this machine's UTC) and the tracked anchor ANCHOR (the lock sha256
      only). The coordinator commits and pushes the anchor before any held-out access; M7 and its guarded children check it.

No held-out, control or reserve file is read. Nothing is downloaded. Numbers are copied from decisions.md (Sean, Oct 13
and Oct 14), the frozen protocols and the development result files named in the text; nothing is computed from data here.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / 'lock-2'
CANDIDATE = OUT_DIR / 'protocol-lock.candidate.json'
LOCK = OUT_DIR / 'protocol-lock.json'
RECORD = OUT_DIR / 'protocol-lock.record.json'
ANCHOR_REL = 'scripts/evaluation/m7/anchors-2/pao-lock-anchor.json'  # tracked; holds the lock sha256 only
ANCHOR = ROOT / ANCHOR_REL
COMMIT = '9da3450a4cbf95b141a2a7d1d62614f2385773be'  # lock-2 r12 pipeline commit (pushed 2026-10-16)
VARIANT = 'v4-18.5'
BUILDER_VERSION = 4.3
LABEL = 'AV2-referenced, development-grade, not surveyed'

# Every frozen file the lock names, with the sha256 already recorded for it. The writer refuses on any difference.
PINS = {
    'split': ('.local/evaluation/pao-accuracy/split.json', '47f64fe007b51d75f4bf1c9891d940fd6393c777e0232a0a4f9428bbbeaafab4'),
    'splitLock': ('.local/evaluation/pao-accuracy/split.lock.json', '1bcfd5442e59f93640c76d93ebc3fab564c0418251b7c3a564687022b153800e'),
    'builderProtocol': ('.local/evaluation/pao-accuracy/lock-2/builder-protocol-v4.3.json', '3cc6d1c93fe298e00bee073694ff798149e9b65b0f5866587f1e30019f07c9ba'),
    'builderSourceProtocol': ('.local/evaluation/pao-accuracy/scene-builder-v4.3/builder-protocol-v4.3.json', '5dc0ab06ac1af4f57aa96b9cea4e451e62801d4084f9b3fbc011a2ccdd11bebe'),
    'builderFreeze': ('.local/evaluation/pao-accuracy/scene-builder-v4.3/builder-protocol-v4.3-freeze.json', 'ce282ff9ec953e49f45b7d4e63f9e2d4d18f1669b7eff45b2382faffcf4c2353'),
    'builderCodeUpdate': ('.local/evaluation/pao-accuracy/lock-2/builder-code-update.json', '5df3352a049bdf76ab60b8ab36d2de7b27358b9af7a023bf69335b85346c2bf5'),
    'builderAllowedChanges': ('.local/evaluation/pao-accuracy/scene-builder-v4.3/allowed-changes-v4.3.json', '0945593fb3a5af9f702c570c8edb14773639525efc1d545a4284d42546ecdb8a'),
    'builderCompareV42V43': ('.local/evaluation/pao-accuracy/scene-builder-v4.3/compare-v4.2-v4.3.json', '4711a75b7589029d5de0d76d1f2b0fc231562c9e23a32f0edd7784c93f59b49e'),
    'placementProtocol': ('.local/evaluation/av2-map-first-2026-10-11/protocol-v3.4.json', '330f1dcd6725201a624cf3c3e0e9c3bde41319b38a63ed5cf2e74b26c0dcab22'),
    'placementFreeze': ('.local/evaluation/av2-map-first-2026-10-11/protocol-v3.4-freeze.json', 'c0b0deab2e40bbcce2620d486d918e3a1c6fa89b6871bbe2332b405ea0139f4a'),
    'shapeProtocol': ('.local/evaluation/pao-accuracy/shape-v2/protocol.json', '70cbb24fabd1509bcb797d651ba2ec6c094c28c869b4a6ecd061f57a5106a95a'),
    'shapeProtocolLock': ('.local/evaluation/pao-accuracy/shape-v2/protocol.lock.json', '697b3d3a5057f8f1e2596429857952ea29bcdaab710dd07d45f48e24a70ddb86'),
    'shapeSplitAddendum': ('.local/evaluation/pao-accuracy/shape-v2/split-addendum.json', '953dae5f24b7ff9ac66fc53001d677bd60c159fbf647f39b9f7966d227f21838'),
    'shapeSplitAddendumLock': ('.local/evaluation/pao-accuracy/shape-v2/split-addendum.lock.json', '8118d9fbae0493d023911edf7b5dffb1a87eeb7ef4941909ad38e645b7a20805'),
    'metricFProtocol': ('.local/evaluation/pao-accuracy/metric-f-shape-v2/protocol-v4.json', 'ff981b219891543a5a0bca7067ebcc234ce0019536404032c5248a3fe623b2b7'),
    'metricFLock': ('.local/evaluation/pao-accuracy/metric-f-shape-v2/protocol-v4.lock.json', 'f6a1f9fd4f2ca7bdc4c7a8f0607ca479cf118503cb5a3302240524f2a606cbcf'),
    'paintProtocolV4': ('.local/evaluation/pao-paint/protocol-v4.json', 'f9a52b440180e84c67d78dfb963849da73d7131f7bcaaf9ea6d0e94fafdfe273'),
    'paintProtocolV4Lock': ('.local/evaluation/pao-paint/protocol-v4.lock.json', 'f3f2ce439cdbcc1cc8b92c872aaa379936094bae41810d62368f178957b34399'),
    # The paint v4 estimators as locked (git f9be0be). scripts/california/measure_paint_edge.py now holds v5 (development only).
    'paintCodeV4': ('.local/evaluation/pao-paint/code-v4/measure_paint_edge.py', 'bd2c6996551e03fc39160cbcb718f314467334979264816dca4f14fd0f92617a'),
    'isolationMoveRecord': ('.local/evaluation/pao-labels/isolation-moves-20261014T031706Z.json', '9e2a033d7980cdc30daea42df7e1755de0e7c5930f541a01fcbdba8f4d597a55'),
    'labelsProtocolV2': ('.local/evaluation/pao-labels/v2/protocol-v2.json', 'c35dff1792b737c039a7a324ea3f56e90876cc98e1c329c1a036b284c775367b'),
    'labelsProtocolV2Lock': ('.local/evaluation/pao-labels/v2/protocol-v2.lock.json', 'deed6614a6fce86c5791aa166abcb70a40ad1571e91baeb8007a6de1b6ed4d93'),
    'shapeV2Review': ('.local/evaluation/pao-accuracy/shape-v2-review-2026-10-14/README.md', '1aa115837cb1647e52251b3fead2c3f9915ac2289e43810b33be777cedd24594'),
}
# Written after r1; the writer refuses while any of these is None. paintV4Review: the fresh independent review that paint
# v4 scope.heldOutUse requires before any held-out paint measurement (verdict fix); paintV4ReviewFollowUp: the fresh M7
# re-review that checked each of its 10 findings (fixed or honestly worded; the two left open there, the same defects in
# the other drivers, were fixed in e6154dd and d2b5023 and checked by m7-review2 and m7-review3). labelsAddendum: the labels
# addendum (labels.scoringCondition), written after the label v2 gate, its review and the reviewed D/E code; it is left
# out only when NO_ADDENDUM holds Sean's recorded answer (the gate failed and he chose to run M7 without D and E).
LATER_PINS = {
    'paintV4Review': ('.local/evaluation/pao-paint/review-v4-heldout-2026-10-14/review/README.md',
                      'ebdba40c524704ad4acc7cb57b40d2d1ea0aab087c272f3229e6da91d8f22e6a'),
    'paintV4ReviewFollowUp': ('.local/evaluation/pao-accuracy/m7-review-2026-10-14/README.md',
                              '5817931a60cc4d629b5d0b445cf6dfeb722f8232dd2dfdefb73395166cebb414'),
    'labelsAddendum': ('.local/evaluation/pao-accuracy/lock/labels-addendum.json', 'c1c61b3d19cf5cea749fb1436de86a2e4a73f233e0603fa0cbd767713deb55a8'),
}
NO_ADDENDUM = None  # Sean's recorded answer (who, when, how, words) if the label gate fails and he chooses M7 without D and E
# The app and evaluation code that scores held-out rows, the shared pipeline step the builder does not pin, the metric I
# sentence set, and the M7 code (driver, held-out paint tool and metric E grid). Each must equal its committed version at COMMIT.
SCORING_CODE = ('scripts/evaluation/pao-baseline.mjs', 'scripts/evaluation/metric-f.mjs', 'scripts/evaluation/metric_f_car_top.py',
                'src/scene.mjs', 'src/suggestions.mjs', 'src/geometry.mjs', 'src/scenario-assistant.mjs',
                'scripts/california/assemble_av2_scene.py', 'tests/av2-ground.test.mjs', 'scripts/evaluation/metric_k.py')
M7_CODE = ('scripts/evaluation/m7/run_heldout.py', 'scripts/evaluation/m7/paint_heldout.py', 'scripts/evaluation/m7/common.py',
           'scripts/evaluation/m7/evaluate_site.py', 'scripts/evaluation/m7/score.py', 'scripts/evaluation/metric-e-grid.mjs',
           'scripts/evaluation/m7/build_scene.py')
# Label method v1 (gate failed) report, moved out of the repository on Oct 14 with the other answer-bearing files.
LABELS_V1_REPORT = ('$PAO_LABELS_COORDINATOR_DIR/pao-labels/pao-labels-v2/moved-from-repo/.local/evaluation/pao-labels/validation-report.json',
                    'b0bd33a7ca26f4fc88e584793cb21db3a51d53c2296592a09c1960bef89b3fba')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rel(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def fail(message):
    sys.exit(f'write_protocol_lock: {message}')


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def check_pins():
    out = {}
    for key, (path, expected) in {**PINS, **LATER_PINS}.items():
        if key == 'labelsAddendum' and NO_ADDENDUM is not None:
            if (ROOT / path).exists():
                fail('NO_ADDENDUM is set, but the labels addendum exists')
            continue
        if path is None or expected is None:
            fail(f'{key} is not written yet (LATER_PINS)')
        p = ROOT / path
        if not p.is_file():
            fail(f'{path} is missing')
        if sha(p.read_bytes()) != expected:
            fail(f'{path} changed after its sha256 was recorded')
        out[key] = {'path': path, 'sha256': expected}
    return out


def committed_code(paths):
    if COMMIT is None:
        fail('COMMIT is not set (the M7 code must be committed first)')
    code = {}
    for path in paths:
        disk = ROOT / path
        if not disk.is_file():
            fail(f'{path} is missing')
        committed = subprocess.run(['git', 'show', f'{COMMIT}:{path}'], cwd=ROOT, capture_output=True, check=False)
        if committed.returncode != 0 or sha(committed.stdout) != sha(disk.read_bytes()):
            fail(f'{path} differs from its committed version at {COMMIT[:7]}')
        code[path] = sha(disk.read_bytes())
    return code


def git_head():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def builder_code():
    protocol = json.loads((ROOT / PINS['builderProtocol'][0]).read_bytes())
    source = json.loads((ROOT / PINS['builderSourceProtocol'][0]).read_bytes())
    freeze = json.loads((ROOT / PINS['builderFreeze'][0]).read_bytes())
    if source['code'] != freeze['runtimeCode'] or freeze['sha256'] != PINS['builderSourceProtocol'][1]:
        fail('source builder protocol differs from its historical freeze record')
    code = committed_code(source['code'])
    changed = {path: value for path, value in code.items() if value != source['code'][path]}
    if set(changed) != {'scripts/california/car_shape.py'} or protocol != {**source, 'code': code}:
        fail('lock-2 builder protocol may change only the car_shape.py code pin')
    update = json.loads((ROOT / PINS['builderCodeUpdate'][0]).read_bytes())
    for key, pin in (('sourceProtocol', 'builderSourceProtocol'), ('sourceFreezeRecord', 'builderFreeze'), ('protocol', 'builderProtocol')):
        if update.get(key) != dict(zip(('path', 'sha256'), PINS[pin])):
            fail(f'builder code update names another {key}')
    adapter = 'scripts/evaluation/m7/build_scene.py'
    if update.get('changedCode') != changed or update.get('adapter') != {'path': adapter, 'sha256': sha((ROOT / adapter).read_bytes())}:
        fail('builder code update names another code pin or adapter')
    for path, value in code.items():
        if sha((ROOT / path).read_bytes()) != value:
            fail(f'builder runtime file {path} changed after builder v{BUILDER_VERSION} was frozen')
    if protocol.get('builderVersion') != BUILDER_VERSION or VARIANT not in protocol.get('parametersByVariant', {}):
        fail(f'builder protocol is not v{BUILDER_VERSION} or lacks the chosen variant')
    load_builder().check_frozen_protocol(protocol, PINS['split'][1], PINS['placementProtocol'][1], VARIANT)
    return code


def build_lock(pins, code, head, scoring, m7, written_utc):
    return {
        'schema': 'pao-protocol-lock-v1',
        'name': 'Palo Alto accuracy program: M6 protocol lock',
        'revision': 12,
        'status': 'CANDIDATE (not approved; no held-out run may use it)',
        'label': LABEL,
        'writtenUtc': written_utc,
        'repositoryHeadAtWriting': head,
        'authority': ('Sean, Oct 13 2026 (chat question tool): all 12 lock decisions (recommended answers), "use a measured car '
                      'shape", and the shape answers (accept the coverage cost, add a split check, keep drawing the box, '
                      'downloads, windows solid). Sean, Oct 14 2026 (chat question tool): "More scans" (shape-v2, one LiDAR '
                      'sweep every 0.25 m of ego travel), "Wait for labels v2", "Each street" (metric B passes per held-out '
                      'site; the pooled value is also reported). "Wait for labels v2" is applied as: the label v2 gate, its '
                      'review and the reviewed D/E code come before this lock, and the labels addendum is pinned inside it '
                      '(labels.addendum), so Sean approves them in one step. Sean, Oct 15 2026 (chat question tool): metric '
                      'E "Real spot only" (a grid of head points with the recorded vehicle at its recorded pose and the eye at '
                      'the recorded camera, metrics.E). Recorded in '
                      '.local/project/pao-accuracy-2026-10-12/decisions.md. Sean, 2026-10-16 (chat): "Fix it, new lock". '
                      'His lock-1 answers carry over unchanged: H measured-shape rows; H 50%; B all three hard; D/no '
                      'minimum; band 0.075 m; declared-default. His lost-site answer was "None".'),
        'history': ('Candidate r1 (sha256 aa19954f5ce70be07f0437e4e9166f7cfeed0a850e2f05593f6c5fc3740abf58, shape-v1 and builder '
                    'v4.2) failed a fresh read-only review on 2026-10-14 (1 blocker, 7 major, 5 minor) and was never shown to '
                    'Sean for approval; it is kept read-only in lock/history/ with the review. Candidate r2 (sha256 '
                    '8d7a8ea05918c048e4feacdc5057d0f133a14a8f5d10ad22ecc69ed9aa41108d, shape-v2 and builder v4.3) failed a '
                    'fresh read-only review on 2026-10-15 (2 major, 8 minor, 1 nit) and was never shown to Sean for approval; '
                    'it is kept read-only in lock/history/ with its writer and the review. Revision 3 fixes all 11 findings: '
                    'the labels addendum is pinned inside this lock, the approach split runs at M7, and the runs list, the '
                    'lock record and candidate checks and the approval text checks were added. It also moves metric E to '
                    'the recorded view (Sean, Oct 15: "Real spot only"), so H and referenceBlocked use metric F alone, and '
                    'pins the reviewed D/E code through the labels addendum. Candidate r3 (sha256 '
                    '51e90edf0c1319e7f8c69cec6af13c08a565f9d495181017d886e2cee9404a13) failed a fresh read-only review on '
                    '2026-10-16 (2 major, 1 minor) and was never shown to Sean for approval; it is kept read-only in '
                    'lock/history/ with the review. Revision 4 fixes all 3: referenceBlocked names the rows H counts (the '
                    'recorded vehicle\'s measured-shape setback-sweep rows, as score.py and the development numbers do), '
                    'and the held-out label task builder runs the M7 preflight and checks the D/E code before it reads any '
                    'site, takes roles from the frozen split (never a reserve site) and records a stopped site without a '
                    'task. It also writes out the development B results and that D has no minimum feature count. '
                    'Lock 1 (sha256 7aa2945e61e676ffc3a2a207895ca5639e4d9b4405893b648f4412891c00a475) was approved '
                    '2026-10-16 by Sean and superseded before any held-out access: its code read only the train folder '
                    'while split.json places 88f47a10, 25e5c600 and e1d68dde in val. Sean chose "Fix it, new lock" '
                    '(2026-10-16). Revision 5 uses each log\'s frozen av2Split folder, new lock/record/anchor/marker paths, '
                    'and a builder adapter with only the car_shape.py code pin replaced; every choice and scoring rule '
                    'stays unchanged. The approved lock-1 writer is read-only at '
                    'lock/history/lock1-approved-write_protocol_lock.py. Candidate r5 (sha256 '
                    'c48324f2a15ccd50569c3f696c00ab49996a75608769c35b20350d1269eea5f8) received verdict fix in a fresh '
                    'Codex read-only review on 2026-10-16: one major finding, direct builder adapter and shape CLI access '
                    'could reach held-out data before the lock-2 anchors were pushed. Candidate r5 and its writer are '
                    'read-only in lock/history/lock2-r5-protocol-lock.candidate.json and '
                    'lock/history/lock2-r5-write_protocol_lock.py. Revision 6 adds the full lock-2 preflight at those '
                    'entry points, and before M7 build receipt and score reads. Children check local origin/main without '
                    'fetching; the drivers keep fetching. No site, target, threshold or rule changed. Candidate r6 '
                    '(sha256 b483a5d394fc198d881b650b755b01598e0915c758b97fa788b5b19669b05296) failed its fresh Codex '
                    'read-only review on 2026-10-16 with three major findings: a stale driver builder-protocol pin, '
                    'shape plan/log substitution and direct helper access, and an adapter guard delegated to a cached '
                    'shape module. Candidate r6 and its writer remain read-only in lock/history/lock2-r6-*. Revision 7 '
                    'fixes all three with a writer-checked driver pin, guards and log bindings at shape I/O boundaries, '
                    'and the adapter\'s own fresh by-path M7 preflight. Scientific rules and development algorithms stay unchanged. '
                    'Candidate r7 (sha256 311388c4d2e89de4273e6a6800f8b78a6ee8ce19efb66152c84cb44b93a8bcb0) failed its fresh '
                    'Codex read-only review on 2026-10-16 with one major finding: M7 preflight did not check the pinned '
                    'builder protocol (sceneBuilder.protocolPath/protocolSha256). Candidate r7 and its writer remain read-only '
                    'in lock/history/lock2-r7-*. Revision 8 makes preflight check every <name>Path/<name>Sha256 pair in the lock. '
                    'Candidate r8 (sha256 6eda5ac7882b5c5296bdbd866ef21a59e2b432c4c1629aba50dafc43223959d8) failed its fresh '
                    'Codex read-only review on 2026-10-16 with one major finding: car_shape.py\'s URL guard trusted a query '
                    'prefix over the object path. Candidate r8 and its writer remain read-only in lock/history/lock2-r8-*. '
                    'Revision 9 accepts only the exact lidar sweep and listing URL forms. Candidate r9 (sha256 '
                    '28189ee75bfcad057e481cabac132e13aff9d900970b76467a1eca3f49d8a795) failed its fresh Codex read-only review on '
                    '2026-10-16 with one major finding: the shared M7 download helpers did not check the log their URL requests. '
                    'Candidate r9 and its writer remain read-only in lock/history/lock2-r9-*. Revision 10 guards every M7 '
                    'download URL (exact form, frozen av2Split, full preflight for held-out and control logs). Candidate r10 (sha256 '
                    '4ec520536013487c86472ad6ef1a56b152b7f73ad9bf1384db21ffae1f0ead05) failed its fresh Codex read-only review on '
                    '2026-10-16 with three major findings: shape input paths were opened before they were bound to the checked '
                    'log, the closed NV-Grounding-DINO job downloader accepted any https URL, and standalone local readers have no '
                    'preflight. Candidate r10 and its writer remain read-only in lock/history/lock2-r10-*. Revision 11 binds every '
                    'shape input path to its log before reading and limits the job downloader to the pinned model URLs; '
                    'standalone readers are covered by the data boundary (no protected file on disk before approval, every '
                    'AV2 network path guarded). Candidate r11 (sha256 '
                    '7412ad7bc27332f01e0f04ae0c6222deccaf046e2b75648ac2f089d950e5f157) failed its fresh Codex read-only review on '
                    '2026-10-16 with one major finding: the NV-Grounding-DINO job wrapper ran any GPU-query, Python or TAO command '
                    'named in its plan. Candidate r11 and its writer remain read-only in lock/history/lock2-r11-*. Revision 12 '
                    'accepts only the three fixed child commands and refuses any other before a process starts.'),
        'split': {**pins['split'], 'lock': pins['splitLock'],
                  'note': 'Frozen Oct 12 before any held-out image, LiDAR or calibration fetch. Annotation subsets of held-out '
                          'and reserve logs were read during site selection (split.json exposureBeforeFreeze).'},
        'sceneBuilder': {
            'protocolPath': pins['builderProtocol']['path'], 'protocolSha256': pins['builderProtocol']['sha256'],
            'freezeRecord': pins['builderFreeze'], 'allowedChanges': pins['builderAllowedChanges'],
            'sourceProtocol': pins['builderSourceProtocol'], 'codeUpdate': pins['builderCodeUpdate'],
            'freezeRecordRole': ('Historical freeze of sourceProtocol. The lock-2 protocol differs only in the car_shape.py '
                                 'code pin (codeUpdate); this lock freezes the replacement protocol and current code.'),
            'builderVersion': BUILDER_VERSION, 'code': code, 'variant': VARIANT,
            'frameRule': 'first feasible frame (the frame semantics builder v4/v4.1/v4.2/v4.3 implement); the driver eye starts near 18.5 m',
            'whyV43': ('Sean chose builder v4.1 on Oct 13 (item 6) as the reviewed version, before any car shape existed. His '
                       'later answers ("use a measured car shape", Oct 13; "More scans", Oct 14) need builder v4.3: the v4.1 '
                       'choice rules and parameters unchanged, placement protocol v3.4 (which replaces the v3.2 named on Oct 13, '
                       'through v3.3; it adds the recorded vehicle\'s carShape pinning shape-v2) and car_shape.py as new runtime '
                       'code. Three existing runtime files changed from v4.1: build_av2_scene.py, prepare_av2_ground.py and '
                       'suggest_av2_placements.py. Review status: builderCompareV42V43 passed (v4.3 makes the same frame and '
                       'scene choices as v4.2, which made the v4.1 choices, on every development run), and the shape-v2 build '
                       'had a fresh read-only review (PASS-WITH-FIXES, all four findings fixed). This lock asks Sean to approve '
                       'v4.3 explicitly.'),
            'whyV43Evidence': {'builderCompareV42V43': pins['builderCompareV42V43'], 'shapeV2Review': pins['shapeV2Review']},
        },
        'pipeline': {
            'commit': COMMIT,
            'placementProtocol': {**pins['placementProtocol'], 'freezeRecord': pins['placementFreeze']},
            'shapeProtocol': {**pins['shapeProtocol'], 'freezeRecord': pins['shapeProtocolLock']},
            'shapeSplitAddendum': {**pins['shapeSplitAddendum'], 'freezeRecord': pins['shapeSplitAddendumLock'],
                                   'role': 'report-only evaluation check; never changes ground.json or a result the app shows'},
            'scoringCode': scoring,
            'm7Code': m7,
            'scoringCodeRule': ('M7 runs from the main checkout (the ignored .local/ protocols and data exist only there) with an '
                                'empty git status. HEAD must contain pipeline.commit, and its tracked files may differ from it only '
                                'in the anchor files (scripts/evaluation/m7/anchors-2/pao-lock-anchor.json and scripts/evaluation/m7/anchors-2/). '
                                'The drivers (pipeline.m7Code), the builder adapter and car_shape.py held-out guard check that, '
                                'the lock record (lock-2/protocol-lock.record.json naming '
                                'this lock and its candidate), that this lock is byte for byte the candidate its approval names once '
                                'status and approval are put back as shown, the lock anchor on origin/main (written once), every '
                                'sha256 in this lock and the labels addendum (labels.addendum) before any held-out network or file '
                                'access, including reads of build receipts and scores. Drivers fetch origin/main; builder and '
                                'shape children run every check with fetch=False against the local origin/main tracking ref, '
                                'which must already hold the pushed anchors. Current run markers and output folders do not '
                                'invalidate preflight. Rehearsal refuses inputs inside marked held-out run folders before '
                                'opening them. Builder runtime code is pinned in sceneBuilder.code.'),
            'declaredDefaultTrack': ('A scene whose builder plan names no current-vehicle track (a declared default vehicle) gets '
                                     'currentVehicle.trackPrefix "declared-default" in its per-scene placement protocol. No AV2 '
                                     'track id starts with it, so the frozen suggest_av2_placements.py takes its declared-default '
                                     'path (a missing prefix would crash it). Set before any held-out access.'),
            'm7Steps': ('All from the main checkout with nice -n 10, pinned in pipeline.m7Code. Per site: run_heldout.py (scene '
                        'build). Per held-out crossing: paint_heldout.py (metric A). Per site: evaluate_site.py runs '
                        'pao-baseline.mjs --scene (B, C, I; J Node values), on a site with the recorded vehicle\'s LiDAR shape '
                        'metric F protocol v4 rows, profile (scoring sweeps only) and score (no --box-rows-from), then the '
                        'approach split (metrics.FApproachSplit: car_shape.py approach-split with --held-out-run, metric F rows '
                        'with that shape, profile on the near half, score next to the metric F results), then '
                        'metric_k.py, then (with the labels addendum) metric-e-grid.mjs (the metric E grid at the recorded '
                        'view, metrics.E) and the addendum-pinned D/E scorer (--site, --e-grid when the grid step passed, '
                        '--labels, --out), then score.py site (pass, fail or not scored per target, by the rules in metrics, '
                        'referenceBlocked and statistics). A failed scoring step is handled like a site with no feasible '
                        'frame for its own targets (dataAccess): B cannot pass, H is a miss at a Blocked-test site, the '
                        'others are not scored; a failed approach-split step leaves only FApproachSplit not scored, and a failed metric-e-grid step leaves only E not scored. On a site without the recorded vehicle\'s LiDAR shape H has no '
                        'reference-blocked row (listed by name, leaves the count) unless the app\'s child setback sweep is '
                        'all-Unknown (a miss). Each build receipt records the labels addendum sha256 (or null) the run works '
                        'under; score.py refuses a held-out site built under another value than the committed anchor. When the '
                        'labels addendum exists, score.py site also takes that site\'s D/E file from the addendum-pinned D/E '
                        'scorer (labels.scoringCondition) and refuses one made by other code, for another ground.json, under '
                        'another addendum, or (held-out) from labels other than the committed label manifest lists for the '
                        'site; a D/E file that repeats a JSON key is refused. evaluate_site.py keeps a D/E file only after '
                        'the same checks pass and its decoded text holds neither the coordinator-only folder nor the banned word (else its '
                        'labels-de step is refused, a failed step). Held-out, evaluate_site.py waits up to 1 hour for the git fetch of '
                        'origin/main before its score step and records the wait. Held-out, run_heldout.py, paint_heldout.py, evaluate_site.py and score.py need the committed '
                        'lost-site list (statistics.lostSiteRule) and refuse a lost site; evaluate_site.py runs only on the '
                        'folder the build marker names, with --paint the folder the paint marker names holding one paint '
                        'record made from that site record (checked before its own marker); score.py site and summary run '
                        'preflight (the approved lock, with this score.py pinned) and each site score records the sha256 of '
                        'the lost-site list, label manifest and runs list it was made under. After every held-out build and '
                        'paint run has its record and before any evaluate_site.py run, score.py runs-list writes the runs list '
                        '(statistics.oneRunRule), which the coordinator commits and pushes; evaluate_site.py and score.py site '
                        'refuse a site record or paint record it does not name. A marked run that ended (its process is '
                        'gone) without its own record is recorded once afterwards and never rerun: run_heldout.py '
                        '--record-interrupted (site-stopped.json, step "interrupted": a site with no feasible frame), '
                        'paint_heldout.py --record-interrupted (paint-stopped.json: A not scored), evaluate_site.py '
                        '--record-stopped (evaluate-stopped.json, used when the score step failed or no receipt was written; '
                        'score.py site --evaluation-stopped scores it into stopped-score.json like a site with no feasible '
                        'frame, with A from the marked paint record). Last: score.py summary over exactly the held-out sites '
                        'minus the committed lost-site list (scripts/evaluation/m7/anchors-2/lost-sites.json): B pooled, D per '
                        'site and pooled, misses with the Clopper-Pearson bound, false Blocked, the reported FApproachSplit '
                        'values, and by name every site the '
                        'label manifest lists with labels whose D or E is not scored. It refuses scores made by more than one '
                        'scorer or labels addendum, a held-out site without its paint record (A is fail-only: a missing paint '
                        'run could hide a fail), and, when the addendum is anchored, a site the label manifest lists with '
                        'labels but scored without its D/E file (unless its labels-de step failed, its evaluation stopped or '
                        'it has no built scene) or a D/E file the manifest does not list. It also refuses a site score made '
                        'under another lost-site list, label manifest or runs list than the committed ones, of another site '
                        'record or paint record than the runs list names, made by another score.py than the locked one, or not from the one marked run of each step: a scored site\'s failed steps '
                        'must equal the receipt of that evaluate_site.py run (a failed step is never taken from a bare '
                        'command-line claim), and a stopped site must come from that run\'s evaluate-stopped.json.'),
        },
        'resultRules': {
            'states': ['Blocked', 'Unknown (with a named reason)'],
            'clear': 'does not exist; the app never claims a person is visible',
            'singleRule': 'assessSightline in src/geometry.mjs (pinned in pipeline.scoringCode)',
            'tooCloseToCallM': 0.10,
            'unknownReasons': {'driver-inside-vehicle': 'driver inside a vehicle', 'person-inside-vehicle': 'person inside a vehicle',
                               'driver-inside-object': 'driver inside a real object', 'person-inside-object': 'person inside a real object',
                               'overlaps-real-object': 'overlaps a real object', 'too-close-to-call': 'too close to call',
                               'height-not-proven': 'vehicle height not proven'},
            'sceneRefusals': ('named codes in the reason text (for example crossing-skew-over-limit, '
                              'crossing-point-off-edge-segment); a refused scene gives Unknown, never a silent point'),
            'missingEvidence': 'Unknown; a missing intersection alone never means visible',
            'vehicleModel': ('Recorded stationary vehicle: a line that cuts >= 0.10 m into its box stays Blocked only if it '
                             'is also >= 0.10 m deep in the LiDAR shape (shape-v2: sweeps every 0.25 m of ego travel, at most '
                             '40, alternating shape and scoring); otherwise Unknown "vehicle height not proven". The shape never '
                             'adds a Blocked. Windows count as solid. The box is still drawn in the app. Declared, resized and '
                             'replacement vehicles: the box.'),
            'driverEye': 'camera height (1.47 m), labelled',
            'placementLimitM': 30,
            'declaredVehicle': 'declaring a different current vehicle replaces the real car (labelled)',
            'stationaryPose': 'average of all annotations (placement protocol v3)',
            'heldOutHeightBand': ('Placement protocol v3.4 alongRayUncertainty.heightBand.allowedOnlyFor: on a held-out scene the '
                                  'band is fixed, never refit. Fixed value: 0.075 m, fittedOn "8c52d911/2658849 app scene (grounded-v3.4, '
                                  'placement protocol v3.4)", the largest development fit (c780d53a/2069452 v4-18.5 0.047 m, 8c52d911 '
                                  'builder frames 0.069 and 0.070 m, 453e5558 guard check 0.066 m). Each held-out ground.json records '
                                  'its own residuals as a test of that band (heldOutResiduals); nothing is fitted on held-out data.'),
        },
        'clicks': {
            'locked': 'whole pixel: floor(uv + 0.5), the M0 definition, for metric B and the error budget',
            'diagnosticOnly': 'fractional (exact projection)',
        },
        'tooCloseToCall': {
            'marginM': 0.10,
            'statement': ('Kept at 10 cm and stated as not fully covering the worst case: the largest value of each small '
                          'steady error from any development scene sums to 0.125 m (straight) or 0.069 m (root-sum-square). '
                          'Per scene the straight sums are 0.050-0.063 m. Box shape, the LiDAR shape\'s own height error, frame '
                          'rule, click error and photo registration are not covered. Never tuned to development rows.'),
        },
        'referenceBlocked': ('Only the real recorded vehicle with its measured LiDAR shape (shape-v2): its child (1.1 m) rows of '
                             'the app\'s setback sweep, the rows metric F scores (the same vehicle moved along its lane axis '
                             'to each setback on the 0.25 m grid). These are the rows the development result in metrics.H '
                             'counts. Rows of declared vehicles and hypothetical default-SUV rows never count, at any site. '
                             'The recorded position itself is not on the row grid; its nearest row is reported, information '
                             'only. A site without the measured shape (built-box) has no counted rows. A child row is '
                             'reference-blocked when metric F (pinned protocol, scoring sweeps only) classes it "supported". F '
                             '"ambiguous" and F-unresolved rows are listed by name and are neither reference-blocked nor '
                             'reference-not-blocked. Metric E never changes a row\'s reference: its grid points are at the '
                             'recorded view, not at the setback rows (metrics.E). Metric K is never a reference source: it is '
                             'report-only evidence against Blocked.'),
        'metrics': {
            'A': ('Fail-only check. At each held-out crossing run both frozen paint v4 LiDAR estimators (paint protocol v4, '
                  'code paint.codeV4, 15 up_lidar sweeps chosen by paint v4\'s own sweep selection), unchanged, after this lock, through the '
                  'held-out paint tool pinned in pipeline.m7Code: it reuses paint v4\'s exact parameters and refuses to run '
                  'without this approved lock. Paint v4 scope.heldOutUse requires a fresh independent review of paint v4 '
                  'before held-out use: paint.v4Review. Offset = each estimator\'s edgeSAtEval in paint v4\'s result.json: '
                  'meters along the approach travel direction from the mapped near edge to the paint onset, positive when the '
                  'paint starts past the mapped edge, read at result.json frame.evalL; score.py refuses a result.json whose '
                  'protocolSha256/scriptSha256 are not paint v4\'s. A fails at that crossing only when both |offset| >= 0.30 m, '
                  'they agree within 0.15 m and paint v4\'s own decision (nearEdge.status, which adds its precision rule) is '
                  '"resolved"; would-be fails that paint v4 does not resolve are listed as not confirmed by paint v4 (not '
                  'scored). Disagreeing by more than 0.15 m: "estimators disagree" (counted, not scored). Otherwise: not scored '
                  '(no validated paint method); both offsets reported. A never passes: "not scored" says nothing about the map, '
                  'and the summary states how many crossings were scored. A fail means the true offset is very likely at least '
                  'about 0.20 m, not proven to be 0.30 m. A look-alike line before the crossing (a bright band, a stop line, a '
                  'faint first stripe) can fool both estimators the same way; such a fail still counts against the app and is '
                  'reported with this caveat (Sean, Oct 14: "count it against us"; no blind photo check). A paint run that '
                  'stopped after paint v4 wrote its result.json is still not scored and is listed by name (stopped after '
                  'measuring). No paint v6 (Sean, Oct 13).'),
            'B': ('Click-to-ground error at 12-25 m, whole-pixel clicks, all three parts hard: (1) p95 <= 0.76 m; (2) the '
                  'shown band (fitted on development) covers >= 95% of held-out errors; (3) the shown band is no wider '
                  'than +/-0.76 m at every placed point (median and p95 also reported). Computed per held-out site: every '
                  'held-out site with a built scene (Blocked-test sites, the left-side case and the controls) must pass all '
                  'three parts on its own (Sean, Oct 14: "each street"); the pooled values over those sites are also '
                  'reported. The shown band uses resultRules.heldOutHeightBand. A site without a built scene is listed by name '
                  'and B cannot pass. Placement beyond 30 m '
                  'refused. Failures are reported, not relaxed. Development rehearsal (the M7 scoring code on the two '
                  'development sites, m7-rehearsal-2026-10-14/8c52d911-2658849-r3-eval/score.json and '
                  'c780d53a-2069452-eval/score.json; the band was fitted on development, so this is not a held-out '
                  'measurement): 8c fails parts (1) and (3) (p95 0.897 m, band max 1.709 m; coverage 0.966 passes) and '
                  'c780 fails part (2) (coverage 0.887; p95 0.756 m and band max 0.732 m pass). So B may fail at '
                  'held-out sites.'),
            'C': ('Suggestion points vs their source points <= 0.05 m for the point the app uses. Internal consistency '
                  'only, never called accuracy. Vehicle-front pixel-ray distance reported as a diagnostic, no target.'),
            'D': ('Vehicle registration in the photo: median <= 3 px and max <= 6 px at 1024 px width over the left edge, '
                  'right edge and roofline errors: the aggregated labels L, R and the resolved roofline stations S1-S3, each '
                  'against the app\'s recorded-vehicle model as ground.json defines it (the LiDAR shape where measured, else '
                  'the box) projected into the recorded photo. Unresolved labels are left out and counted; each of L, R, '
                  'S1, S2 and S3 is given once, either as a finite error or as unresolved. There is no minimum number '
                  'of resolved features: one is enough to pass or fail, and a site with none is not scored. Computed per '
                  'held-out site with a labelled recorded vehicle; each such site must pass on its own, as for B; pooled '
                  'values also reported. Scored only under labels.scoringCondition, by the D/E code the labels addendum '
                  'pins; otherwise "not scored: reference not precise enough".'),
            'E': ('Result at the recorded view vs a blind photo outline, with the recorded vehicle only at its recorded pose '
                  '(Sean, Oct 15: "Real spot only"). Grid: metric-e-grid.mjs (pipeline.m7Code; rule text E_GRID_RULE, '
                  'from the loaded scene alone, never labels or a photo outline): ground points on a 0.25 m world X/Z grid '
                  'with mapped ground under them, from the farthest corner of the recorded vehicle\'s box footprint plus '
                  '0.5 m to that plus 10 m from the eye, inside the footprint corners\' bearings widened by 2 degrees each '
                  'side, each with a child (1.1 m) and an adult (1.5 m) head point above the mapped ground. The eye is the '
                  'recorded camera and the vehicle is the recorded vehicle at its recorded pose with its car shape (the '
                  'suggested setup at the full captured distance); each grid row holds the app\'s own result '
                  '(assessSightline, the call the app\'s current result makes). With the eye at the camera, the eye-to-head '
                  'line is the photo ray through the head point\'s pixel. Each row\'s head point is projected into the '
                  'recorded photo (stereo_front_left at the built scene\'s cameraTimestampNs, camera from the build-pinned '
                  'prepared/manifest.json) and compared against the aggregated labels of the recorded vehicle (L, R, T, '
                  'B and roofline stations S1-S3). Margin m = 3 px at 1024 px width (6 px full '
                  'resolution) plus the largest stated label range involved. Resolved not blocked: left of L - m, right of '
                  'R + m, above T - m, or between two neighbouring resolved stations and above both by m. Resolved blocked: '
                  'between two neighbouring resolved stations, below both by m, above B - m, with the vehicle nearer the '
                  'camera than the target. Every other point is unresolved and listed (also a point outside the photo or '
                  'behind the camera). Between two stations the roofline is taken to stay between their two heights; every '
                  'E result states this assumption. Target: 0 false Blocked (app plain Blocked where E is resolved not '
                  'blocked) per site; a site with no resolved point is not scored. The only check of a shape Blocked '
                  'independent of the LiDAR; small sample (one recorded view per scene, at the recorded pose only). E\'s '
                  'rows are exactly the grid\'s rows (one per row, with that row\'s app state). If the grid\'s eye is not '
                  'the photo camera centre within 0.01 m, or the grid does not name the recorded vehicle at its annotated '
                  'box (its track, centre, axis, length, width, height and car shape identity), every row is unresolved and E is not scored, with that reason. A site without an annotated '
                  'recorded vehicle, or with a failed metric-e-grid step, has no E (not scored). Scored only under '
                  'labels.scoringCondition, by the D/E code the labels addendum pins; otherwise not scored.'),
            'F': ('Metric F shape-v2 (protocol v4, pinned in metricProtocols.F): consistency and code cross-check of the '
                  'vehicle model, not accuracy evidence for the recorded vehicle\'s shape; zero likely-false rows is '
                  'expected by construction. Reference: LiDAR car top from carShape.sweeps.scoring only (alternating split). '
                  'Laser rule: F\'s frozen laser check (0.15 degrees) applies to each scoring sweep; a failing sweep is '
                  'dropped and listed, and with fewer than 2 scoring sweeps left every row of that site is F-unresolved. '
                  'Target: 0 likely-false rows per site (a failure means a bug). Unresolved rows are listed and never '
                  'counted as passes.'),
            'F0': ('No new Blocked: every shape Blocked row is box Blocked, and every row obeys the shape-v2 allowed-change '
                   'table. Internal check. Target: 0 violations.'),
            'FApproachSplit': ('Report only (shape-v2 split addendum, pinned in pipeline.shapeSplitAddendum): at every '
                               'held-out site with the recorded vehicle\'s LiDAR shape, after metric F, an evaluation-only '
                               'shape built from the far half of the approach (car_shape.py approach-split --held-out-run, '
                               'from that site\'s hashed sweep plan) is scored on the near half with metric F protocol v4 '
                               'and compared with the app\'s metric F results. Per height: rows, box and shape Blocked, '
                               'supported and lost rows, app Blocked where the approach shape is height-not-proven, and the '
                               'reverse. Never a pass or a fail, never read by another target, never changes a result the '
                               'app shows. A failed approach step leaves only FApproachSplit not scored: it is never a miss '
                               'and changes no other target. Sean chose this check on Oct 13 ("add a split check").'),
            'H': ('Useful coverage: at each Blocked-test site at least half of the reference-blocked child (1.1 m) rows '
                  '(referenceBlocked: the recorded vehicle\'s measured-shape setback-sweep rows) are reported Blocked. No Blocked-test site may be all-Unknown. A Blocked-test site '
                  'with no feasible frame fails H (a miss). Adult (1.5 m) reported, no target. Development result, locked '
                  'variant v4-18.5, shape-v2 (tuning set, metric-f-shape-v2/README.md): of the child rows F supports, the '
                  'app keeps 21 of 36 Blocked on 8c (58%) and 5 of 17 on c780 (29%); with the box alone it kept all of them. '
                  'At the 50% target H passes on 8c and fails on c780, so H may fail at held-out sites. Sean accepted '
                  '"child keeps about 65-86%" on Oct 13 before these measurements.'),
            'I': ('Honesty: 0 Blocked with the person or driver inside a box; 0 real-object overlaps reported as Blocked; '
                  '0 out-of-range inputs accepted; 0 sentences of the fixed bad-sentence set pass the answer check. The set '
                  'is the 11 "wording" and 4 "role" sentences in tests/av2-ground.test.mjs (pinned in pipeline.scoringCode), '
                  'with each number replaced by the held-out site\'s own value of the same role: 0.42 m by the suggested '
                  'setback, 1.49 m by the recorded vehicle\'s height and 1.88 m by its width; the three values written '
                  'to two decimals must differ from each other, else I fails (a role sentence could be true). Each must '
                  'fail with the same kind ("wording" or "role"). Counts over 3000 seeded placement samples (fewer, or a sweep error, leaves I '
                  'not scored unless a count is already above 0); each out-of-range case must be refused by its own '
                  'field\'s rule. A scene without a suggested setup leaves I not scored.'),
            'J': ('Speed on Sean\'s Mac: first result <= 500 ms after a cold page load; Drag/Compare update p95 <= 16.7 ms; '
                  'setback sweep p50 <= 5 ms with the shape, measured alone.'),
            'K': ('LiDAR-shadow check, report only (scripts/evaluation/metric_k.py, pinned in pipeline.scoringCode): the '
                  'frozen protocol v3.1 lidarShadow rule, which every scene protocol carries unchanged, on the recorded '
                  'vehicle\'s tested box. Per site: rays through the box (K1) with cut depth p50/p95/max, and unboxed returns '
                  '(K2), from the scene sweep and, separately, from the scoring sweeps the shape result is scored with. A '
                  'scene without a recorded-vehicle box is not scored. K classifies no row: Blocked rows with LiDAR evidence '
                  'against them are metric F\'s likely-false rows, listed in the F results. Never a reference source '
                  '(referenceBlocked).'),
            'errorBudget': ('Combined (root-sum-square) setback error <= 0.76 m for suggested positions; dragged positions '
                            'reported, no target. Real car height is modelled by the LiDAR shape (shape-v2) for the recorded '
                            'vehicle; declared vehicles keep the box. Today: not established (lower bounds only).'),
            'stretch': 'Report only: road direction vs lane paint <= 1.0 deg; curb estimate vs LiDAR curb step <= 0.20 m.',
            'outcomes': 'Each target is written as pass, fail, or not scored (with its reason).',
        },
        'metricProtocols': {'F': {**pins['metricFProtocol'], 'lock': pins['metricFLock']}},
        'paint': {'protocolV4': {**pins['paintProtocolV4'], 'lock': pins['paintProtocolV4Lock']},
                  'codeV4': pins['paintCodeV4'], 'v4Review': pins['paintV4Review'],
                  'v4ReviewFollowUp': {**pins['paintV4ReviewFollowUp'],
                                       'note': ('the v4 review returned fix; its 10 findings were applied in de163f1 and '
                                                'checked by this fresh M7 re-review (section "The 10 paint v4 review '
                                                'findings"); the two it left open were fixed in e6154dd and d2b5023 (run '
                                                'markers), checked by m7-review2 and m7-review3')},
                  'appBound': 'the app keeps the mapped crossing edge with the v4 bounds (8c +/-0.18 m, c780 +/-0.26 m)',
                  'note': 'scripts/california/measure_paint_edge.py now holds paint v5 (development only, not for held-out use)'},
        'labels': {
            'v1': {'gate': 'FAILED on coverage (resolved share 0.055 vs 0.70)', 'report': {'path': LABELS_V1_REPORT[0], 'sha256': LABELS_V1_REPORT[1]}},
            'v2': {'protocol': pins['labelsProtocolV2'], 'lock': pins['labelsProtocolV2Lock'],
                   'gate': ('protocol-v2.json scoring.syntheticGate and lidarCoarse, in full-resolution pixels: resolved '
                            'share >= 0.70; at least 10 exact aggregated features in each family (car extremes, roofline '
                            'profile, tire contacts, paint); car extremes, roofline profile and tire contacts median '
                            '|error| <= 2 px and max <= 4 px; paint median <= 1 px and max <= 2 px; LiDAR coarse check '
                            'consistent share >= 0.80 with an 8 px registration allowance. This lock adds: repeat-label '
                            'spread <= 3 px at 1024 px width (6 px full resolution), the largest |original - blind repeat| '
                            'over the resolved features of the repeat cases.')},
            'addendum': None if NO_ADDENDUM is not None else pins['labelsAddendum'],
            'noAddendum': NO_ADDENDUM,
            'scoringCondition': ('D and E are scored only under the labels addendum this lock pins (labels.addendum), which '
                                 'Sean approves together with this lock. It was written before this lock, after all of these '
                                 'held: (1) label protocol v2 (labels.v2, locked with lock_protocol.py) ran on its development '
                                 'tasks; (2) its gate passed (labels.v2.gate, with the repeat-spread limit this lock adds); (3) '
                                 'a fresh read-only review of that gate result passed; (4) the D/E code (the held-out label '
                                 'task builder and the D/E scorer) was written after the gate passed and passed a fresh '
                                 'read-only review. The addendum (lock/labels-addendum.json) holds only the v2 protocol sha256, '
                                 'its lock record, the gate result, the gate review record and the D/E code pins (with the record of the D/E code review). Its sha256 is '
                                 'also committed and pushed in the tracked labels addendum anchor '
                                 '(scripts/evaluation/m7/anchors-2/labels-addendum-anchor.json) before M7 starts; preflight '
                                 'refuses unless the addendum file, that anchor and labels.addendum agree. labels.addendum is '
                                 'null only if the gate failed and Sean answered to run M7 without D and E (labels.noAddendum '
                                 'holds his recorded answer); then there is no addendum file and D and E are "not scored: '
                                 'reference not precise enough". The D/E code is '
                                 'written only after the gate passes, implements metrics.D and metrics.E as written here, names '
                                 'the sha256 of the labels it used, and its output is the only D/E input score.py site accepts '
                                 '(checked against the addendum pins). The D/E code lives in the ignored folder '
                                 '.local/evaluation/pao-accuracy/lock/de-code/: the addendum pins every file in it (deCode.files: '
                                 'repository path and sha256; nothing else in the folder, no link) and names the scorer '
                                 '(deCode.scorer, one of those files); evaluate_site.py and score.py refuse when the folder '
                                 'differs. The scorer runs with python -I -B and can import only the standard library, that '
                                 'folder and files this lock pins. It writes into a new temporary folder outside the repository; '
                                 'its output is kept (labels-de.json) only if it names neither the coordinator-only folder nor '
                                 'the labels folder, otherwise the labels-de step is recorded as refused (a failed step). The '
                                 'output has exactly the keys kind, site ({logId, crossingId} strings), groundSha256, '
                                 'labelsAddendumSha256, labels ({sha256} only), scorer, D and E. Held-out labels: after every '
                                 'held-out site is built and before evaluate_site.py runs anywhere, one car task per site whose '
                                 'recorded vehicle is in the builder\'s photo (task region from the AV2 annotation cuboid, never '
                                 'an app output), made by the addendum-pinned held-out label task builder from the build-pinned '
                                 'photo (stereo_front_left at the built scene\'s cameraTimestampNs, checked against the build\'s '
                                 'prepared/manifest.json), under anonymous task ids (the site map is written only inside the '
                                 'coordinator-only folder), labelled, scanned, aggregated and hashed exactly as in the gate, one '
                                 'task list and one aggregate per site; the label '
                                 'manifest (scripts/evaluation/m7/anchors-2/labels-manifest.json: the addendum sha256 and every '
                                 'held-out site once, with its label aggregate\'s sha256, or null and the reason) is committed and '
                                 'pushed before any held-out app result is computed or viewed by the coordinator or a labeler. '
                                 'evaluate_site.py takes --labels exactly for the sites the manifest lists with labels, with that '
                                 'sha256, from inside the coordinator-only folder named by PAO_LABELS_COORDINATOR_DIR.'),
            'method': ('Fresh AI agents that never see the app output draw the references (never called human labels); one '
                       'fresh agent per (labeler, task); labels hashed before scoring.'),
            'answerIsolation': {'rule': ('Answer-bearing files live outside the repository in the coordinator-only folder named '
                                         'by PAO_LABELS_COORDINATOR_DIR (v1 files moved Oct 14).'),
                                'moveRecord': pins['isolationMoveRecord']},
        },
        'sites': {
            'heldOutBlockedTests': ['88f47a10/2659268', '3153b5b3/2659599', 'd33f667d/2399287'],
            'heldOutLeftSideCase': 'fd5c6932/2659589 (expected Unknown)',
            'heldOutControls': ['25e5c600/2659878', 'e1d68dde/5987490 (declared vehicles only)'],
            'development': ['8c52d911/2658849', 'c780d53a/2069452 (second development crossing)'],
            'doesNotApply': '453e5558/20499018 (expected Unknown with crossing-skew-over-limit and crossing-point-off-edge-segment)',
            'reserve': 'split.json reserve sites stay unopened; they cannot replace a Blocked test',
            'plannedMinimumNotMet': 'split.json planned 4 Blocked-test sites and has 3; no untouched intersection is left to add',
        },
        'dataAccess': [
            'Before any held-out access: this lock is approved; its sha256 is recorded in current.md and committed in the '
            'tracked anchor (anchor) and pushed to the private remote. The labels addendum anchor '
            '(scripts/evaluation/m7/anchors-2/labels-addendum-anchor.json: the sha256 labels.addendum pins, or null only under '
            'labels.noAddendum) is committed and pushed too; the drivers refuse unless the addendum file equals it and '
            'labels.addendum (null: no addendum file). The builder adapter and shape held-out guard run the same full '
            'preflight before opening data; their fetch=False checks require the pushed anchors in local origin/main. '
            'M7 checks before opening build receipts or scores too.',
            'M7 step 1 (logged): fetch the public S3 listings for each held-out log from its sha-checked split.json av2Split folder (train or val only; reserve and test refused). Every metadata, sensor and shape download and listing/sweep-plan check uses that same folder, never a guess, default or network probe.',
            'Read each held-out log\'s calibration (about 10 KB per log) to confirm the stereo setup; the log metadata '
            'files are downloaded before this check. A site whose log fails the check is recorded as not built (as a site '
            'with no feasible frame): B fails for it, a Blocked-test site counts as a miss, and the other targets are '
            '"not scored: calibration".',
            'Per site (run_heldout.py): the log metadata the frozen builder reads (annotations, poses, calibration, intrinsics, '
            '3 map files; about 4 MB for a development log, 16 MB cap per file), then one stereo pair and one LiDAR sweep for '
            'the builder\'s chosen frame. Per held-out crossing (paint_heldout.py, on the same receipted metadata): the 15 '
            'sweeps that paint v4\'s selection names (about 1 MB each; plan.md said 10, paint v4 fixed 15). A paint step '
            'that fails leaves metric A "not scored" at that crossing only.',
            'Shape sweeps: at most 40 per held-out site by the frozen shape-v2 sweep rule (ego within 30 m, >= 0.25 m ego '
            'travel between kept sweeps, alternating shape/scoring); the sweep plan is hashed before any download. Shape '
            'sweeps use car_shape.py\'s frozen attempt rule (4 attempts, waits 5/10/20 s); a sweep that never downloads is '
            'recorded as dropped, as in shape-v2, and does not stop the site.',
            'Every download has a size/ETag/SHA-256 receipt. A failed or mismatched download may be retried at most 3 times, '
            'each into a new empty folder; a file that never matches stops that site, which is then handled like a site '
            'with no feasible frame. Any other step that fails stops that site with the step named; it is not rerun and is '
            'handled like a site with no feasible frame (a miss for a Blocked-test site). Public AV2 files only, $0. Run the '
            'frozen builder and pipeline once: each held-out driver (run_heldout.py, paint_heldout.py, evaluate_site.py) '
            'writes a run marker for the site (.local/evaluation/pao-accuracy/m7/lock-2/run-markers/, new files only) naming its '
            'output folder and its process before that folder is made, and refuses a second run; the summary refuses a site '
            'score that is not from the folders the markers name. A marked run that ended (its process is gone) without its '
            'own record (for example a crash or a power loss) is recorded once afterwards with the same driver '
            '(run_heldout.py and paint_heldout.py --record-interrupted, evaluate_site.py --record-stopped; pipeline.m7Steps) '
            'and handled like a site with no feasible frame (paint: A not scored); it is never rerun. The git fetch of '
            'origin/main that the held-out drivers run is tried at most 3 times, 10 s apart, each stopped after 120 s '
            'together with every process it started. '
            'Builder and shape children repeat every preflight check with fetch=False against local origin/main, '
            'so a later network outage cannot stop a site at these checks. '
            'Every record and log is written whole or not at all (a temporary file, flushed to disk, then linked to its '
            'new name). Step logs, error text and every written file keep the coordinator-only labels folder out, as '
            'given, resolved or relative to the repository, in any letter case or JSON escaping (replaced by a '
            'placeholder; text that still names it is withheld whole; a file that names it is refused); no step is '
            'given the environment variable that names it, and evaluate_site.py, when given held-out labels, refuses '
            'that folder or the temporary folder inside the repository.',
        ],
        'statistics': {
            'unit': 'one intersection (queries at one site are correlated)',
            'falseBlocked': ('When metric E is scored: the app reports plain Blocked for at least one E grid point at a site '
                             'where E shows that head point resolved not blocked. Real recorded vehicles only. Counted '
                             'over every held-out site with at least one E-resolved result for a real recorded vehicle; a site '
                             'with none is listed by name and left out of the count. When E is not scored, false Blocked is '
                             '"not measured independently": no bound is stated, and metric F\'s likely-false count is reported '
                             'only as a same-LiDAR consistency check.'),
            'miss': ('A Blocked-test site that fails metric H, including an all-Unknown site and a site with no feasible '
                     'frame. Counted over the Blocked-test sites; a site with no reference-blocked child row is listed by '
                     'name and leaves the count.'),
            'method': ('One-sided 95% exact binomial (Clopper-Pearson), separately for each error kind, from the counts '
                       'after scoring; never a number fixed in advance. With 0 errors in k sites the upper bound is '
                       '1 - 0.05^(1/k): 3 sites 63.2%; with 1 or 2 misses at 3 sites 86.5% or 98.3%.'),
            'lostSiteRule': ('Before the first held-out build the coordinator writes the list of lost sites (possibly empty) '
                             'as scripts/evaluation/m7/anchors-2/lost-sites.json, each entry with logId, crossingId, evidence '
                             '(non-empty text) and date (YYYY-MM-DD, no later than the UTC day of the list\'s one commit), '
                             'commits it and pushes it; run_heldout.py, paint_heldout.py, evaluate_site.py and score.py refuse '
                             'without it and refuse a lost site. Each build and paint record holds the list\'s sha256, and '
                             'score.py refuses one made under another list. A site is lost only on evidence dated before that commit that it was opened before this lock or guided a code '
                             'change; the pre-freeze annotation exposure in split.json exposureBeforeFreeze does not make a '
                             'site lost. A lost site moves to development and leaves both counts; it cannot be replaced, so '
                             'the bound widens. The list is written once, so no site can be declared lost after it is committed. A site run once with '
                             'no feasible frame is not lost: it stays in the miss count. Any code change after the first '
                             'scoring run is exploratory: its results are labelled exploratory and never replace the first '
                             'result. Replaces plan.md M7 line "A lost Blocked site cannot be replaced, so the bound widens '
                             '(2 sites = 77.6%)".'),
            'oneRunRule': ('Each held-out site is built, painted and evaluated once (run markers, dataAccess). Because the '
                           'markers live in an ignored folder, the runs are also anchored: after every held-out build and '
                           'paint run has its record (a marked run that ended without one is recorded first) and before any '
                           'evaluate_site.py run, score.py runs-list writes scripts/evaluation/m7/anchors-2/runs.json, each '
                           'held-out site minus the lost-site list once with the sha256 of the one site record and the one '
                           'paint record in the folders its markers name; the coordinator commits and pushes it. It is '
                           'written once. evaluate_site.py and score.py site refuse a site record or paint record it does '
                           'not name, every site score records its sha256, and the summary refuses a score under another '
                           'runs list or of another record. A site run again after a marker was removed therefore cannot '
                           'be scored.'),
        },
        'gate': ('The visibility gate is recorded as incomplete, never as passed. Its bound is stated from the scored '
                 'counts, using statistics.method.'),
        'changeRule': ('After approval nothing in this lock changes; protocol-lock.json, its record and the anchors are '
                       'never deleted, edited or overwritten. Any change needs Sean\'s approval of a new lock, made this '
                       'way: a fresh pipeline commit with new M7 code that reads the new lock at its own new path (a new '
                       'lock folder, its own record and its own tracked anchor files) and refuses this one. No target, threshold, '
                       'site, rule or pinned file changes after any held-out file is opened. Results are reported as '
                       'measured, labelled "' + LABEL + '".'),
        'anchor': {'trackedFile': ANCHOR_REL,
                   'rule': ('approve writes the lock sha256, a schema name and the recording time (nothing else from the '
                            'lock) to this tracked file; the coordinator commits and pushes it to the private remote before '
                            'any held-out access. The M7 drivers, builder adapter and shape held-out guard refuse unless '
                            'origin/main holds that sha256 and exactly one '
                            'commit on origin/main touches the file. approve refuses approval text (--by, --utc, --how) that '
                            'names the coordinator-only folder or holds the banned vehicle-status word, and a --utc later than '
                            'now. The coordinator commits and pushes the labels addendum anchor, the lost-site list, the '
                            'held-out label manifest and the runs list under scripts/evaluation/m7/anchors-2/ '
                            '(labels-addendum-anchor.json, lost-sites.json, labels-manifest.json, runs.json); the drivers '
                            'refuse each unless it is committed unchanged, equal to origin/main and written once (exactly one '
                            'commit on origin/main touches it), so a change after a result is seen makes every driver refuse. '
                            'The lost-site list is committed before any held-out build (statistics.lostSiteRule) and the runs '
                            'list before any evaluation (statistics.oneRunRule). The ignored D/E code folder is not an anchor: the labels '
                            'addendum pins every file in it (labels.scoringCondition). After pipeline.commit no other tracked '
                            'file may change.')},
        'approval': None,
    }


def load_builder():
    spec = importlib.util.spec_from_file_location('m7_build_scene', ROOT / 'scripts/evaluation/m7/build_scene.py')
    tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
    return tool.load_builder()


def builder_accepts(path, lock_sha, code):
    tool = load_builder()
    tool.LOCK_PATH_V4 = Path(path)
    return tool.check_protocol_lock(ROOT / PINS['split'][0], lock_sha, PINS['split'][1], PINS['builderProtocol'][1], code,
                                    BUILDER_VERSION, VARIANT)


def dump(obj):
    return (json.dumps(obj, indent=1, ensure_ascii=False) + '\n').encode()


def gather():
    if COMMIT is None:
        fail('COMMIT is a placeholder (the coordinator must commit the lock-2 pipeline and set its full hash first)')
    pins = check_pins(); code = builder_code()
    scoring = committed_code(SCORING_CODE); m7 = committed_code(M7_CODE)
    check_driver_protocol(pins)  # committed_code has proved these disk bytes equal COMMIT
    return pins, code, scoring, m7


def check_driver_protocol(pins):
    """A candidate must not consume a run with a stale builder-protocol argument in its committed driver."""
    path = ROOT / 'scripts/evaluation/m7/run_heldout.py'
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(path.read_bytes()).body
              if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in ('BUILDER_PROTOCOL', 'BUILDER_PROTOCOL_SHA')}
    if values != {'BUILDER_PROTOCOL': pins['builderProtocol']['path'], 'BUILDER_PROTOCOL_SHA': pins['builderProtocol']['sha256']}:
        fail('committed run_heldout.py names another builder-protocol path or sha256')


def write_new(path, data, mode=0o444):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(data)


def candidate(_args):
    pins, code, scoring, m7 = gather()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = dump(build_lock(pins, code, git_head(), scoring, m7, utc_now()))
    CANDIDATE.write_bytes(data)
    trial = json.loads(data); trial['approval'] = {'by': 'placeholder', 'utc': 'placeholder'}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'protocol-lock.json'; raw = dump(trial); p.write_bytes(raw)
        builder_accepts(p, sha(raw), code)
    print(json.dumps({'candidate': rel(CANDIDATE), 'sha256': sha(data), 'bytes': len(data),
                      'builderCheckOnPlaceholderApproval': 'accepted'}, indent=1))


def check_approval_text(args):
    """--by, --utc and --how are written into the lock and its record: each must be non-empty text that names neither the
    coordinator-only folder (in any form scripts/evaluation/m7/common.py names_folder finds, so its variable must be set)
    nor holds the banned vehicle-status word; --utc is a UTC time (YYYY-MM-DDTHH:MM:SSZ) no later than now."""
    spec = importlib.util.spec_from_file_location('m7_common', ROOT / 'scripts/evaluation/m7/common.py')
    m7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m7)
    if not m7.coordinator_folders():
        fail(f'set {m7.COORDINATOR_ENV} so the approval text can be checked for the coordinator-only folder')
    for key in ('by', 'utc', 'how'):
        value = getattr(args, key)
        if not value.strip():
            fail(f'--{key} is empty')
        if m7.names_folder(value) or ('pa' + 'rk') in m7.fold(value):
            fail(f'--{key} names the coordinator-only folder or holds the banned vehicle-status word')
    try:
        given = datetime.strptime(args.utc, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    except ValueError:
        fail('--utc must be a UTC time like 2026-10-20T18:30:00Z')
    if given > datetime.now(timezone.utc):
        fail('--utc is later than now')


def approve(args):
    if COMMIT is None:
        fail('COMMIT is a placeholder (the coordinator must commit the lock-2 pipeline and set its full hash first)')
    for path in (LOCK, RECORD, ANCHOR):
        if os.path.lexists(path):
            fail(f'{rel(path)} already exists (never overwritten)')
    check_approval_text(args)
    raw = CANDIDATE.read_bytes()
    if sha(raw) != args.candidate_sha256:
        fail('the candidate on disk is not the one approved')
    pins, code, scoring, m7 = gather()
    lock = json.loads(raw)
    rebuilt = build_lock(pins, code, lock['repositoryHeadAtWriting'], scoring, m7, lock['writtenUtc'])
    if dump(rebuilt) != raw:
        fail('the candidate no longer matches what this writer builds from the pinned files')
    recorded = utc_now()
    lock['status'] = 'LOCKED (approved; held-out runs may use it after its sha256 is recorded in current.md and the anchor is pushed)'
    lock['approval'] = {'by': args.by, 'utc': args.utc, 'how': args.how, 'candidateSha256': args.candidate_sha256,
                        'recordedUtc': recorded}
    data = dump(lock); lock_sha = sha(data)
    ANCHOR.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT_DIR / f'.protocol-lock.{os.getpid()}.tmp'
    write_new(temp, data)
    try:
        builder_accepts(temp, lock_sha, code)  # the builder must accept the exact bytes before they reach the fixed path
        os.link(temp, LOCK)  # fails if LOCK appeared meanwhile
    finally:
        temp.unlink()
    builder_accepts(LOCK, lock_sha, code)
    record = {'file': rel(LOCK), 'sha256': lock_sha, 'bytes': len(data), 'approvedBy': args.by,
              'approvedUtc': args.utc, 'recordedUtc': recorded, 'candidateSha256': args.candidate_sha256,
              'repositoryHead': git_head(),
              'writer': {'file': rel(__file__), 'sha256': sha(Path(__file__).read_bytes())},
              'builderCheck': f'check_protocol_lock (builder v{BUILDER_VERSION}) accepted the written file'}
    write_new(RECORD, dump(record))
    write_new(ANCHOR, dump({'schema': 'pao-lock-anchor-v1', 'protocolLockSha256': lock_sha, 'recordedUtc': recorded}),
              mode=0o644)
    print(json.dumps(record, indent=1))
    print(f'Next: record the sha256 in current.md, then commit and push {ANCHOR_REL} before any held-out access.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('candidate')
    a = sub.add_parser('approve')
    a.add_argument('--candidate-sha256', required=True); a.add_argument('--by', required=True)
    a.add_argument('--utc', required=True); a.add_argument('--how', required=True)
    args = ap.parse_args()
    (candidate if args.cmd == 'candidate' else approve)(args)


if __name__ == '__main__':
    main()
