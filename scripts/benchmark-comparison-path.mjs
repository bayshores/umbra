import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import { performance } from 'node:perf_hooks';

const root = new URL('../', import.meta.url);
const out = new URL('.local/evaluation/comparison-path/', root);
const mode = process.argv[2];
const hash = value => createHash('sha256').update(value).digest('hex');
const json = value => JSON.stringify(value);
const save = (name, value) => writeFile(new URL(name, out), JSON.stringify(value, null, 2) + '\n');
const read = async name => JSON.parse(await readFile(new URL(name, out)));
const identities = async paths => Object.fromEntries(await Promise.all(paths.map(async path => {
  const bytes = await readFile(new URL(path, root));
  return [path, {bytes: bytes.length, sha256: hash(bytes)}];
})));
const quantiles = values => {
  const sorted = [...values].sort((a, b) => a - b);
  return {count: sorted.length, min: sorted[0], p50: sorted[Math.ceil(.5 * sorted.length) - 1],
    p95: sorted[Math.ceil(.95 * sorted.length) - 1], max: sorted.at(-1),
    mean: sorted.reduce((a, b) => a + b, 0) / sorted.length};
};
const limits = [
  'One manually annotated China development frame, not three physical locations or a held-out accuracy study.',
  'The grid represents current UI actor ranges; it does not certify road support, physical crossing truth, calibration, free space or safety.',
  'Repeated output identity establishes repeatability only. Expected values come from the same frozen production code, not independent ground truth.',
  'No reconstruction, data acquisition, new inference, caching or runtime optimization is performed.',
  'Node synchronous scenario cost excludes browser paint, rendering, React event handling, dragging, GPU, memory retention, FPS and model inference.',
  'Single-run sequential timings include natural GC and timer overhead; profile samples are collected separately and cannot establish a speedup.',
];
await mkdir(out, {recursive: true});

if (mode === 'prepare') {
  // Freeze both source identities and all expected results before measured calls.
  await assert.rejects(readFile(new URL('protocol.json', out)), {code: 'ENOENT'}, 'Protocol already exists; preserve the prior run.');
  const catalog = JSON.parse(await readFile(new URL('public/scenes/index.json', root)));
  assert.equal(catalog.length, 1);
  const directory = `public/scenes/${catalog[0].id}/`;
  const paths = ['scripts/benchmark-comparison-path.mjs', 'src/geometry.mjs', 'src/scene.mjs',
    'src/App.jsx', 'src/StreetView.jsx', 'tests/scene-input.mjs', 'public/scenes/index.json',
    ...(await readdir(new URL(directory, root))).sort().map(name => directory + name)];
  const frozen = await identities(paths);
  const {scene, input} = await import('../tests/scene-input.mjs');
  const {createScenario} = await import('../src/geometry.mjs');
  const baseline = createScenario(scene, input);
  const setbackRange = [scene.processing.roadToleranceMeters, baseline.limits.maximumSetback];
  const driverRange = [baseline.limits.minimumDriverDistance, baseline.limits.captureDistance];
  const fractions = [0, .25, .5, .75, 1];
  const cases = fractions.flatMap((driverFraction, d) => fractions.map((setbackFraction, s) => ({
    id: `grid-d${d}-s${s}`, driverFraction, setbackFraction,
    input: {...input, driverDistance: driverRange[0] + (driverRange[1] - driverRange[0]) * driverFraction,
      proposedSetback: setbackRange[0] + (setbackRange[1] - setbackRange[0]) * setbackFraction},
  })));
  cases.push({id: 'selected-12.94-9.89853874686384', input: {...input, proposedSetback: 12.94, driverDistance: 9.89853874686384}});
  for (const item of cases) {
    item.inputSha256 = hash(json(item.input));
    item.expectedOutput = createScenario(scene, item.input);
    item.expectedOutputSha256 = hash(json(item.expectedOutput));
  }
  assert.deepEqual(await identities(paths), frozen, 'Source changed during preregistration.');
  await save('protocol.json', {
    schemaVersion: 1, preregisteredAt: new Date().toISOString(), frozen,
    annotationSource: 'tests/scene-input.mjs', scene: {id: scene.id, sourceClass: scene.sourceClass,
      pointCount: scene.pointCount, triangleCount: scene.triangleCount, sourceHashes: scene.sourceHashes},
    uiRangeSource: 'src/App.jsx:117-118; limits from production createScenario, setback minimum scene.processing.roadToleranceMeters',
    grid: {fractions, driverRange, setbackRange, order: 'Driver outer, setback inner; selected case last.'},
    declaredActors: {vehicle: input.vehicle, targetHeight: input.targetHeight, pixels: input.pixels},
    timing: {function: 'Production createScenario(scene, input)', warmupsPerCase: 3, samplesPerCase: 10,
      quantiles: 'Nearest rank, ceil(percentile * count) - 1', clock: 'performance.now around the synchronous function only',
      outputChecks: 'assert.deepEqual and JSON SHA-256 outside timing on every call', gc: 'Natural; no forced GC',
      maximumCombinedCpuPhaseWallMilliseconds: 45000},
    profile: {caseId: cases.at(-1).id, warmups: 3, calls: 300, samplingIntervalMicroseconds: 1000,
      command: 'node --cpu-prof --cpu-prof-interval=1000 --cpu-prof-dir=.local/evaluation/comparison-path --cpu-prof-name=selected.cpuprofile scripts/benchmark-comparison-path.mjs profile'},
    limitations: limits, cases,
  });
  console.log(json({status: 'preregistered', cases: cases.length, driverRange, setbackRange}));
} else if (mode === 'baseline' || mode === 'profile') {
  const started = performance.now(), cpuStarted = process.cpuUsage();
  const report = {status: 'failed', mode, startedAt: new Date().toISOString(), limitations: limits,
    environment: {node: process.version, platform: process.platform, arch: process.arch,
      osRelease: os.release(), cpu: os.cpus()[0]?.model, logicalCpuCount: os.cpus().length}, cases: [], errors: []};
  try {
    const protocol = await read('protocol.json');
    report.protocolSha256 = hash(await readFile(new URL('protocol.json', out)));
    const paths = Object.keys(protocol.frozen);
    const checkFrozen = async () => assert.deepEqual(await identities(paths), protocol.frozen, 'Frozen source or asset changed; stop without rerunning.');
    await checkFrozen();
    const {scene} = await import('../tests/scene-input.mjs');
    const {createScenario} = await import('../src/geometry.mjs');
    const priorElapsed = mode === 'profile' ? (await read('baseline.json')).elapsedMilliseconds : 0;
    const cases = mode === 'profile' ? protocol.cases.filter(item => item.id === protocol.profile.caseId) : protocol.cases;
    for (const item of cases) {
      await checkFrozen();
      assert.equal(hash(json(item.input)), item.inputSha256);
      const result = {id: item.id, inputSha256: item.inputSha256, expectedOutputSha256: item.expectedOutputSha256,
        visibility: {current: item.expectedOutput.currentVisibility, proposed: item.expectedOutput.proposedVisibility},
        warmupOutputIdentities: [], samples: []};
      report.cases.push(result);
      for (const [phase, count] of [['warmup', protocol.timing.warmupsPerCase], ['measured', mode === 'profile' ? protocol.profile.calls : protocol.timing.samplesPerCase]]) {
        for (let i = 0; i < count; i++) {
          assert.ok(priorElapsed + performance.now() - started < protocol.timing.maximumCombinedCpuPhaseWallMilliseconds, 'Combined CPU phase exceeded 45 seconds.');
          const before = performance.now(), actual = createScenario(scene, item.input), elapsedMilliseconds = performance.now() - before;
          assert.deepEqual(actual, item.expectedOutput, `${item.id} ${phase} ${i}: output changed`);
          const outputSha256 = hash(json(actual));
          assert.equal(outputSha256, item.expectedOutputSha256);
          if (phase === 'warmup') result.warmupOutputIdentities.push(outputSha256);
          else result.samples.push({index: i, elapsedMilliseconds, outputSha256});
        }
      }
      result.timingMilliseconds = quantiles(result.samples.map(sample => sample.elapsedMilliseconds));
      await checkFrozen();
    }
    await checkFrozen();
    report.stateReasonCounts = {};
    for (const item of report.cases) for (const [side, visibility] of Object.entries(item.visibility)) {
      const key = `${side}|${visibility.state}|${visibility.reason}`;
      report.stateReasonCounts[key] = (report.stateReasonCounts[key] ?? 0) + 1;
    }
    report.timingMilliseconds = quantiles(report.cases.flatMap(item => item.samples.map(sample => sample.elapsedMilliseconds)));
    report.status = 'completed';
  } catch (error) {
    report.errors.push({name: error.name, message: error.message});
    process.exitCode = 1;
  } finally {
    report.elapsedMilliseconds = performance.now() - started;
    report.cpuMicroseconds = process.cpuUsage(cpuStarted);
    await save(`${mode}.json`, report);
    console.log(json({status: report.status, mode, elapsedMilliseconds: report.elapsedMilliseconds, errors: report.errors}));
  }
} else if (mode === 'summarize') {
  const profile = await read('selected.cpuprofile');
  const nodes = new Map(profile.nodes.map(node => [node.id, node]));
  const parents = new Map(profile.nodes.flatMap(node => (node.children ?? []).map(id => [id, node.id])));
  const rows = new Map();
  const key = node => `${node.callFrame.url}:${node.callFrame.lineNumber + 1}:${node.callFrame.functionName || '(anonymous)'}`;
  for (let i = 0; i < profile.samples.length; i++) {
    const elapsed = profile.timeDeltas[i], leaf = profile.samples[i];
    const seen = new Set();
    for (let id = leaf; id !== undefined; id = parents.get(id)) {
      const node = nodes.get(id), identity = key(node);
      const row = rows.get(identity) ?? {identity, function: node.callFrame.functionName || '(anonymous)',
        url: node.callFrame.url, line: node.callFrame.lineNumber + 1, selfMicroseconds: 0, inclusiveMicroseconds: 0, samples: 0};
      if (id === leaf) {row.selfMicroseconds += elapsed; row.samples++;}
      if (!seen.has(identity)) row.inclusiveMicroseconds += elapsed;
      rows.set(identity, row); seen.add(identity);
    }
  }
  const totalMicroseconds = profile.timeDeltas.reduce((a, b) => a + b, 0);
  const values = [...rows.values()].map(row => ({...row,
    selfPercent: row.selfMicroseconds / totalMicroseconds * 100,
    inclusivePercent: row.inclusiveMicroseconds / totalMicroseconds * 100}));
  await save('profile-summary.json', {profileSha256: hash(await readFile(new URL('selected.cpuprofile', out))),
    durationMicroseconds: profile.endTime - profile.startTime, sampledMicroseconds: totalMicroseconds, sampleCount: profile.samples.length,
    interpretation: 'Sampled Node CPU call tree; inclusive values overlap. Includes module loading and preparation. Separate from unprofiled baseline.',
    topSelf: values.sort((a, b) => b.selfMicroseconds - a.selfMicroseconds).slice(0, 25),
    geometry: values.filter(row => row.url.endsWith('/src/geometry.mjs')).sort((a, b) => b.inclusiveMicroseconds - a.inclusiveMicroseconds)});
  console.log(json({status: 'summarized', sampleCount: profile.samples.length, durationMicroseconds: profile.endTime - profile.startTime}));
} else {
  throw new Error('Use prepare, baseline, profile, or summarize.');
}
