import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import os from 'node:os';
import { performance } from 'node:perf_hooks';

// Fixed development workload; no duplicated/scaled scenes or acceptance thresholds.
const root = new URL('../', import.meta.url);
const output = new URL('.local/evaluation/performance-baseline.json', root);
const warmupCount = 10, sampleCount = 100, started = performance.now();
const cpuStarted = process.cpuUsage();
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const report = {
  schemaVersion: 1, status: 'failed', generatedAt: new Date().toISOString(),
  command: 'node scripts/benchmark-geometry.mjs',
  scope: 'Single prepared DrivingStereo scene, sequential Node CPU geometry queries, development-only annotations.',
  limitations: [
    'Repeated queries do not test throughput versus physical scene size or additional locations.',
    'No browser rendering, drag responsiveness, GPU, model inference or Nebius execution is measured.',
    'Timing does not establish calibration or visibility accuracy; no acceptance thresholds are set.',
    'Timer overhead is not subtracted; sub-microsecond fast-path timings may be dominated by measurement overhead.',
    'Memory includes the Node process, hashing, scene loading and natural garbage collection; it is not isolated geometry allocation.',
  ],
  environment: {
    node: process.version, versions: process.versions, platform: process.platform,
    architecture: process.arch, osRelease: os.release(), osVersion: os.version(),
    cpuModel: os.cpus()[0]?.model, logicalCpuCount: os.cpus().length,
    availableParallelism: os.availableParallelism(), systemMemoryBytes: os.totalmem(),
  },
  protocol: {
    warmupCountPerWorkload: warmupCount, measuredCountPerWorkload: sampleCount,
    order: 'Sequential, listed workload order; one synchronous call per timing sample.',
    timing: 'performance.now around production function only; deterministic-output check after timing.',
    quantiles: 'Nearest rank on sorted raw milliseconds: ceil(percentile * count) - 1.',
    garbageCollection: 'Natural; no forced GC.', maximumElapsedSeconds: 55,
  },
  sourceFiles: {}, assets: {}, workloads: [], errors: [],
};

async function identities(paths) {
  return Object.fromEntries(await Promise.all(paths.map(async path => {
    const bytes = await readFile(new URL(path, root));
    return [path, {bytes: bytes.length, sha256: hash(bytes)}];
  })));
}

function measure(name, input, run, expected) {
  const result = {
    name, input, inputSha256: hash(JSON.stringify(input)), expectedOutput: expected,
    warmup: {attempts: 0, successes: 0, errors: 0},
    measured: {attempts: 0, successes: 0, errors: 0},
    memoryBefore: process.memoryUsage(), milliseconds: [],
  };
  report.workloads.push(result);
  for (const [phase, count] of [['warmup', warmupCount], ['measured', sampleCount]]) {
    for (let i = 0; i < count; i++) {
      if (performance.now() - started > 55_000) throw new Error('CPU baseline elapsed-time budget exceeded.');
      result[phase].attempts++;
      try {
        const before = performance.now(), actual = run(), elapsed = performance.now() - before;
        assert.deepEqual(actual, expected, `${name}: output changed at ${phase} sample ${i}`);
        assert.ok(Number.isFinite(elapsed) && elapsed >= 0);
        result[phase].successes++;
        if (phase === 'measured') result.milliseconds.push(elapsed);
      } catch (error) {
        result[phase].errors++;
        throw new Error(`${name}, ${phase} sample ${i}: ${error.message}`, {cause: error});
      }
    }
  }
  const sorted = [...result.milliseconds].sort((a, b) => a - b);
  const total = sorted.reduce((a, b) => a + b, 0);
  result.timing = {
    unit: 'milliseconds', count: sorted.length, p50: sorted[Math.ceil(sorted.length * .50) - 1],
    p95: sorted[Math.ceil(sorted.length * .95) - 1], min: sorted[0], max: sorted.at(-1),
    mean: total / sorted.length, total,
  };
  result.memoryAfter = process.memoryUsage();
}

try {
  const catalog = JSON.parse(await readFile(new URL('public/scenes/index.json', root)));
  assert.equal(catalog.length, 1, 'This fixed baseline supports exactly one catalog scene.');
  assert.match(catalog[0].id, /^[a-zA-Z0-9_-]+$/);
  const sceneDirectory = `public/scenes/${catalog[0].id}/`;
  const sourcePaths = ['scripts/benchmark-geometry.mjs', 'src/geometry.mjs', 'src/scene.mjs', 'tests/scene-input.mjs', 'public/scenes/index.json'];
  const assetPaths = (await readdir(new URL(sceneDirectory, root))).sort().map(name => sceneDirectory + name);
  report.sourceFiles = await identities(sourcePaths);
  report.assets = await identities(assetPaths);
  report.memoryBeforeSceneLoad = process.memoryUsage();
  // This fixture calls the real checksum-validating loadScene; annotations never enter the app.
  const {scene, input} = await import('../tests/scene-input.mjs');
  const {createScenario, classifySightline} = await import('../src/geometry.mjs');
  assert.equal(scene.id, catalog[0].id);
  report.memoryAfterSceneLoad = process.memoryUsage();
  report.scene = {
    id: scene.id, sourceClass: scene.sourceClass, pointCount: scene.pointCount,
    triangleCount: scene.triangleCount, depthPixelCount: scene.depth.length,
    supportedDepthPixels: scene.depth.reduce((n, depth) => n + Number(depth > 0), 0),
    cameraWidth: scene.camera.width, cameraHeight: scene.camera.height,
    preparedAssetBytes: Object.values(report.assets).reduce((n, asset) => n + asset.bytes, 0),
    geometryBufferBytes: scene.positions.byteLength + scene.colors.byteLength + scene.triangles.byteLength + scene.depth.byteLength,
    annotationSource: 'tests/scene-input.mjs; development-only what-if actor dimensions and measured-pixel annotations.',
    sourceHashesDeclaredBySceneMetadata: scene.sourceHashes,
    rawSourceHashVerification: 'Original upstream inputs are not read by this benchmark; prepared files are hashed and buffers validated by loadScene.',
  };
  const baseline = createScenario(scene, input);
  const movedInput = {...input, proposedSetback: baseline.limits.maximumSetback, driverDistance: baseline.limits.minimumDriverDistance};
  const moved = createScenario(scene, movedInput);
  measure('createScenario: annotated baseline', input, () => createScenario(scene, input), baseline);
  measure('createScenario: measured-range endpoints', movedInput, () => createScenario(scene, movedInput), moved);
  for (const [name, scenario, center] of [
    ['classifySightline: baseline vehicle', baseline, baseline.baseline.center],
    ['classifySightline: proposed vehicle at measured-range endpoints', moved, moved.proposed.center],
  ]) {
    const args = [scenario.eye, scenario.target, scenario.vehicle, scenario.axis, center];
    measure(name, args, () => classifySightline(scene, ...args), classifySightline(scene, ...args));
  }
  assert.deepEqual(await identities(sourcePaths), report.sourceFiles, 'Source changed during baseline.');
  assert.deepEqual(await identities(assetPaths), report.assets, 'Scene assets changed during baseline.');
  report.status = 'completed';
} catch (error) {
  report.errors.push({name: error.name, message: error.message});
  process.exitCode = 1;
} finally {
  report.elapsedMilliseconds = performance.now() - started;
  report.cpuMicroseconds = process.cpuUsage(cpuStarted);
  report.memoryAfter = process.memoryUsage();
  report.peakResidentSetBytes = process.resourceUsage().maxRSS * 1024;
  await mkdir(new URL('.local/evaluation/', root), {recursive: true});
  await writeFile(output, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({status: report.status, output: '.local/evaluation/performance-baseline.json', elapsedMilliseconds: report.elapsedMilliseconds, errors: report.errors}));
}
