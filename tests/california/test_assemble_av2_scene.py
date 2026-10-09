"""Retained analytic preparation fixtures only; no real payload or producer execution."""
from io import BytesIO
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4
import zipfile

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('assemble_av2_scene', ROOT / 'scripts/california/assemble_av2_scene.py')
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)


class IndexedAV2SceneTests(unittest.TestCase):
    def setUp(self):
        # Keep every generated input/output available for inspection; never overwrite a run.
        self.directory = ROOT / '.local/worker-output/av2-runtime-bridge' / ('fixture-' + str(uuid4()))
        self.prepared = self.directory / 'prepared'; self.prepared.mkdir(parents=True)
        self.output = self.directory / 'scene'; self.manifest_path = self.prepared / 'manifest.json'
        angle = .3; c, s = np.cos(angle), np.sin(angle)
        self.city_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        self.extrinsic_rotation = np.array([[1., 0, 0], [0, c, -s], [0, s, c]])
        self.r1 = np.array([[c, 0, s], [0, 1., 0], [-s, 0, c]])
        city_pose = np.eye(4); city_pose[:3, :3] = self.city_rotation; city_pose[:3, 3] = [1200., -700., 31.]
        extrinsic = np.eye(4); extrinsic[:3, :3] = self.extrinsic_rotation; extrinsic[:3, 3] = [.3, -.2, 1.2]
        self.city_from_camera = city_pose @ extrinsic; self.origin = self.city_from_camera[:3, 3]
        self.virtual_rotation = self.city_rotation @ self.extrinsic_rotation @ self.r1.T
        # First two have a pixel collision; third lands on an inclusive crop center.
        # Outside-positive, behind, zero-Z, unknown, and near-positive-Z rows all survive.
        self.rect = np.array([[0., 0, 10], [.01, .01, 10], [-10.825, -3.4875, 5],
                              [100., 0, 10], [0., 0, -10], [1., 1, 0],
                              [np.nan, 0, 10], [1., 1, 1e-10], [0., .001, .005]])
        self.city = self.rect @ self.virtual_rotation.T + self.origin
        self.city[np.isnan(self.rect).any(1)] = np.nan
        self.p1 = np.array([[100., 0, 236.5, 0], [0, 200., 149.5, 0], [0, 0, 1., 0]])
        self.a = np.array([[1., 0, -20.], [0, 1., -10.], [0, 0, 1.]])
        with np.errstate(invalid='ignore', divide='ignore'):
            projected = self.rect @ self.p1[:, :3].T
            self.uv = projected[:, :2] / projected[:, 2:3] - [20, 10]
        domain = np.isfinite(self.rect).all(1) & (self.rect[:, 2] > 0)
        self.uv[~domain] = np.nan
        self.eligible = domain & (self.uv[:, 0] >= 0) & (self.uv[:, 0] <= 735) & (self.uv[:, 1] >= 0) & (self.uv[:, 1] <= 319)
        count = len(self.rect)
        self.arrays = dict(sourceIndex=np.arange(count, dtype=np.int64), egoXYZ=self.city.copy(), cityXYZ=self.city.copy(),
                           offsetNs=np.arange(count, dtype=np.int64), leftRectifiedXYZ=self.rect.copy(), rightRectifiedXYZ=self.rect.copy(),
                           leftCropUV=self.uv.copy(), rightCropUV=self.uv.copy(), leftEligible=self.eligible.copy(), rightEligible=self.eligible.copy(),
                           bothEligible=self.eligible.copy(), insideFootprint=np.ones(count, bool), nominalGroundHeights=np.full(count, 9000.),
                           groundSupport=np.ones(count, bool), groundBoundaryAlias=np.zeros(count, bool), groundIntegerInBounds=np.ones(count, bool),
                           groundImageXY=np.zeros((count, 2)), groundIndices=np.zeros((count, 2), dtype=np.int64))
        yy, xx = np.indices((320, 736))
        self.rgb = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=2).astype(np.uint8)
        png = BytesIO(); Image.fromarray(self.rgb).save(png, format='PNG')
        self.calibration = dict(sourceClass='Argoverse2Sensor', decoded_source_size=[800, 400], source_rectified_size=[800, 400],
                                crop_size=[736, 320], original=dict(left_projection=self.p1.tolist()),
                                transformed=dict(left_projection=(self.a @ self.p1).tolist()))
        self.files = {'left.png': png.getvalue(), 'right.png': png.getvalue(), 'calibration.json': tool.canonical(self.calibration),
                      'references.npz': self.npz(), 'LICENSE-AND-ATTRIBUTION.md': b'Analytic fixture only; no dataset rights or sponsor execution.\n'}
        source_hashes = {role: tool.digest(('analytic-' + role).encode()) for role in tool.SOURCE_ROLES}
        self.manifest = dict(schemaVersion=1, sourceClass='Argoverse2Sensor', dataset='Argoverse2Sensor', analyticFixture=True,
                             logId='12345678-1234-1234-1234-123456789abc', cameraTimestampNs='100', lidarTimestampNs='110', crossingId='8',
                             plan=dict(path='analytic-plan.json', bytes=10, sha256=tool.digest(b'fixture-plan')),
                             sources={role: dict(path='unread-' + role, bytes=10, sha256=sha) for role, sha in source_hashes.items()},
                             crop=dict(x=20, y=10, width=736, height=320, source_to_crop_matrix=self.a.tolist()),
                             processing=dict(cityFromEgoCamera=city_pose.tolist(), egoFromCameraLeft=extrinsic.tolist(),
                                             leftCameraCityOrigin=self.origin.tolist(), R1=self.r1.tolist(), P1=self.p1.tolist(),
                                             rectifiedSize=[800, 400], sourceRows=count, eligibleLeft=int(self.eligible.sum()), sourceHashes=source_hashes))
        self.refresh()

    def npz(self):
        stream = BytesIO(); np.savez_compressed(stream, **self.arrays); return stream.getvalue()

    def refresh(self):
        self.files['references.npz'] = self.npz()
        self.files['calibration.json'] = tool.canonical(self.calibration)
        for name, data in self.files.items(): (self.prepared / name).write_bytes(data)
        self.manifest['outputs'] = {name: dict(bytes=len(data), sha256=tool.digest(data)) for name, data in self.files.items()}
        self.manifest_path.write_bytes(tool.canonical(self.manifest))

    def assemble(self):
        return tool.assemble(self.manifest_path, self.output, 'analytic_av2')

    def buffer(self, scene, key, dtype, width=1, measured=True):
        owner = scene['measurements'] if measured else scene
        data = (self.output / owner['buffers'][key]).read_bytes()
        self.assertEqual(tool.digest(data), owner['artifactHashes'][key])
        return np.frombuffer(data, dtype=dtype).reshape((-1, width)) if width > 1 else np.frombuffer(data, dtype=dtype)

    def rejected(self):
        before = {name: (self.prepared / name).read_bytes() for name in self.files}
        with self.assertRaises((ValueError, KeyError, TypeError, zipfile.BadZipFile)):
            self.assemble()
        self.assertFalse(self.output.exists())
        for name, data in before.items(): self.assertEqual((self.prepared / name).read_bytes(), data)

    def test_exact_identity_camera_world_uv_and_render_subset(self):
        scene = self.assemble()
        expected_world = (self.city - self.origin)[:, [0, 2, 1]] * [1., 1., -1.]
        np.testing.assert_array_equal(self.buffer(scene, 'referencePositions', '<f8', 3), expected_world)
        np.testing.assert_array_equal(self.buffer(scene, 'referenceCameraPositions', '<f8', 3), self.rect)
        np.testing.assert_array_equal(self.buffer(scene, 'referenceUV', '<f8', 2), self.uv)
        np.testing.assert_array_equal(self.buffer(scene, 'sourceIndices', '<u4'), np.arange(len(self.rect)))
        np.testing.assert_array_equal(self.buffer(scene, 'referenceEligible', 'u1'), self.eligible)
        rendered = self.buffer(scene, 'positions', '<f4', 3, False)
        np.testing.assert_array_equal(rendered, expected_world[self.eligible].astype('<f4'))
        self.assertEqual(scene['pointCount'], int(self.eligible.sum()))
        self.assertEqual(scene['triangleCount'], 0); self.assertEqual((self.output / 'triangles.bin').read_bytes(), b'')
        expected_rotation = tool.WORLD_FROM_CITY @ self.virtual_rotation
        np.testing.assert_allclose(scene['transform']['worldFromCamera'], expected_rotation, atol=1e-15)
        self.assertGreater(np.max(np.abs(expected_rotation - tool.WORLD_FROM_CITY @ self.city_rotation @ self.extrinsic_rotation)), .1)
        np.testing.assert_allclose(scene['camera']['forward'], expected_rotation[:, 2], atol=1e-15)
        np.testing.assert_allclose(scene['camera']['up'], -expected_rotation[:, 1], atol=1e-15)
        self.assertEqual(scene['camera']['eye'], [0, 0, 0]); self.assertEqual(scene['transform']['cameraOrigin'], [0, 0, 0])
        self.assertEqual(scene['transform']['originRole'], 'leftCamera')
        self.assertEqual([scene['camera'][k] for k in ('fx', 'fy', 'cx', 'cy')], [100, 200, 216.5, 139.5])
        self.assertEqual(scene['bounds']['min'], rendered.min(0).tolist()); self.assertEqual(scene['bounds']['max'], rendered.max(0).tolist())
        bound = np.maximum(np.abs(rendered.min(0).astype(float)), np.abs(rendered.max(0).astype(float)))
        self.assertEqual(scene['camera']['far'], 2 * np.linalg.norm(bound))
        self.assertEqual(scene['measurements']['binaryEncoding'], 'littleEndian')
        self.assertEqual((self.output / 'recorded.png').read_bytes(), self.files['left.png'])
        self.assertEqual((self.output / 'LICENSE-AND-ATTRIBUTION.md').read_bytes(), self.files['LICENSE-AND-ATTRIBUTION.md'])
        provenance = scene['measurements']['provenance']
        self.assertEqual(provenance['preparedManifestSha256'], tool.digest(self.manifest_path.read_bytes()))
        self.assertEqual(provenance['referencesSha256'], tool.digest(self.files['references.npz']))
        self.assertEqual((self.output / provenance['references']['path']).resolve(), self.prepared / 'references.npz')
        self.assertEqual(provenance['originalSources'], self.manifest['sources'])
        for key in ('logId', 'cameraTimestampNs', 'lidarTimestampNs', 'crossingId'):
            self.assertEqual(provenance[key], self.manifest[key])

    def test_collisions_outside_unavailable_near_clip_and_projected_color(self):
        scene = self.assemble(); ids = self.buffer(scene, 'sourceIndices', '<u4')
        cells = np.floor(self.uv[self.eligible] + .5).astype(int)
        np.testing.assert_array_equal(cells[0], cells[1])
        self.assertEqual(ids[:2].tolist(), [0, 1])  # Collision preserves both returns.
        self.assertEqual(scene['measurements']['pixelCellPolicy'], 'nearestIntegerCenterRejectCollisions')
        self.assertGreater(self.buffer(scene, 'referenceUV', '<f8', 2)[3, 0], 735)
        self.assertTrue(np.isnan(self.buffer(scene, 'referenceUV', '<f8', 2)[4:7]).all())
        self.assertTrue(np.isnan(self.buffer(scene, 'referencePositions', '<f8', 3)[6]).all())
        self.assertFalse(self.buffer(scene, 'referenceEligible', 'u1')[6])
        self.assertGreater(self.buffer(scene, 'referenceUV', '<f8', 2)[7, 0], 1e11)
        # Rendering near clip never alters admission: this eligible row has Z=.005m.
        self.assertTrue(self.buffer(scene, 'referenceEligible', 'u1')[8]); self.assertLess(self.rect[8, 2], scene['camera']['near'])
        srgb = self.rgb[cells[:, 1], cells[:, 0]].astype(float) / 255
        linear = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4).astype('<f4')
        np.testing.assert_array_equal(self.buffer(scene, 'colors', '<f4', 3, False), linear)
        self.assertEqual(scene['measurements']['groundPolicy'], 'unvalidated')
        self.assertNotIn('ground', scene); self.assertNotIn('depth', scene['buffers'])
        self.assertNotIn('roadToleranceMeters', scene['processing']); self.assertNotIn('depthDivisor', scene['processing'])

    def test_nominal_terrain_does_not_filter_or_create_ground(self):
        baseline = self.assemble(); original_positions = (self.output / 'referencepositions.bin').read_bytes()
        self.arrays['nominalGroundHeights'][:] = np.nan; self.arrays['groundSupport'][:] = False
        self.arrays['insideFootprint'][:] = False; self.refresh(); self.output = self.directory / 'unknown-terrain-scene'
        second = self.assemble()
        self.assertEqual((self.output / 'referencepositions.bin').read_bytes(), original_positions)
        self.assertEqual(second['pointCount'], baseline['pointCount']); self.assertNotIn('ground', second)
        self.assertEqual(second['measurements']['groundPolicy'], 'unvalidated')

    def test_semantic_reference_corruptions_with_refreshed_hashes(self):
        original = {key: value.copy() for key, value in self.arrays.items()}
        changes = [('sourceIndex', 1, 0), ('sourceIndex', 0, 2), ('leftCropUV', (0, 0), self.uv[0, 0] + 1e-5),
                   ('leftCropUV', 3, np.nan), ('leftCropUV', 4, [1., 1.]), ('leftEligible', 0, False),
                   ('leftEligible', 3, True), ('cityXYZ', (0, 0), self.city[0, 0] + 1e-6),
                   ('leftRectifiedXYZ', (0, 0), 1.), ('cityXYZ', (0, 0), np.inf), ('leftCropUV', (0, 0), np.inf)]
        for key, index, value in changes:
            with self.subTest(key=key, index=index):
                self.arrays = {name: array.copy() for name, array in original.items()}
                self.arrays[key][index] = value; self.refresh(); self.rejected()

    def test_pose_crop_calibration_and_identity_corruptions(self):
        original = tool.canonical(self.manifest); original_cal = tool.canonical(self.calibration)
        changes = [('R1', [[1., 0, 0], [0, 1., 0], [0, 0, -1.]]), ('leftCameraCityOrigin', [0, 0, 0]),
                   ('eligibleLeft', 0), ('sourceRows', len(self.rect) - 1)]
        for key, value in changes:
            self.manifest = tool.parse(original); self.manifest['processing'][key] = value; self.refresh(); self.rejected()
        self.manifest = tool.parse(original)
        self.manifest['crop']['x'] = 21; self.refresh(); self.rejected()
        self.manifest = tool.parse(original)
        self.calibration['transformed']['left_projection'][1][2] += .5; self.refresh(); self.rejected()
        self.calibration = tool.parse(original_cal)
        for key, value in [('logId', 'invalid'), ('cameraTimestampNs', 100), ('lidarTimestampNs', '1.1'), ('schemaVersion', True)]:
            self.manifest = tool.parse(original); self.manifest[key] = value; self.refresh(); self.rejected()

    def test_npz_members_headers_payload_and_dtype_boundaries(self):
        baseline = {key: value.copy() for key, value in self.arrays.items()}
        for key, value in [('sourceIndex', np.arange(len(self.rect), dtype=float)), ('leftEligible', self.eligible.astype('u1')),
                           ('cityXYZ', self.city.astype('f4')), ('leftCropUV', np.zeros((len(self.rect), 3))),
                           ('nominalGroundHeights', np.full(len(self.rect), object(), object))]:
            self.arrays = {name: array.copy() for name, array in baseline.items()}; self.arrays[key] = value
            self.refresh(); self.rejected()
        self.arrays = baseline; original = self.npz()
        with zipfile.ZipFile(BytesIO(original)) as source:
            members = [(entry.filename, source.read(entry.filename)) for entry in source.infolist()]
        for entries in [members[:-1], members + [('unknown.npy', members[0][1])], members[:-1] + [members[0]]]:
            stream = BytesIO()
            with zipfile.ZipFile(stream, 'w') as target:
                for name, payload in entries: target.writestr(name, payload)
            with self.assertRaises(ValueError): tool.references(stream.getvalue(), len(self.rect))
        # Honest ZIP length with dishonest NPY shape must fail before np.load allocates.
        header = BytesIO(); np.lib.format.write_array_header_1_0(header, dict(descr='<f8', fortran_order=False, shape=(tool.MAX_ROWS, 3)))
        stream = BytesIO()
        with zipfile.ZipFile(stream, 'w') as target:
            for name, payload in members: target.writestr(name, header.getvalue() if name == 'cityXYZ.npy' else payload)
        with patch.object(tool.np, 'load', side_effect=AssertionError('must reject before allocation')):
            with self.assertRaises(ValueError): tool.references(stream.getvalue(), len(self.rect))
        with patch.object(tool, 'MAX_NPZ_EXPANDED', 1), patch.object(tool.np, 'load', side_effect=AssertionError('must reject before allocation')):
            with self.assertRaises(ValueError): tool.references(original, len(self.rect))

    def test_byte_hash_json_and_exclusive_publication_boundaries(self):
        original = self.manifest_path.read_bytes()
        for raw in [original.replace(b'"schemaVersion":1', b'"schemaVersion":1,"schemaVersion":1'),
                    original.replace(b'"schemaVersion":1', b'"schemaVersion":NaN'), b'[]', b' ' * (tool.MAX_MANIFEST + 1)]:
            self.manifest_path.write_bytes(raw); self.rejected()
        self.manifest_path.write_bytes(original)
        self.manifest['outputs']['left.png']['bytes'] = tool.MAX_FILE + 1
        self.manifest_path.write_bytes(tool.canonical(self.manifest)); self.rejected()
        self.manifest = tool.parse(original)
        self.manifest['outputs']['left.png'].pop('sha256'); self.manifest_path.write_bytes(tool.canonical(self.manifest)); self.rejected()
        self.manifest = tool.parse(original); self.refresh()
        (self.prepared / 'left.png').write_bytes(b'changed'); self.rejected()
        self.refresh(); self.assemble(); before = (self.output / 'scene.json').read_bytes()
        with self.assertRaises(ValueError): self.assemble()
        self.assertEqual((self.output / 'scene.json').read_bytes(), before)

    def test_header_byte_ceiling_rejects_before_numpy_header_parser(self):
        with zipfile.ZipFile(BytesIO(self.files['references.npz'])) as source:
            members = [(entry.filename, source.read(entry.filename)) for entry in source.infolist()]
        for version, length_bytes in [((1, 0), 2), ((2, 0), 4)]:
            for declared in [(4097).to_bytes(length_bytes, 'little') + b' ' * 4097, b'\x01']:
                with self.subTest(version=version, truncated=len(declared) == 1):
                    prefix = np.lib.format.magic(*version)
                    observed = BytesIO(prefix + declared)
                    self.assertEqual(np.lib.format.read_magic(observed), version)
                    self.assertEqual(observed.tell(), 8)  # Runtime observation, guard saves tell() rather than hardcoding it.
                    stream = BytesIO()
                    with zipfile.ZipFile(stream, 'w') as target:
                        # Invalid member first proves the guard precedes any header parser.
                        target.writestr('cityXYZ.npy', prefix + declared)
                        for name, payload in members:
                            if name != 'cityXYZ.npy': target.writestr(name, payload)
                    data = stream.getvalue()
                    self.files['references.npz'] = data; (self.prepared / 'references.npz').write_bytes(data)
                    self.manifest['outputs']['references.npz'] = dict(bytes=len(data), sha256=tool.digest(data))
                    self.manifest_path.write_bytes(tool.canonical(self.manifest))
                    with patch.object(tool.np.lib.format, 'read_array_header_1_0', side_effect=AssertionError('must reject before header parse')), \
                         patch.object(tool.np.lib.format, 'read_array_header_2_0', side_effect=AssertionError('must reject before header parse')):
                        self.rejected()

    def test_capture_once_uses_captured_bytes_and_no_original_source_reads(self):
        reads = {}; original = tool.snapshot
        def once(path, *args, **kwargs):
            path = Path(path).resolve(); reads[path] = reads.get(path, 0) + 1
            captured = original(path, *args, **kwargs)
            if path.name == 'left.png': path.write_bytes(b'changed after capture')
            return captured
        with patch.object(tool, 'snapshot', side_effect=once): scene = self.assemble()
        self.assertEqual(len(reads), 6); self.assertTrue(all(count == 1 for count in reads.values()))
        self.assertEqual((self.output / 'recorded.png').read_bytes(), self.files['left.png'])
        self.assertEqual(scene['measurements']['provenance']['preparedManifestSha256'], tool.digest(self.manifest_path.read_bytes()))


if __name__ == '__main__':
    unittest.main()
