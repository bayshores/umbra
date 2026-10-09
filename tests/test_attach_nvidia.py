"""Actual saved-output attachment checks; no inference or mocked model results."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAME = '2018-10-19-09-30-39_2018-10-19-09-40-40-716'
BASELINE = ROOT / '.local/evaluation/model-app-integration/measured-baseline' / FRAME
PREDICTION = ROOT / '.local/evaluation/nvidia-local-experiment/registered-evaluation/prediction.json'
MODEL = ROOT / '.local/models/c-foundationstereo-s/deployable_foundationstereo_small_320x736_v2.0.onnx'
PREDICTION_HASH = '2e84c905cefab18258bb6e5e7c5dec08b022f364437e4af1cd2a24a7fbb89aea'
spec = importlib.util.spec_from_file_location('attach_nvidia', ROOT / 'scripts/attach-nvidia-reconstruction.py')
attachment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(attachment)


@unittest.skipUnless(BASELINE.is_dir() and PREDICTION.is_file() and MODEL.is_file(),
                     'Real licensed saved model output and measured baseline are required')
class ActualAttachment(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / '.local/evaluation/model-app-integration/preparation-checks'
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.output = self.directory / 'output'

    def run_attachment(self, scene=BASELINE/'scene.json', prediction_hash=PREDICTION_HASH, model=MODEL):
        return attachment.attach(scene, PREDICTION, prediction_hash, model, self.output)

    def copy_scene(self):
        copied = self.directory / 'input'
        shutil.copytree(BASELINE, copied)
        return copied, json.loads((copied/'scene.json').read_text())

    def test_genuine_attachment_preserves_every_measured_field_and_artifact(self):
        result = self.run_attachment()
        original = json.loads((BASELINE/'scene.json').read_text())
        self.assertEqual({k:v for k,v in result.items() if k != 'reconstruction'}, original)
        for name in ('positions.bin','colors.bin','triangles.bin','depth.bin','recorded.jpg','right.jpg','support.png'):
            self.assertEqual((self.output/name).read_bytes(), (BASELINE/name).read_bytes())
        self.assertEqual(result['reconstruction']['evidence'], 'predicted')
        self.assertEqual(result['reconstruction']['role'], 'rendererOnly')
        self.assertEqual(result['reconstruction']['provenance']['registeredPredictionSha256'], PREDICTION_HASH)
        self.assertEqual(result['reconstruction']['model'], attachment.nvidia_stereo.MODEL)

    def test_missing_model_cannot_be_accepted_by_cache_metadata_alone(self):
        with self.assertRaises(FileNotFoundError):
            self.run_attachment(model=self.directory/'missing.onnx')
        self.assertFalse(self.output.exists())

    def test_wrong_registered_pin_stops_without_output(self):
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            self.run_attachment(prediction_hash='0'*64)
        self.assertFalse(self.output.exists())

    def test_existing_destination_is_retained(self):
        self.output.mkdir(); sentinel=self.output/'original';sentinel.write_bytes(b'retain')
        with self.assertRaisesRegex(ValueError, 'Existing destination retained'):
            self.run_attachment()
        self.assertEqual(sentinel.read_bytes(), b'retain')

    def test_altered_scene_id_stops_before_source_attachment(self):
        copied, scene = self.copy_scene();scene['id']='different-observation'
        (copied/'scene.json').write_text(json.dumps(scene))
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.run_attachment(scene=copied/'scene.json')
        self.assertFalse(self.output.exists())

    def test_repinned_depth_cannot_change_measurement_support(self):
        copied, scene = self.copy_scene();path=copied/scene['buffers']['depth']
        raw = bytearray(path.read_bytes());raw[0:2]=(1000).to_bytes(2,'little');path.write_bytes(raw)
        scene['artifactHashes']['depth']=attachment.common.digest(raw)
        (copied/'scene.json').write_text(json.dumps(scene))
        with self.assertRaisesRegex(ValueError, 'support differs'):
            self.run_attachment(scene=copied/'scene.json')
        self.assertFalse(self.output.exists())

    def test_consistent_camera_shift_cannot_disagree_with_measured_geometry(self):
        copied, scene = self.copy_scene()
        origin=np.asarray(scene['transform']['cameraOriginGround'],dtype=float);origin[0]+=.2
        scene['transform']['cameraOriginGround']=origin.tolist()
        scene['camera']['eye']=(-np.asarray(scene['transform']['worldFromCamera'])@origin).tolist()
        (copied/'scene.json').write_text(json.dumps(scene))
        with self.assertRaisesRegex(ValueError, 'Measured geometry differs'):
            self.run_attachment(scene=copied/'scene.json')
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
