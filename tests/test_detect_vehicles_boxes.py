"""detect-vehicles.py --boxes: externally supplied boxes through the existing measured-depth placement rules.

Run with numpy + Pillow (no model is loaded in --boxes mode), for example:
  .local/venv-detect/bin/python -I -B -m unittest discover -s tests -p 'test_detect_vehicles_boxes.py' -v
  .local/venv-preparation/bin/python -I -B -m unittest discover -s tests -p 'test_detect_vehicles_boxes.py' -v
Every run works on a temporary copy of the scene; the shipped files are only read.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/detect-vehicles.py'
SCENE_ID = '2018-10-19-09-30-39_2018-10-19-09-40-40-716'
SCENE = ROOT / 'public/scenes' / SCENE_ID
# Placement fields produced by place()/candidacy(); generator hashes, timestamps and timings are not compared.
PLACEMENT = ['id', 'class', 'score', 'box', 'measuredPixels', 'lowerMeasuredPixels', 'lowerMedianWorld', 'bodyPoints',
             'bodyForwardRangeM', 'bodyLateralRangeM', 'curbLaneCandidate']
NVDINO_MODEL = {  # the model block consume_nvdino_result.py carries (labelled fixture: no real NV-DINO boxes here)
    'name': 'NV-Grounding-DINO', 'ngc': 'nvidia/tao/grounding_dino',
    'version': 'grounding_dino_swin_tiny_commercial_trainable_v1.1',
    'package': 'NGC nvidia/tao/grounding_dino grounding_dino_swin_tiny_commercial_trainable_v1.1, run once with '
               'NVIDIA TAO Toolkit 7.2.0',
    'license': 'NVIDIA Open Model License',
    'attribution': 'Licensed by NVIDIA Corporation under the NVIDIA Open Model License',
    'weightsFile': 'grounding_dino_swin_tiny_commercial_trainable.pth', 'weightsBytes': 2070860191,
    'weightsSha256': '8ea7e089e174e72a7fe57ff63cdba5e1e4994b159e41cf72122a7e0d841beaa6'}


def boxes_doc(detections, model, **extra):
    doc = {'schemaVersion': 1, 'kind': 'sightline-external-boxes', 'testOnly': False,
           'image': {'name': 'recorded.jpg', 'sha256': detections['inputs']['recorded.jpg']},
           'model': model, 'threshold': 0.25,
           'boxes': [{'class': r['class'], 'score': r['score'], 'box': r['box']} for r in detections['detections']],
           'source': {'channel': 'test-fixture'}}
    doc.update(extra)
    return doc


def keys_anywhere(value):
    if isinstance(value, dict):
        return set(value) | set().union(*(keys_anywhere(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(keys_anywhere(v) for v in value)) if value else set()
    return set()


@unittest.skipUnless((SCENE / 'detections.json').is_file(), 'prepared DrivingStereo scene is not present locally')
class DetectBoxes(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.scene = self.root / SCENE_ID
        self.scene.mkdir()
        for name in ('scene.json', 'recorded.jpg', 'depth.bin'):
            shutil.copyfile(SCENE / name, self.scene / name)
        self.shipped = json.loads((SCENE / 'detections.json').read_text())

    def run_boxes(self, doc, *extra, output='out.json'):
        path = self.root / 'boxes.json'
        path.write_text(json.dumps(doc))
        args = [sys.executable, '-I', '-B', str(SCRIPT), str(self.scene), '--boxes', str(path), *extra]
        if output:
            args += ['--output', str(self.root / output)]
        return subprocess.run(args, capture_output=True, text=True, timeout=120)

    def test_rf_detr_boxes_reproduce_shipped_placements(self):
        result = self.run_boxes(boxes_doc(self.shipped, self.shipped['model']))
        self.assertEqual(result.returncode, 0, result.stderr)
        out = json.loads((self.root / 'out.json').read_text())
        self.assertEqual([{k: r[k] for k in PLACEMENT} for r in out['detections']],
                         [{k: r[k] for k in PLACEMENT} for r in self.shipped['detections']])
        self.assertEqual(out['rightCurb'], self.shipped['rightCurb'])
        self.assertEqual(out['curbLaneCandidates'], self.shipped['curbLaneCandidates'])
        self.assertEqual({k: out['rules'][k] for k in self.shipped['rules']}, self.shipped['rules'])
        self.assertEqual(out['inputs'], self.shipped['inputs'])
        self.assertEqual(out['model'], self.shipped['model'])
        self.assertEqual(out['boxesSource']['sha256'], hashlib.sha256((self.root / 'boxes.json').read_bytes()).hexdigest())

    def test_nvdino_model_block_and_no_timings(self):
        result = self.run_boxes(boxes_doc(self.shipped, NVDINO_MODEL))
        self.assertEqual(result.returncode, 0, result.stderr)
        out = json.loads((self.root / 'out.json').read_text())
        self.assertEqual(out['model'], NVDINO_MODEL)
        self.assertNotIn('run', out)
        self.assertNotIn('torch', out['generator'])
        self.assertFalse([k for k in keys_anywhere(out) if 'timing' in k.lower() or 'seconds' in k.lower()])
        self.assertIn('not calibrated', out['rules']['externalBoxes'])
        self.assertNotIn('testOnly', out)
        self.assertNotIn('{', result.stderr.split('\n')[0])  # no timing JSON line on stderr

    def test_refusals(self):
        cases = {
            'other image': boxes_doc(self.shipped, NVDINO_MODEL, image={'name': 'recorded.jpg', 'sha256': '0' * 64}),
            'test only': boxes_doc(self.shipped, NVDINO_MODEL, testOnly=True),
            'low score': boxes_doc(self.shipped, NVDINO_MODEL,
                                   boxes=[{'class': 'car', 'score': 0.2, 'box': [1, 2, 3, 4]}]),
            'inverted box': boxes_doc(self.shipped, NVDINO_MODEL,
                                      boxes=[{'class': 'car', 'score': 0.5, 'box': [5, 2, 3, 4]}]),
            'unknown key': boxes_doc(self.shipped, NVDINO_MODEL,
                                     boxes=[{'class': 'car', 'score': 0.5, 'box': [1, 2, 3, 4], 'x': 1}]),
            'no license': boxes_doc(self.shipped, {k: v for k, v in NVDINO_MODEL.items() if k != 'license'}),
            'wrong kind': boxes_doc(self.shipped, NVDINO_MODEL, kind='detections'),
        }
        for name, doc in cases.items():
            result = self.run_boxes(doc)
            self.assertNotEqual(result.returncode, 0, name)
            self.assertFalse((self.root / 'out.json').exists(), name)
        self.assertNotEqual(self.run_boxes(boxes_doc(self.shipped, NVDINO_MODEL), output=None).returncode, 0)

    def test_test_boxes_only_with_flag_and_marked(self):
        result = self.run_boxes(boxes_doc(self.shipped, NVDINO_MODEL, testOnly=True), '--allow-test-boxes')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIs(json.loads((self.root / 'out.json').read_text())['testOnly'], True)

    def test_refuses_overwrite(self):
        (self.root / 'out.json').write_text('{}')
        result = self.run_boxes(boxes_doc(self.shipped, NVDINO_MODEL))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--replace', result.stderr)
        self.assertEqual((self.root / 'out.json').read_text(), '{}')


if __name__ == '__main__':
    unittest.main()
