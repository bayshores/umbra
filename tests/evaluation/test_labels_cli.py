"""CLI-level tests for labelkit.py on a synthetic task in a temporary folder (no AV2 data)."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_labels_tools import KIT, PROTOCOL, RectOracle, blurred_rect, car_task  # noqa: E402

common, imaging, labelkit = KIT.common, KIT.imaging, KIT.labelkit


class LabelkitCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        rgb, self.cov = blurred_rect()
        buffer = io.BytesIO()
        from PIL import Image
        Image.fromarray(rgb).save(buffer, format='PNG')
        image = base / 'image.png'
        image.write_bytes(buffer.getvalue())
        protocol = base / 'protocol-v1.json'
        protocol.write_bytes(common.pretty_json(PROTOCOL))
        common.write_json(base / 'protocol-v1.lock.json', {'sha256': common.sha256_file(protocol)})
        task = car_task()
        png, meta = imaging.render_overview(rgb, task['overview']['region'], task['overview']['cell'])
        (base / 'overview.png').write_bytes(png)
        meta.update(path=str(base / 'overview.png'), sha256=common.sha256_bytes(png))
        task['overview'] = meta
        task['image'] = {'path': str(image), 'sha256': common.sha256_file(image), 'width': 200, 'height': 160}
        self.task = task
        tasks = {'protocol': {'sha256': common.sha256_file(protocol)}, 'tasks': [task]}
        (base / 'tasks.json').write_bytes(common.pretty_json(tasks))
        self.common_args = ['--labeler', 'L1', '--task', 'T01', '--tasks', str(base / 'tasks.json'), '--protocol', str(protocol),
                            '--sessions', str(base / 'sessions')]
        self.base = base

    def run_cli(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = labelkit.main(self.common_args + list(extra))
        return code, out.getvalue()

    def pending(self):
        args = labelkit.argparse.Namespace(labeler='L1', task='T01', tasks=self.common_args[5], protocol=self.common_args[7],
                                           sessions=self.common_args[9])
        with contextlib.redirect_stdout(io.StringIO()):
            return labelkit.command_next(args, echo=False)

    def test_full_session_hash_chain_and_closure(self):
        oracle = RectOracle(self.cov, self.task)
        code, text = self.run_cli('next')
        self.assertEqual(code, 0)
        self.assertIn('QUESTION:', text)
        code, text = self.run_cli('answer', 'maybe')
        self.assertIn('INVALID ANSWER', text)
        steps = 0
        while True:
            state = self.pending()
            if state is None:
                break
            question, image_path = state
            self.assertTrue(Path(image_path).exists())
            code, _ = self.run_cli('answer', oracle.answer(question))
            self.assertEqual(code, 0)
            steps += 1
            self.assertLess(steps, 400)
        folder = self.base / 'sessions/L1/T01'
        label = json.loads((folder / 'label.json').read_text())
        self.assertEqual((folder / 'label.sha256').read_text().split()[0], common.sha256_file(folder / 'label.json'))
        self.assertEqual(label['answers'], steps)
        self.assertLessEqual(abs(label['result']['features']['L']['value'] - 60.3), 0.5)
        code, text = self.run_cli('answer', 'yes')
        self.assertEqual(code, 2)
        self.assertIn('already complete', text)

    def test_tampered_log_is_refused(self):
        self.run_cli('next')
        oracle = RectOracle(self.cov, self.task)
        for _ in range(3):
            question, _ = self.pending()
            self.run_cli('answer', oracle.answer(question))
        log = self.base / 'sessions/L1/T01/answers.jsonl'
        lines = log.read_text().splitlines()
        record = json.loads(lines[1])
        record['answer'] = 'A1' if record['answer'] != 'A1' else 'B1'
        lines[1] = json.dumps(record, sort_keys=True)
        log.write_text('\n'.join(lines) + '\n')
        code, text = self.run_cli('next')
        self.assertEqual(code, 2)
        self.assertIn('chain broken', text)

    def test_answer_requires_a_shown_question(self):
        code, text = self.run_cli('answer', 'A1:B2')
        self.assertEqual(code, 2)
        self.assertIn('Run `next` first', text)


if __name__ == '__main__':
    unittest.main()
