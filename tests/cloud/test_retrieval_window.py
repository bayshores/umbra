"""Real local acknowledgement/deadline checks; no model or cloud execution."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'cloud'))
import run_stereo_job as wrapper

ID = '0123456789abcdef0123456789abcdef'


class RetrievalWindow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / ID
        self.ack = self.output.parent / ('.' + ID + '-retrieved')

    def test_real_delayed_acknowledgement_exits_without_waiting_full_window(self):
        timer = threading.Timer(0.05, self.ack.touch)
        timer.start()
        self.addCleanup(timer.join)
        started = time.monotonic()
        with redirect_stdout(StringIO()):
            wrapper.wait_for_retrieval(self.output, ID, 5)
        self.assertLess(time.monotonic() - started, 3)

    def test_real_deadline_without_acknowledgement_fails(self):
        started = time.monotonic()
        with redirect_stdout(StringIO()), self.assertRaises(TimeoutError):
            wrapper.wait_for_retrieval(self.output, ID, 1)
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 1)
        self.assertLess(elapsed, 3)

    def test_acknowledgement_after_descheduling_past_deadline_is_not_delivery(self):
        timer = threading.Timer(0.05, self.ack.touch)
        timer.start()
        self.addCleanup(timer.join)
        real_sleep = time.sleep
        # Model scheduler delay: resume with an ACK, but beyond the actual deadline.
        with patch.object(wrapper.time, 'sleep', side_effect=lambda _: real_sleep(1.1)), \
                redirect_stdout(StringIO()), self.assertRaises(TimeoutError):
            wrapper.wait_for_retrieval(self.output, ID, 1)

    def test_nonempty_or_symlink_acknowledgement_is_not_delivery(self):
        self.ack.write_text('not an acknowledgement')
        with redirect_stdout(StringIO()), self.assertRaises(ValueError):
            wrapper.wait_for_retrieval(self.output, ID, 1)
        self.ack.unlink()
        other = self.output.parent / 'other'
        other.touch()
        self.ack.symlink_to(other)
        with redirect_stdout(StringIO()), self.assertRaises(ValueError):
            wrapper.wait_for_retrieval(self.output, ID, 1)

    def wait_with_ack_created_during_window(self, create, seconds=1):
        # The first poll sleep creates the acknowledgement, after the pre-wait refusal.
        with patch.object(wrapper.time, 'sleep', side_effect=lambda _: create()):
            wrapper.wait_for_retrieval(self.output, ID, seconds)

    def test_fifo_directory_or_foreign_owner_acknowledgement_is_not_delivery(self):
        for name, create, cleanup in (('fifo', lambda: os.mkfifo(self.ack), self.ack.unlink),
                                      ('directory', self.ack.mkdir, self.ack.rmdir)):
            with self.subTest(name), redirect_stdout(StringIO()), self.assertRaises(ValueError):
                self.wait_with_ack_created_during_window(create)
            cleanup()
        with patch.object(wrapper.os, 'getuid', return_value=os.getuid() + 1), \
                redirect_stdout(StringIO()), self.assertRaises(ValueError):
            self.wait_with_ack_created_during_window(self.ack.touch)

    def test_acknowledgement_observed_after_expiry_is_not_delivery(self):
        # Remaining-time check passes, then the post-lstat recheck lands after expiry.
        with patch.object(wrapper.time, 'monotonic', side_effect=[0.0, 0.5, 0.6, 2.0]), \
                redirect_stdout(StringIO()), self.assertRaises(TimeoutError):
            self.wait_with_ack_created_during_window(self.ack.touch)

    def test_success_with_timely_acknowledgement_exits_zero(self):
        argv = ['--submission-id', ID, '--plan-sha256', '0' * 64,
                '--plan', 'not-read.json', '--model', 'not-read.onnx',
                '--output', str(self.output), '--execute']
        timer = threading.Timer(0.2, self.ack.touch)
        timer.start()
        self.addCleanup(timer.join)
        stderr = StringIO()
        with patch.object(wrapper, 'run_job', return_value={}), redirect_stdout(StringIO()), \
                redirect_stderr(stderr):
            self.assertEqual(wrapper.main(argv + ['--retrieval-window-seconds', '5']), 0)
        self.assertEqual(stderr.getvalue(), '')
        with patch.object(wrapper, 'run_job', return_value={}), redirect_stdout(StringIO()):
            self.assertEqual(wrapper.main(argv), 0)

    def test_disabled_window_does_not_wait_or_require_acknowledgement(self):
        with patch.object(wrapper.time, 'sleep', side_effect=AssertionError('Unexpected wait')):
            wrapper.wait_for_retrieval(self.output, ID, 0)

    def test_invalid_window_or_stale_acknowledgement_refuses_before_inference(self):
        argv = ['--submission-id', ID, '--plan-sha256', '0' * 64,
                '--plan', 'not-read.json', '--model', 'not-read.onnx',
                '--output', str(self.output), '--execute']
        for seconds in ('-1', '601'):
            with self.subTest(seconds=seconds), patch.object(wrapper, 'run_job') as run, \
                    redirect_stderr(StringIO()):
                self.assertEqual(wrapper.main(argv + ['--retrieval-window-seconds', seconds]), 1)
                run.assert_not_called()
        self.ack.touch()
        with patch.object(wrapper, 'run_job') as run, redirect_stderr(StringIO()):
            self.assertEqual(wrapper.main(argv + ['--retrieval-window-seconds', '600']), 1)
            run.assert_not_called()

    def test_acknowledgement_created_during_inference_is_refused(self):
        argv = ['--submission-id', ID, '--plan-sha256', '0' * 64,
                '--plan', 'not-read.json', '--model', 'not-read.onnx',
                '--output', str(self.output), '--execute', '--retrieval-window-seconds', '600']
        with patch.object(wrapper, 'run_job', side_effect=lambda *a, **k: self.ack.touch()), \
                redirect_stderr(StringIO()):
            self.assertEqual(wrapper.main(argv), 1)


if __name__ == '__main__':
    unittest.main()
