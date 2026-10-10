"""Analytic HTTP/JSON fixtures only. Never sponsor proof or live provider calls."""
import copy
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import signal
import ssl
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "cloud"))
import job_evidence as evidence


class JobEvidence(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / ".local").mkdir()
        self.root_patch = patch.object(evidence, "PROJECT_ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        # Every test forbids real HTTP, even if a later test accidentally calls it.
        forbidden = patch.object(evidence.http.client, "HTTPSConnection",
                                 side_effect=AssertionError("Live network forbidden; analytic fixtures only"))
        forbidden.start()
        self.addCleanup(forbidden.stop)
        self.expected = {"schemaVersion": 1, "jobId": "analytic-job-only", "projectId": "analytic-project-only",
                         "image": "registry.example/analytic-fixture@sha256:" + "a" * 64,
                         "containerCommand": "python", "args": "analytic fixture only"}
        self.job = {"metadata": {"id": self.expected["jobId"], "parentId": self.expected["projectId"],
                                 "resourceVersion": "1", "createdAt": "2026-10-09T00:00:00Z"},
                    "spec": {key: self.expected[key] for key in ("image", "containerCommand", "args")},
                    "status": {"state": "COMPLETED", "startedAt": "2026-10-09T00:00:01.000000001Z",
                               "finishedAt": "2026-10-09T00:00:01.000000002Z"}}
        self.acquired = "2026-10-09T00:00:02Z"
        self.expected_path = self.root / "analytic-expected.json"
        self.expected_path.write_bytes(evidence.canonical(self.expected))
        self.token_path = self.root / "analytic-token.txt"
        self.token_path.write_text("analytic-fixture-credential-not-real\n")
        self.token_path.chmod(0o600)
        self.output_path = self.root / ".local" / "analytic-receipt.json"

    def normalized(self, job=None, acquired=None):
        return evidence.normalized_job(evidence.canonical(job or self.job), self.expected,
                                       acquired or self.acquired)

    def response(self, raw=None, status=200, headers=None):
        response = Mock()
        response.status = status
        values = {"Content-Type": "application/json"}
        values.update(headers or {})
        response.getheader.side_effect = lambda key, default=None: values.get(key, default)
        response.read.side_effect = BytesIO(raw if raw is not None else evidence.canonical(self.job)).read
        return response

    def connection(self, response):
        connection = Mock()
        connection.getresponse.return_value = response
        return connection

    def test_expected_schema_and_path_identity_rejections(self):
        for key, values in {"jobId": ["../job", "job/child", "job?view=SECRET", "https://other", "job\n", ""],
                            "projectId": ["../project", "project#fragment"],
                            "image": ["registry.example/mutable:latest", "https://registry.example/a@sha256:" + "a" * 64],
                            "containerCommand": [["python"], "python\n"],
                            "args": [["--analytic"], 0], "schemaVersion": [True, 2]}.items():
            for value in values:
                candidate = dict(self.expected, **{key: value})
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    evidence.expected_spec(evidence.canonical(candidate))
        for change in ({"extra": "not accepted"},):
            with self.assertRaises(ValueError):
                evidence.expected_spec(evidence.canonical(dict(self.expected, **change)))
        missing = dict(self.expected)
        del missing["args"]
        with self.assertRaises(ValueError):
            evidence.expected_spec(evidence.canonical(missing))

    def test_strict_json_at_every_depth_and_size_limit(self):
        for raw in (b'{"a":1,"a":2}', b'{"nested":{"a":1,"a":2}}', b'{"a":NaN}',
                    b'{"a":Infinity}', b'{"a":-Infinity}', b'{"a":1e999}', b'[]', b'\xff'):
            with self.subTest(raw=raw), self.assertRaises((ValueError, UnicodeError)):
                evidence.parse_json(raw)
        with self.assertRaises(ValueError):
            evidence.parse_json(b" " * (evidence.MAX_BYTES + 1))

    def test_exact_identity_spec_and_completed_job_state(self):
        self.assertEqual(self.normalized(), self.job)
        numeric = copy.deepcopy(self.job)
        numeric["status"]["state"] = 6
        self.assertEqual(self.normalized(numeric)["status"]["state"], 6)
        for section, key, values in (("metadata", "id", ["other"]), ("metadata", "parentId", ["other"]),
                                     ("spec", "image", [self.expected["image"].replace("a" * 64, "b" * 64)]),
                                     ("spec", "containerCommand", ["different"]), ("spec", "args", ["different"]),
                                     ("status", "state", ["RUNNING", "FAILED", "completed", "6", 6.0, 7, True, None])):
            for value in values:
                candidate = copy.deepcopy(self.job)
                candidate[section][key] = value
                with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                    self.normalized(candidate)

    def test_utc_order_and_nanosecond_lexemes(self):
        self.assertEqual(self.normalized()["status"]["startedAt"], "2026-10-09T00:00:01.000000001Z")
        for section, key, value in (("metadata", "createdAt", "2026-10-09T00:00:01.000000002Z"),
                                    ("status", "startedAt", "2026-10-09T00:00:01.000000003Z"),
                                    ("status", "finishedAt", "2026-10-09T00:00:03Z"),
                                    ("status", "finishedAt", "2026-10-09T00:00:01+00:00"),
                                    ("status", "finishedAt", "2026-10-09T00:00:01.0000000001Z"),
                                    ("status", "finishedAt", "2026-02-30T00:00:01Z"),
                                    ("status", "finishedAt", None)):
            candidate = copy.deepcopy(self.job)
            candidate[section][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.normalized(candidate)
        with self.assertRaises(ValueError):
            self.normalized(acquired="2026-10-09T00:00:01.000000001Z")

    def test_resource_version_int64_bounds_and_missing_unknown(self):
        for version in ("0", "9223372036854775807", 0, (1 << 63) - 1):
            candidate = copy.deepcopy(self.job)
            candidate["metadata"]["resourceVersion"] = version
            self.assertEqual(self.normalized(candidate)["metadata"]["resourceVersion"], version)
        for version in (-1, 1 << 63, True, 1.0, "-1", "1.0", "1e2", "", "9223372036854775808", "private-content", None):
            candidate = copy.deepcopy(self.job)
            candidate["metadata"]["resourceVersion"] = version
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.normalized(candidate)
        candidate = copy.deepcopy(self.job)
        del candidate["metadata"]["resourceVersion"]
        self.assertIsNone(self.normalized(candidate)["metadata"]["resourceVersion"])

    def test_proto_default_empty_command_args_are_exactly_matched(self):
        for key in ("containerCommand", "args"):
            candidate = copy.deepcopy(self.job)
            del candidate["spec"][key]
            with self.assertRaises(ValueError):
                self.normalized(candidate)
            expected = dict(self.expected, **{key: ""})
            normalized = evidence.normalized_job(evidence.canonical(candidate), expected, self.acquired)
            self.assertEqual(normalized["spec"][key], "")
            candidate["spec"][key] = None
            with self.assertRaises(ValueError):
                evidence.normalized_job(evidence.canonical(candidate), expected, self.acquired)
        candidate = copy.deepcopy(self.job)
        del candidate["spec"]["image"]
        with self.assertRaises(KeyError):
            self.normalized(candidate)

    def test_whitelist_discards_all_unrelated_and_sensitive_fields(self):
        candidate = copy.deepcopy(self.job)
        candidate["metadata"].update(labels={"private": "discard-me"}, name="discard-me")
        candidate["spec"].update(environmentVariables=[{"value": "discard-me"}],
                                  registryCredentials={"password": "discard-me"},
                                  injectedFiles=[{"content": "discard-me"}],
                                  volumes=[{"s3Config": {"credentials": {"accessKeyId": "discard-me"}}}])
        candidate["status"].update(instances=[{"publicIp": "203.0.113.1"}],
                                    stateDetails={"message": "discard-me"}, publicEndpoints=["discard-me"])
        candidate["arbitrary"] = "discard-me"
        self.assertEqual(self.normalized(candidate), self.job)

    def test_single_fixed_tls_get_and_no_retry(self):
        connection = self.connection(self.response())
        with patch.object(evidence.http.client, "HTTPSConnection", return_value=connection) as factory:
            raw = evidence.fetch_job(self.expected["jobId"], "analytic-fixture-only")
        self.assertEqual(raw, evidence.canonical(self.job))
        factory.assert_called_once()
        self.assertEqual(factory.call_args.args, ("api.nebius.cloud",))
        context = factory.call_args.kwargs["context"]
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertEqual(factory.call_args.kwargs["timeout"], 30)
        connection.request.assert_called_once_with("GET", "/ai/v1/jobs/analytic-job-only",
                                                  headers={"Authorization": "Bearer analytic-fixture-only",
                                                           "Accept": "application/json", "Accept-Encoding": "identity"})
        connection.close.assert_called_once()
        connection.request.side_effect = TimeoutError("analytic sensitive detail")
        connection.request.reset_mock()
        with patch.object(evidence.http.client, "HTTPSConnection", return_value=connection), self.assertRaises(TimeoutError):
            evidence.fetch_job(self.expected["jobId"], "analytic-fixture-only")
        connection.request.assert_called_once()

    def test_redirects_and_error_bodies_are_never_followed_or_read(self):
        for status in (301, 302, 307, 308, 401, 403, 500):
            response = self.response(b"analytic-sensitive-body", status, {"Location": "https://other.example/SECRET"})
            connection = self.connection(response)
            with self.subTest(status=status), patch.object(evidence.http.client, "HTTPSConnection", return_value=connection), self.assertRaises(ValueError):
                evidence.fetch_job(self.expected["jobId"], "analytic-fixture-only")
            connection.request.assert_called_once()
            response.read.assert_not_called()

    def test_body_bounds_content_length_and_encoding(self):
        cases = [(b" " * (evidence.MAX_BYTES + 1), {}),
                 (b"{}", {"Content-Length": str(evidence.MAX_BYTES + 1)}),
                 (b"{}", {"Content-Length": "3"}), (b"{}", {"Content-Length": "-1"}),
                 (b"{}", {"Content-Encoding": "gzip"}), (b"{}", {"Content-Type": "text/html"})]
        for raw, headers in cases:
            response = self.response(raw, headers=headers)
            with self.subTest(headers=headers), patch.object(evidence.http.client, "HTTPSConnection", return_value=self.connection(response)), self.assertRaises(ValueError):
                evidence.fetch_job(self.expected["jobId"], "analytic-fixture-only")
        response = self.response()
        with patch.object(evidence.http.client, "HTTPSConnection", return_value=self.connection(response)):
            evidence.fetch_job(self.expected["jobId"], "analytic-fixture-only")
        response.read.assert_called_once_with(evidence.MAX_BYTES + 1)

    def test_total_deadline_restores_handler_on_success_and_failure(self):
        previous = signal.getsignal(signal.SIGALRM)
        for fail in (False, True):
            try:
                with evidence.deadline():
                    self.assertGreater(signal.getitimer(signal.ITIMER_REAL)[0], 0)
                    if fail:
                        signal.getsignal(signal.SIGALRM)(signal.SIGALRM, None)
            except TimeoutError:
                self.assertTrue(fail)
            self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
            self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        with patch.object(signal, "getitimer", return_value=(1.0, 0.0)), patch.object(signal, "setitimer") as setter:
            with self.assertRaises(ValueError), evidence.deadline():
                pass
            setter.assert_not_called()

    def test_non_main_thread_fails_before_request(self):
        connection = self.connection(self.response())
        errors = []
        def run():
            try:
                evidence.fetch_job(self.expected["jobId"], "analytic-fixture-only")
            except ValueError:
                errors.append("clean failure")
        with patch.object(evidence.http.client, "HTTPSConnection", return_value=connection):
            thread = threading.Thread(target=run)
            thread.start()
            thread.join()
        self.assertEqual(errors, ["clean failure"])
        connection.request.assert_not_called()

    def test_actual_offline_alarm_interrupts_sleep_and_restores_state(self):
        previous = signal.getsignal(signal.SIGALRM)
        started = time.monotonic()
        with patch.object(evidence, "TIMEOUT_SECONDS", 0.05), self.assertRaises(TimeoutError):
            with evidence.deadline():
                time.sleep(2)
        self.alarm_elapsed_seconds = time.monotonic() - started
        self.assertLess(self.alarm_elapsed_seconds, 1.0)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))

    def test_private_receipt_hashes_and_claim_limits(self):
        raw = evidence.canonical(self.job)
        with patch.object(evidence, "fetch_job", return_value=raw) as fetch:
            receipt = evidence.collect(self.expected_path, self.token_path, self.output_path)
        fetch.assert_called_once_with("analytic-job-only", "analytic-fixture-credential-not-real")
        self.assertEqual(os.stat(self.output_path).st_mode & 0o777, 0o600)
        self.assertEqual(json.loads(self.output_path.read_bytes()), receipt)
        self.assertEqual(receipt["claim"], "job_control_only")
        self.assertIs(receipt["modelExecutionVerified"], False)
        self.assertIs(receipt["outputsBound"], False)
        self.assertNotIn("authenticated", receipt)
        self.assertNotIn("attemptId", receipt)
        source = receipt["source"]
        self.assertEqual(source["responseSha256"], evidence.digest(raw))
        self.assertEqual(source["expectedSha256"], evidence.digest(self.expected_path.read_bytes()))
        self.assertEqual(source["requestSha256"], evidence.digest(evidence.canonical(source["request"])))
        self.assertEqual(source["collectorSha256"], evidence.digest(Path(evidence.__file__).read_bytes()))
        self.assertNotIn("analytic-fixture-credential", self.output_path.read_text())

    def test_output_collision_before_token_or_network(self):
        for symlink in (False, True):
            if symlink:
                self.output_path.symlink_to(self.root / "absent-target")
            else:
                self.output_path.write_text("preserved")
            with patch.object(evidence, "read_regular") as read, patch.object(evidence, "fetch_job") as fetch:
                with self.assertRaises(FileExistsError):
                    evidence.collect(self.expected_path, self.token_path, self.output_path)
                read.assert_not_called()
                fetch.assert_not_called()
            if not symlink:
                self.assertEqual(self.output_path.read_text(), "preserved")
            self.output_path.unlink()

    def test_output_rejects_outside_local_and_symlink_directories(self):
        (self.root / ".local" / "linked").symlink_to(self.root, target_is_directory=True)
        for target in (self.root / "outside.json", self.root / ".local" / "linked" / "receipt.json"):
            with self.subTest(target=target), patch.object(evidence, "read_regular") as read, self.assertRaises((ValueError, OSError)):
                evidence.collect(self.expected_path, self.token_path, target)
            read.assert_not_called()

    def test_token_requires_regular_private_non_symlink_file(self):
        for mode in (0o644, 0o400, 0o700):
            self.token_path.chmod(mode)
            with self.subTest(mode=mode), patch.object(evidence, "fetch_job") as fetch, self.assertRaises(ValueError):
                evidence.collect(self.expected_path, self.token_path, self.output_path)
            fetch.assert_not_called()
            self.assertFalse(self.output_path.exists())
        self.token_path.unlink()
        self.token_path.symlink_to(self.expected_path)
        with self.assertRaises(OSError):
            evidence.read_regular(self.token_path, private=True)
        self.token_path.unlink()
        os.mkfifo(self.token_path, 0o600)
        with self.assertRaises(ValueError):
            evidence.read_regular(self.token_path, private=True)

    def test_redacted_cli_failure_and_failed_output_cleanup(self):
        stderr, stdout = StringIO(), StringIO()
        with patch.object(evidence, "fetch_job", side_effect=RuntimeError("analytic-secret-body-token")), redirect_stderr(stderr), redirect_stdout(stdout):
            result = evidence.main(["--expected", str(self.expected_path), "--token-file", str(self.token_path),
                                    "--output", str(self.output_path)])
        self.assertEqual(result, 1)
        self.assertEqual(stderr.getvalue(), "Job control evidence collection failed; details redacted.\n")
        self.assertEqual(stdout.getvalue(), "")
        self.assertFalse(self.output_path.exists())
        stderr = StringIO()
        with redirect_stderr(stderr):
            self.assertEqual(evidence.main(["--unexpected", "analytic-secret"]), 1)
        self.assertNotIn("analytic-secret", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
