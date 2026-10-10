#!/usr/bin/env python3
"""Bounded ordinary Nebius Get; records Job control only, never model proof."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import ssl
import stat
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOST = "api.nebius.cloud"
API_COMMIT = "97201c925d3b03924ee4d1020a4ce31ab35f83e8"
MAX_BYTES = 2 * 1024 * 1024
TIMEOUT_SECONDS = 30
EXPECTED_KEYS = {"schemaVersion", "jobId", "projectId", "image", "containerCommand", "args"}
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
IMAGE = re.compile(r"[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}\Z")
UTC = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?Z\Z")


def require(condition):
    if not condition:
        raise ValueError("Invalid Job evidence")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def parse_json(raw):
    """Consistency parser only: unauthenticated input is never provider proof."""
    require(isinstance(raw, bytes) and len(raw) <= MAX_BYTES)

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result)
            result[key] = value
        return result

    def finite(value):
        number = float(value)
        require(math.isfinite(number))
        return number

    def reject_constant(_):
        raise ValueError("Invalid JSON")

    result = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                        parse_float=finite, parse_constant=reject_constant)
    require(isinstance(result, dict))
    return result


def safe_text(value, limit=8192):
    require(isinstance(value, str) and len(value) <= limit)
    require(all(32 <= ord(char) < 127 for char in value))
    return value


def safe_id(value):
    require(isinstance(value, str) and ID.fullmatch(value) is not None)
    return value


def expected_spec(raw):
    expected = parse_json(raw)
    require(set(expected) == EXPECTED_KEYS and type(expected["schemaVersion"]) is int
            and expected["schemaVersion"] == 1)
    safe_id(expected["jobId"])
    safe_id(expected["projectId"])
    require(isinstance(expected["image"], str) and len(expected["image"]) <= 512
            and IMAGE.fullmatch(expected["image"]) is not None and "//" not in expected["image"])
    safe_text(expected["containerCommand"])
    safe_text(expected["args"])
    return expected


def timestamp(value):
    """Compare protobuf nanoseconds exactly, retaining original strings elsewhere."""
    require(isinstance(value, str))
    match = UTC.fullmatch(value)
    require(match is not None)
    instant = datetime.strptime(match[1], "%Y-%m-%dT%H:%M:%S")
    return (instant, int((match[2] or "").ljust(9, "0")))


def normalized_job(raw, expected, acquired_at):
    """Whitelisted consistency check; this function does not authenticate raw bytes."""
    job = parse_json(raw)
    metadata, spec, status = job["metadata"], job["spec"], job["status"]
    require(all(isinstance(part, dict) for part in (metadata, spec, status)))
    require(metadata["id"] == expected["jobId"] and metadata["parentId"] == expected["projectId"])
    selected_spec = {"image": spec["image"], "containerCommand": spec.get("containerCommand", ""),
                     "args": spec.get("args", "")}
    for key, value in selected_spec.items():
        require(type(value) is str and value == expected[key])
    state = status["state"]
    # ProtoJSON supports enum names and integers. Job COMPLETED=6, not instance=7.
    require(state == "COMPLETED" or (type(state) is int and state == 6))
    created, started, finished = metadata["createdAt"], status["startedAt"], status["finishedAt"]
    require(timestamp(created) <= timestamp(started) <= timestamp(finished) <= timestamp(acquired_at))
    version = metadata.get("resourceVersion")
    if "resourceVersion" in metadata:
        require(type(version) is int or (isinstance(version, str)
                and re.fullmatch(r"[0-9]{1,19}", version) is not None))
        require(0 <= int(version) <= (1 << 63) - 1)
    return {"metadata": {"id": metadata["id"], "parentId": metadata["parentId"],
                         "resourceVersion": version, "createdAt": created},
            "spec": selected_spec,
            "status": {"state": state, "startedAt": started, "finishedAt": finished}}


@contextmanager
def deadline():
    """Configured POSIX wall alarm; native DNS/TLS slow-path behavior unverified."""
    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0))

    def expired(_signum, _frame):
        raise TimeoutError("Collection deadline exceeded")

    previous = signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, TIMEOUT_SECONDS)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def fetch_job(job_id, token):
    """Exactly one GET. HTTPSConnection neither redirects nor retries; no proxies."""
    safe_id(job_id)
    connection = http.client.HTTPSConnection(HOST, timeout=TIMEOUT_SECONDS,
                                             context=ssl.create_default_context())
    try:
        with deadline():
            connection.request("GET", "/ai/v1/jobs/" + job_id,
                               headers={"Authorization": "Bearer " + token,
                                        "Accept": "application/json", "Accept-Encoding": "identity"})
            response = connection.getresponse()
            require(response.status == 200)
            require(response.getheader("Content-Encoding", "identity") == "identity")
            require(response.getheader("Content-Type", "").split(";")[0].strip().lower() == "application/json")
            length = response.getheader("Content-Length")
            if length is not None:
                require(length.isdigit() and int(length) <= MAX_BYTES)
            raw = response.read(MAX_BYTES + 1)
            require(len(raw) <= MAX_BYTES)
            if length is not None:
                require(len(raw) == int(length))
            return raw
    finally:
        connection.close()


def read_regular(path, *, private=False, limit=MAX_BYTES):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_size <= limit)
        if private:
            require(stat.S_IMODE(info.st_mode) == 0o600 and info.st_uid == os.getuid())
        raw = stream.read(limit + 1)
        require(len(raw) <= limit)
        return raw


@contextmanager
def private_output(path):
    """Reserve before credential/network access; walk .local without symlinks."""
    target = Path(os.path.abspath(path))
    relative = target.relative_to(PROJECT_ROOT)
    require(len(relative.parts) >= 2 and relative.parts[0] == ".local")
    directory = os.open(PROJECT_ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = None
    completed = False
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(relative.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode))
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        completed = True
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not completed and "info" in locals():
            # Remove only the failed collector's own reserved file.
            try:
                current = os.stat(relative.name, dir_fd=directory, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino):
                    os.unlink(relative.name, dir_fd=directory)
            except FileNotFoundError:
                pass
        os.close(directory)


def collect(expected_path, token_path, output_path):
    with private_output(output_path) as output:
        expected_raw = read_regular(expected_path)
        expected = expected_spec(expected_raw)
        collector_sha = digest(Path(__file__).read_bytes())
        token = read_regular(token_path, private=True, limit=16384).decode("ascii").strip()
        require(bool(token) and re.fullmatch(r"[A-Za-z0-9._~+/=-]+", token) is not None)
        request = {"method": "GET", "url": "https://" + HOST + "/ai/v1/jobs/" + expected["jobId"],
                   "headers": {"Accept": "application/json", "Accept-Encoding": "identity"}}
        raw = fetch_job(expected["jobId"], token)
        acquired_at = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
        job = normalized_job(raw, expected, acquired_at)
        receipt = {"schemaVersion": 1, "claim": "job_control_only", "modelExecutionVerified": False,
                   "outputsBound": False, "acquiredAt": acquired_at, "job": job,
                   "source": {"kind": "nebius_job_get", "apiSchemaCommit": API_COMMIT,
                              "collectorSha256": collector_sha, "expectedSha256": digest(expected_raw),
                              "request": request, "requestSha256": digest(canonical(request)),
                              "responseSha256": digest(raw), "httpStatus": 200}}
        output.write(canonical(receipt) + b"\n")
    return receipt


class RedactedParser(argparse.ArgumentParser):
    def error(self, _message):
        raise ValueError("Invalid arguments")


def main(argv=None):
    parser = RedactedParser(description=__doc__)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    try:
        args = parser.parse_args(argv)
        collect(args.expected, args.token_file, args.output)
    except (Exception, KeyboardInterrupt):
        print("Job control evidence collection failed; details redacted.", file=sys.stderr)
        return 1
    print("Saved private Job control evidence; model execution and outputs remain unverified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
