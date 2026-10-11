#!/usr/bin/env python3
"""Explicit one-job NVIDIA CPU wrapper; output binding is not Nebius proof."""
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

from job_evidence import RedactedParser, canonical, digest, require
from result_binding import (bundle_payloads, make_binding, nvidia, open_directory,
                            read_path, sha, submission)


def write_new(directory, name, raw):
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory)
    info = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        current = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino):
            os.unlink(name, dir_fd=directory)
        raise
    return info.st_dev, info.st_ino


def run_job(submission_id, plan_sha256, plan_path, model_path, output, *, execute=False):
    require(execute is True)
    submission(submission_id)
    sha(plan_sha256)
    output = Path(os.path.abspath(output))
    parent = open_directory(output.parent)
    root = None
    try:
        # Atomic reservation refuses files, directories and dangling symlinks.
        os.mkdir(output.name, 0o700, dir_fd=parent)
        root = os.open(output.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        root_info = os.fstat(root)

        def unchanged_root():
            current = os.stat(output.name, dir_fd=parent, follow_symlinks=False)
            require((current.st_dev, current.st_ino) == (root_info.st_dev, root_info.st_ino))

        plan_raw = read_path(plan_path)
        require(digest(plan_raw) == plan_sha256)
        validated_raw, plan, _, _ = nvidia.validate_plan(plan_path)
        require(validated_raw == plan_raw)
        require(plan['limits']['jobs'] == 1 and plan['limits']['retries'] == 0
                and plan['limits']['timeoutSeconds'] == 120)
        wrapper_raw = read_path(__file__)
        write_new(root, 'plan.json', plan_raw)
        unchanged_root()
        try:
            # No retry or cache fallback: fresh path under the reserved root.
            require(not os.path.lexists(output / 'prediction'))
            result = nvidia.run(plan_path, model_path, output / 'prediction')
            require(result is None)  # Existing cache branch from a race is refused.
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as failure:
            # Preserve captured child bytes; never echo them or claim success.
            for name, raw in (('adapter-failure.stdout', failure.output),
                              ('adapter-failure.stderr', failure.stderr)):
                if isinstance(raw, bytes):
                    write_new(root, name, raw)
            raise
        unchanged_root()
        nvidia.cached(plan_path, output / 'prediction', model_path)
        require(read_path(plan_path) == plan_raw and read_path(__file__) == wrapper_raw)
        saved_plan, payloads = bundle_payloads(output)
        require(saved_plan == plan_raw)
        binding = make_binding(plan_raw, payloads, submission_id, digest(wrapper_raw))
        require(read_path(plan_path) == plan_raw)
        unchanged_root()
        published = write_new(root, 'job-result.json', canonical(binding) + b'\n')
        try:
            require(read_path(plan_path) == plan_raw and read_path(__file__) == wrapper_raw)
            unchanged_root()
        except BaseException:
            # Roll back only our own success marker, retaining all raw outputs.
            current = os.stat('job-result.json', dir_fd=root, follow_symlinks=False)
            if (current.st_dev, current.st_ino) == published:
                os.unlink('job-result.json', dir_fd=root)
            raise
        return binding
    finally:
        if root is not None:
            os.close(root)
        os.close(parent)


def wait_for_retrieval(output, submission_id, seconds):
    """Keep a successful ephemeral result alive until retrieval is acknowledged."""
    require(type(seconds) is int and 0 <= seconds <= 600)
    if seconds == 0:
        return
    submission(submission_id)
    acknowledgement = Path(output).parent / ('.' + submission_id + '-retrieved')
    require(not os.path.lexists(acknowledgement))
    expires = time.monotonic() + seconds
    print('Result ready; awaiting retrieval acknowledgement.', flush=True)
    while True:
        remaining = expires - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Result retrieval window expired')
        try:
            info = acknowledgement.lstat()
        except FileNotFoundError:
            info = None
        if info is not None:
            require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_size == 0)
            if time.monotonic() >= expires:
                raise TimeoutError('Result retrieval window expired')
            return
        time.sleep(min(1.0, remaining))


def main(argv=None):
    parser = RedactedParser(description=__doc__)
    parser.add_argument('--submission-id', required=True)
    parser.add_argument('--plan-sha256', required=True)
    for name in ('plan', 'model', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--retrieval-window-seconds', type=int, default=0,
                        help='Optional acknowledgement window, 0–600 seconds; no inference retries.')
    try:
        args = parser.parse_args(argv)
        require(0 <= args.retrieval_window_seconds <= 600)
        if args.retrieval_window_seconds:
            submission(args.submission_id)
            acknowledgement = args.output.parent / ('.' + args.submission_id + '-retrieved')
            require(not os.path.lexists(acknowledgement))
        run_job(args.submission_id, args.plan_sha256, args.plan, args.model, args.output,
                execute=args.execute)
        wait_for_retrieval(args.output, args.submission_id, args.retrieval_window_seconds)
    except (Exception, KeyboardInterrupt):
        print('Stereo job failed or refused; details redacted; no successful delivery.', file=sys.stderr)
        return 1
    print('Saved application output binding; service execution remains unauthenticated.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
