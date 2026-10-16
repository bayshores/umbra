"""Run the unchanged v4.3 scene builder against lock 2; all builder checks remain active."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
LOCK_PATH = ROOT / '.local/evaluation/pao-accuracy/lock-2/protocol-lock.json'
SUPERSEDED_LOCK_SHA = '7aa2945e61e676ffc3a2a207895ca5639e4d9b4405893b648f4412891c00a475'


def load_builder():
    # Do not import M7's common by name: the builder verifies the origin of sponsor common.
    spec = importlib.util.spec_from_file_location('build_av2_scene', ROOT / 'scripts/california/build_av2_scene.py')
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    tool.LOCK_PATH_V4 = LOCK_PATH
    build = tool.build

    def guarded_build(args):
        # Shared with the shape CLI and in-process callers; runs before the builder opens any log or output.
        tool.car_shape.check_held_out(args.log_id, args.held_out_run, args.protocol_lock_sha256)
        return build(args)

    tool.build = guarded_build
    return tool


def main():
    if SUPERSEDED_LOCK_SHA in sys.argv[1:]:
        raise SystemExit('Lock 1 is superseded; use lock-2/protocol-lock.json')
    load_builder().main()


if __name__ == '__main__':
    main()
