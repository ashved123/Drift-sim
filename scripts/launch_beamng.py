"""Start BeamNG.drive with its research API port open (Windows or Linux), and wait until it's ready.

    python scripts/launch_beamng.py              # launch if needed, report the state
    python scripts/launch_beamng.py --restart    # also restart a BeamNG that was opened without the port
    python scripts/launch_beamng.py --status     # just report, don't launch

Windows runs BeamNG.drive.x64.exe directly; Linux runs the Windows build through
Proton (see src/envs/bng_launch.py). Drift Studio's "Launch BeamNG" button does
the same thing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs import bng_launch as L  # noqa: E402

STATES = {"ready": "running, API port open", "no_port": "running WITHOUT the API port (started from Steam?)",
          "stopped": "not running"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=L.DEFAULT_PORT)
    parser.add_argument("--restart", action="store_true", help="restart BeamNG if it's open without the API port")
    parser.add_argument("--status", action="store_true", help="only report the state")
    args = parser.parse_args()

    print(f"BeamNG install: {L.find_beamng_home() or 'not found'}")
    if sys.platform != "win32":
        print(f"Proton: {L.find_proton() or 'not found'}")
    state = L.game_state(args.port)
    print(f"BeamNG is {STATES[state]} (port {args.port})")
    if args.status or state == "ready":
        return 0
    print("Starting BeamNG… (takes a little while)")
    try:
        L.ensure_running(args.port, restart_without_port=args.restart)
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1
    print(f"Ready: BeamNG is listening on port {args.port}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
