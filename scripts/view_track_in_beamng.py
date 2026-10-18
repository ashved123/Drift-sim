"""Load generated tracks into BeamNG from the terminal (no GUI).

Type a seed number to load that track, press Enter for the next seed, or q to
quit. The next few seeds are prepared in the background. For sliders, a
preview and car/camera buttons, use scripts/track_control_panel.py instead.

    python scripts/view_track_in_beamng.py --seed 0            # island with hills
    python scripts/view_track_in_beamng.py --seed 0 --flat     # flat smallgrid

Connects to a running BeamNG, or launches it from BNG_HOME / the default
Windows Steam location (see track_control_panel.connect).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.envs.track_loader import TrackLoader, TrackParams  # noqa: E402
from track_control_panel import connect  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flat", action="store_true", help="flat smallgrid instead of the hilly island")
    parser.add_argument("--port", type=int, default=25252)
    args = parser.parse_args()

    bng = connect(args.port)
    loader = TrackLoader(bng, TrackParams(mode="flat" if args.flat else "island"))
    seed = args.seed
    try:
        while True:
            s = loader.load(seed).stats
            print(f"seed {seed}: {s['length']:.0f} m, {s['corners']} corners, tightest {s['tightest']:.0f} m"
                  + (f", {s['climb']:.1f} m climb" if "climb" in s else ""))
            cmd = input("seed number / Enter = next / q = quit > ").strip().lower()
            if cmd == "q":
                break
            seed = int(cmd) if cmd.lstrip("-").isdigit() else seed + 1
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        loader.close()
        bng.disconnect()  # leave the game running


if __name__ == "__main__":
    main()
