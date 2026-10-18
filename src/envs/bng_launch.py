"""Find and connect to BeamNG.drive on Windows or Linux.

Shared by the control panel, the terminal viewer and the training env.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from beamngpy import BeamNGpy

BEAMNG_STEAM_DIR = Path("steamapps") / "common" / "BeamNG.drive"


def find_beamng_home() -> str | None:
    """BeamNG.drive install folder: BNG_HOME, else searched for in every Steam library."""
    if os.environ.get("BNG_HOME"):
        return os.environ["BNG_HOME"]
    steam_roots = []
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
                steam_roots.append(Path(winreg.QueryValueEx(key, "SteamPath")[0]))
        except OSError:
            pass
        steam_roots.append(Path(r"C:\Program Files (x86)\Steam"))
    else:
        steam_roots += [Path.home() / ".steam" / "debian-installation", Path.home() / ".steam" / "steam",
                        Path.home() / ".local" / "share" / "Steam"]
    libraries = []
    for root in steam_roots:
        libraries.append(root)
        vdf = root / "steamapps" / "libraryfolders.vdf"
        if vdf.exists():  # other Steam library folders, e.g. on a second drive
            libraries += [Path(p.replace("\\\\", "\\")) for p in
                          re.findall(r'"path"\s+"([^"]+)"', vdf.read_text(errors="ignore"))]
    for lib in libraries:
        if (lib / BEAMNG_STEAM_DIR / "Bin64").exists():
            return str(lib / BEAMNG_STEAM_DIR)
    return None


def connect(port: int = 25252) -> BeamNGpy:
    """Attach to a running BeamNG, or (Windows) launch one if none is running."""
    bng = BeamNGpy("localhost", port)
    try:
        bng.open(launch=False)
        return bng
    except Exception:
        home = find_beamng_home()
        if not home:
            raise RuntimeError("Couldn't find BeamNG.drive. Start it with the research API "
                               "(scripts/launch_beamng_proton.sh on Linux) or set BNG_HOME to its folder.")
        if sys.platform != "win32":
            raise RuntimeError(f"BeamNG isn't running. Start it first with scripts/launch_beamng_proton.sh "
                               f"(found the game at {home}).")
        bng = BeamNGpy("localhost", port, home=home)
        bng.open(launch=True)
        return bng
