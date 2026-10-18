"""Find, launch and connect to BeamNG.drive on Windows or Linux.

BeamNG only accepts Python connections when it's started with its research
API ("tcom") port open: `-nosteam -tcom -tport <port>`. Starting it normally
from Steam leaves the port closed, so:

- `launch_beamng()` starts it with the port open (localhost only):
    Windows: BeamNG.drive.x64.exe directly.
    Linux:   the Windows build through Proton (the native Linux build's Vulkan
             renderer crashes on this laptop's Intel GPU; see CLAUDE.md), the
             same way scripts/launch_beamng_proton.sh does.
- `game_state()` says whether it's running, and whether the port is open.
- `connect()` attaches if the port is open, otherwise launches and waits.

Shared by Drift Studio, the training env and the scripts. Environment
overrides: BNG_HOME (game folder), PROTON (Linux: path to a `proton` script),
STEAM_ROOT (Linux: Steam install folder).
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

from beamngpy import BeamNGpy

BEAMNG_STEAM_DIR = Path("steamapps") / "common" / "BeamNG.drive"
BEAMNG_APP_ID = "284160"
EXE_NAME = "BeamNG.drive.x64.exe"
DEFAULT_PORT = 25252
HOST = "127.0.0.1"


# ---------------------------------------------------------------- finding things

def steam_roots() -> list[Path]:
    """Steam install folders to look in (STEAM_ROOT first, then the usual places)."""
    roots = [Path(os.environ["STEAM_ROOT"])] if os.environ.get("STEAM_ROOT") else []
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
                roots.append(Path(winreg.QueryValueEx(key, "SteamPath")[0]))
        except OSError:
            pass
        roots.append(Path(r"C:\Program Files (x86)\Steam"))
    else:
        roots += [Path.home() / ".steam" / "debian-installation", Path.home() / ".steam" / "steam",
                  Path.home() / ".local" / "share" / "Steam",
                  Path.home() / ".var" / "app" / "com.valvesoftware.Steam" / ".local" / "share" / "Steam"]
    return [r for r in roots if r.exists()]


def steam_libraries() -> list[Path]:
    """Every Steam library folder (games can live on other drives)."""
    libraries = []
    for root in steam_roots():
        libraries.append(root)
        vdf = root / "steamapps" / "libraryfolders.vdf"
        if vdf.exists():
            libraries += [Path(p.replace("\\\\", "\\")) for p in
                          re.findall(r'"path"\s+"([^"]+)"', vdf.read_text(errors="ignore"))]
    seen, unique = set(), []
    for lib in libraries:
        key = str(lib.resolve()) if lib.exists() else str(lib)
        if key not in seen and lib.exists():
            seen.add(key)
            unique.append(lib)
    return unique


def find_beamng_home() -> str | None:
    """BeamNG.drive install folder: BNG_HOME, else searched for in every Steam library."""
    if os.environ.get("BNG_HOME"):
        return os.environ["BNG_HOME"]
    for lib in steam_libraries():
        if (lib / BEAMNG_STEAM_DIR / "Bin64" / EXE_NAME).exists():
            return str(lib / BEAMNG_STEAM_DIR)
    return None


def find_proton() -> Path | None:
    """Linux: Proton to run the Windows build with (PROTON, else Experimental, else newest installed)."""
    if os.environ.get("PROTON"):
        return Path(os.environ["PROTON"])
    found = []
    for lib in steam_libraries():
        common = lib / "steamapps" / "common"
        experimental = common / "Proton - Experimental" / "proton"
        if experimental.exists():
            return experimental
        found += sorted(common.glob("Proton */proton"))
    return found[-1] if found else None


# ---------------------------------------------------------------- state

def port_open(port: int = DEFAULT_PORT, timeout: float = 0.5) -> bool:
    """Is something listening on the research API port?"""
    try:
        with socket.create_connection((HOST, port), timeout=timeout):
            return True
    except OSError:
        return False


def beamng_running() -> bool:
    """Is a BeamNG process running at all (with or without the API port)?"""
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {EXE_NAME}", "/NH"], capture_output=True,
                             text=True, creationflags=0x08000000).stdout
        return EXE_NAME.lower() in out.lower()
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if b"BeamNG.drive.x64" in cmdline.read_bytes():
                return True
        except OSError:
            pass
    return False


def game_state(port: int = DEFAULT_PORT) -> str:
    """'ready' (API port open), 'no_port' (running but started without the API), or 'stopped'."""
    if port_open(port):
        return "ready"
    return "no_port" if beamng_running() else "stopped"


def close_beamng() -> None:
    """Close every BeamNG process (used to restart one that was started without the API port)."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/IM", EXE_NAME], capture_output=True, creationflags=0x08000000)
    else:
        subprocess.run(["pkill", "-f", "BeamNG.drive.x64"], capture_output=True)
    for _ in range(40):  # wait up to ~20 s for it to exit
        if not beamng_running():
            return
        time.sleep(0.5)


# ---------------------------------------------------------------- launch / connect

def launch_beamng(port: int = DEFAULT_PORT) -> subprocess.Popen:
    """Start BeamNG with the research API port open. Returns immediately; use wait_for_port()."""
    home = find_beamng_home()
    if not home:
        raise RuntimeError("Couldn't find BeamNG.drive in any Steam library. Install it, or set BNG_HOME "
                           "to the folder containing Bin64.")
    home = Path(home)
    exe = home / "Bin64" / EXE_NAME
    args = ["-nosteam", "-tcom", "-tport", str(port), "-tcom-listen-ip", HOST]
    if sys.platform == "win32":
        # Detached, so closing Drift Studio doesn't take the game with it.
        flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        return subprocess.Popen([str(exe), *args], cwd=home, creationflags=flags, close_fds=True)

    proton = find_proton()
    if proton is None or not proton.exists():
        raise RuntimeError("Couldn't find Proton. Install 'Proton Experimental' in Steam, or set PROTON to the "
                           "path of a 'proton' script.")
    steamapps = home.parents[1]  # <library>/steamapps/common/BeamNG.drive -> <library>/steamapps
    steam_root = next(iter(steam_roots()), steamapps.parent)
    env = dict(os.environ, STEAM_COMPAT_CLIENT_INSTALL_PATH=str(steam_root),
               STEAM_COMPAT_DATA_PATH=str(steamapps / "compatdata" / BEAMNG_APP_ID),
               SteamAppId=BEAMNG_APP_ID, SteamGameId=BEAMNG_APP_ID)
    return subprocess.Popen([str(proton), "run", str(exe), *args], cwd=home, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def wait_for_port(port: int = DEFAULT_PORT, timeout: float = 240.0, proc: subprocess.Popen | None = None) -> None:
    """Block until the API port answers (the game takes a while to start)."""
    end = time.time() + timeout
    while time.time() < end:
        if port_open(port):
            return
        if proc is not None and proc.poll() is not None and not beamng_running():
            raise RuntimeError(f"BeamNG exited during startup (exit code {proc.returncode}).")
        time.sleep(1.0)
    raise RuntimeError(f"BeamNG didn't open its API port {port} within {timeout:.0f} s.")


def ensure_running(port: int = DEFAULT_PORT, restart_without_port: bool = False) -> str:
    """Make sure BeamNG is up with the API port open. Returns what it had to do."""
    state = game_state(port)
    if state == "ready":
        return "already running"
    if state == "no_port":
        if not restart_without_port:
            raise RuntimeError("BeamNG is open but without its API port (probably started from Steam). "
                               "Close it — or use 'Launch BeamNG' in Drift Studio, which can restart it.")
        close_beamng()
    wait_for_port(port, proc=launch_beamng(port))
    return "launched"


def connect(port: int = DEFAULT_PORT, attach_timeout: float = 120.0) -> BeamNGpy:
    """Attach to BeamNG, launching it first (with the API port open) if it isn't running."""
    ensure_running(port)
    # The port opens early in startup; the API answers once the game has finished loading.
    end, last_error = time.time() + attach_timeout, None
    while time.time() < end:
        bng = BeamNGpy(HOST, port)
        try:
            bng.open(launch=False)
            return bng
        except Exception as e:  # noqa: BLE001 - keep retrying until the game is ready
            last_error = e
            time.sleep(3.0)
    raise RuntimeError(f"Couldn't attach to BeamNG on port {port}: {last_error}")
