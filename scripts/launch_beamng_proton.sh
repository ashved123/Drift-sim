#!/usr/bin/env bash
# Launch the Windows BeamNG.drive build through Proton with the research API
# (tcom) listening on localhost, then attach with:
#   python scripts/test_beamng_connection.py   (no BNG_HOME)
#
# Used instead of the native Linux build, whose Vulkan renderer crashes in
# Mesa's Intel driver (libvulkan_intel.so) on Arrow Lake graphics.
#
# Env overrides: STEAM_ROOT, PROTON, BNG_PORT
set -euo pipefail

STEAM_ROOT="${STEAM_ROOT:-$HOME/.steam/debian-installation}"
APPS="$STEAM_ROOT/steamapps"
PROTON="${PROTON:-$APPS/common/Proton - Experimental/proton}"
GAME="$APPS/common/BeamNG.drive"
PORT="${BNG_PORT:-25252}"

export STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM_ROOT"
export STEAM_COMPAT_DATA_PATH="$APPS/compatdata/284160"
export SteamAppId=284160 SteamGameId=284160

cd "$GAME"
exec "$PROTON" run "$GAME/Bin64/BeamNG.drive.x64.exe" \
    -nosteam -tcom -tport "$PORT" -tcom-listen-ip 127.0.0.1 "$@"
