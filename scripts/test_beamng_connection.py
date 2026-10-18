"""Smoke test: confirm BeamNGpy can connect to a running BeamNG.drive instance.

Run this after BeamNG.drive is installed and launched (with its console/API
enabled), before trusting anything in src/envs/drift_env.py that talks to
BeamNG. Prints the vehicle's live state for a few seconds then disconnects.

Usage:
    BNG_HOME=/path/to/BeamNG.drive python scripts/test_beamng_connection.py
"""

from __future__ import annotations

import os
import time

from beamngpy import BeamNGpy, Scenario, Vehicle


SPAWN_POS = (-122.0, -188.0, 100.75)


def main():
    # If BNG_HOME is set, launch BeamNG from there; otherwise attach to a
    # running instance.
    home = os.environ.get("BNG_HOME")
    bng = BeamNGpy("localhost", 25252, home=home)
    bng.open(launch=home is not None)

    scenario = Scenario("gridmap_v2", "drift_rl_connection_test")
    vehicle = Vehicle("ego", model="etk800")
    # GridMap v2's ground is at z~100, not 0; this is its "spawn_middle" point.
    scenario.add_vehicle(vehicle, pos=SPAWN_POS, rot_quat=(0, 0, 0, 1))
    scenario.make(bng)

    bng.scenario.load(scenario)
    bng.scenario.start()
    vehicle.connect(bng)

    for _ in range(10):
        vehicle.poll_sensors()
        state = vehicle.state
        print(f"pos={state['pos']} vel={state['vel']}")
        time.sleep(0.5)

    bng.close()

    # Catch the car falling through the map (wrong spawn height).
    drop = SPAWN_POS[2] - state["pos"][2]
    if drop > 2.0:
        raise SystemExit(f"FAIL: car fell {drop:.1f} m below spawn height")
    print("OK: connected, scenario loaded, car resting on the ground")


if __name__ == "__main__":
    main()
