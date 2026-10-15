"""Smoke test: confirm BeamNGpy can connect to a running BeamNG.drive instance.

Run this after BeamNG.drive is installed and launched (with its console/API
enabled), before trusting anything in src/envs/drift_env.py that talks to
BeamNG. Prints the vehicle's live state for a few seconds then disconnects.

Usage:
    python scripts/test_beamng_connection.py
"""

from __future__ import annotations

import time

from beamngpy import BeamNGpy, Scenario, Vehicle


def main():
    bng = BeamNGpy("localhost", 25252)
    bng.open()

    scenario = Scenario("gridmap_v2", "drift_rl_connection_test")
    vehicle = Vehicle("ego", model="etk800")
    scenario.add_vehicle(vehicle, pos=(0, 0, 0.5), rot_quat=(0, 0, 0, 1))
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


if __name__ == "__main__":
    main()
