"""Select manufacturer sensors or load a calibrated configuration from JSON."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scentience_olfaction import OlfactionWorld
from scentience_olfaction.config import load_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("scd30", "scd40", "scd41"), default="scd41")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    world = load_world(args.config) if args.config else OlfactionWorld.simple(
        species="carbon_dioxide", strength_ppm=3000, device_profile=args.profile)
    for _ in range(1200):
        world.step(0.1)
        reading = world.read((2.0, 0.0, 1.0), heading=0.0)
    print(f"device={world.device_profile}; t={world.plume.t:.1f} s")
    print("reading:", reading)
    print("plume:", world.truth((2.0, 0.0, 1.0)))


if __name__ == "__main__":
    main()
