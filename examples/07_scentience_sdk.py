"""Run offline; add --ovl to exercise the optional SDK's local channel mapping."""
from __future__ import annotations

import argparse
import json

from scentience_olfaction import OlfactionWorld
from scentience_olfaction.bridge import (
    COMPOUND_FIELDS, ovl_sensor_channels, ovl_sensor_window, readings_to_numpy,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ovl', action='store_true', help='use the installed Scentience SDK mapper')
    args = parser.parse_args()
    world = OlfactionWorld.simple(seed=7)
    frames = []
    for _ in range(64):
        world.step(.05)
        frames.append(world.read_ble((1., 0., 1.), uid='SIM001'))
    print(json.dumps(frames[-1], indent=2, allow_nan=False))
    print('Compound columns:', COMPOUND_FIELDS)
    print('NumPy shape:', readings_to_numpy(frames).shape)
    if args.ovl:
        print('SDK OVL columns:', ovl_sensor_channels())
        print('OVL window shape:', ovl_sensor_window(frames).shape)


if __name__ == '__main__':
    main()
