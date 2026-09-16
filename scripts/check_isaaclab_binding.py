"""Check real Sim 6 / Lab 3 class bindings after AppLauncher.

The former Lab 2.3 Kit-stub harness was retired: it cannot verify Lab 3.
Usage: ./isaaclab.sh -p /repo/scripts/check_isaaclab_binding.py --headless
For a no-Isaac source check, use check_isaaclab_contract.py instead.
"""
from validate_install import main

if __name__ == "__main__":
    raise SystemExit(main())
