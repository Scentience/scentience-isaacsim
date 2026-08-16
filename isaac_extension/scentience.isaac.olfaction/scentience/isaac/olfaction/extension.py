"""
Kit extension scaffold. Lifecycle only in v0.1: the supported integration is
Isaac Lab (scentience_isaaclab/); this exists so GUI users can enable the
extension and find the docs. Transport equations do NOT belong in this file.
UNVALIDATED in a live Isaac install -- see docs/ISAAC_COMPATIBILITY.md.
"""
import carb
import omni.ext


class ScentienceOlfactionExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        carb.log_info("[scentience.isaac.olfaction] startup (v0.1 scaffold)")

    def on_shutdown(self) -> None:
        carb.log_info("[scentience.isaac.olfaction] shutdown")
