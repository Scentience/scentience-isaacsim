"""Kit lifecycle entry point; simulation remains configured through Python."""
import carb
import omni.ext


class ScentienceOlfactionExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        try:
            from scentience_olfaction import __version__
        except ImportError as exc:
            raise RuntimeError("Install scentience-olfaction in the Isaac interpreter before "
                               "enabling this extension: python -m pip install -e .") from exc
        carb.log_info(f"[scentience.isaac.olfaction] core {__version__} available; "
                      "configure simulations with OlfactionWorld or OlfactorySensorCfg")

    def on_shutdown(self) -> None:
        carb.log_info("[scentience.isaac.olfaction] shutdown")
