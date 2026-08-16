"""
Standard Isaac Sim extension startup test (the [[test]] entry in
extension.toml runs this under Kit's test runner). Runs ONLY inside Isaac.
"""
import omni.kit.test


class TestExtensionStartup(omni.kit.test.AsyncTestCase):
    async def test_extension_loaded(self):
        import omni.kit.app
        mgr = omni.kit.app.get_app().get_extension_manager()
        self.assertTrue(mgr.is_extension_enabled("scentience.isaac.olfaction"))

    async def test_core_package_importable(self):
        # the pip-installed physics core must be reachable from Kit's python
        from scentience_olfaction import OlfactionWorld  # noqa: F401
