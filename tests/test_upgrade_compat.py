"""Compatibility when QGIS unloads an already-running older plugin release."""

import ctypes
import ctypes.wintypes
from types import SimpleNamespace


def test_legacy_windows_unload_after_replacing_plugin_files(monkeypatch):
    removed = []
    free_calls = []

    def free_library(handle):
        free_calls.append(handle)

    monkeypatch.setattr(
        ctypes, 'windll',
        SimpleNamespace(kernel32=SimpleNamespace(FreeLibrary=free_library)),
        raising=False,
    )

    class PreviouslyLoadedPlugin:
        """The Windows unload path from releases before the DLL was removed."""

        def __init__(self):
            self.provider = object()

        def unload(self):
            # QGIS retains this old method while replacing files on disk.
            from ArrNorm.core.auxil.auxil import lib
            try:
                getattr(ctypes, 'windll').kernel32.FreeLibrary.argtypes = [ctypes.wintypes.HMODULE]
                getattr(ctypes, 'windll').kernel32.FreeLibrary(lib._handle)
                del lib
            except Exception:
                pass
            removed.append(self.provider)

    plugin = PreviouslyLoadedPlugin()
    plugin.unload()

    assert removed == [plugin.provider]
    assert free_calls == []  # No DLL is loaded by the new auxil module.
