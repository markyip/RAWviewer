"""PyInstaller runtime hook: apply lite profile defaults before app imports."""
try:
    from rawviewer_profile import apply_profile_runtime_defaults

    apply_profile_runtime_defaults()
except Exception:
    pass
