"""PyInstaller runtime hook: apply lite profile + RAM-tier defaults before app imports."""
try:
    from core.profile import apply_runtime_defaults

    apply_runtime_defaults()
except Exception:
    pass
