"""LibRaw process-pool worker entry (plain Python, not the GUI .app).

macOS ``ProcessPoolExecutor`` with the default spawn target re-executes the
frozen RAWviewer.app (splash / Qt / heavy imports). Point the pool at this
module via ``RAWVIEWER_PROCESS_POOL_PYTHON`` (a real ``python3`` / pixi
interpreter) so workers stay lightweight:

    RAWVIEWER_USE_PROCESS_POOL=1 \\
    RAWVIEWER_PROCESS_POOL_PYTHON="$(pwd)/.pixi/envs/default/bin/python" \\
      ./scripts/Launch/shell/launch_dev_full.sh

Dev (non-frozen) launches already use the interpreter as ``sys.executable``;
this module is the documented spawn target for frozen macOS packages and for
explicit helper wiring in ``image_load_manager``.
"""

from __future__ import annotations

import os
import sys


def _ensure_src_on_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


def main() -> int:
    _ensure_src_on_path()
    # Import only what decode workers need — never Qt / main window.
    from unified_image_processor import decode_raw_file  # noqa: F401

    print("libraw_pool_worker ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
