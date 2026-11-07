"""
Bootstrap installer for RAWviewer.

Original monolithic module split into ``bootstrap_pkg`` package.
This module re-exports the full public interface for backward compatibility.
"""
# Re-export everything from the package
from bootstrap_pkg import main
from bootstrap_pkg.constants import *  # noqa: F401, F403
from bootstrap_pkg.gui import *  # noqa: F401, F403
from bootstrap_pkg.helpers import *  # noqa: F401, F403
from bootstrap_pkg.worker import *  # noqa: F401, F403

if __name__ == "__main__":
    main()
