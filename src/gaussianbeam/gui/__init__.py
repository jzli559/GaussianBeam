"""Interactive GUI for Gaussian beam system design.

Launch with ``gaussianbeam-gui`` or ``python -m gaussianbeam.gui.app``.
"""

__all__ = ["main"]


def main():
    """Launch the GUI (imports PySide6 lazily)."""
    from .app import main as _main

    _main()
