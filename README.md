# GaussianBeam

A Python toolkit for modeling **Gaussian beam propagation** using the **ABCD matrix formalism**.  
Supports both **symbolic (SymPy)** and **numeric (NumPy)** calculations, with intuitive chain-style propagation of optical elements.

## Features

- Numeric and symbolic Gaussian beam propagation.
- Abstracted optical elements: free space propagation, lenses, etc.
- Intuitive chain-style API.
- Supports unit handling: nm, um, mm, cm, m, mrad, ...
- **Interactive GUI** (PySide6 + pyqtgraph): design optical systems visually —
  add/insert/remove/reorder elements, edit parameters with unit selectors,
  watch the beam envelope w(z) update live, and hover over the plot to read
  w(z) and wavefront curvature R(z) at any position. Configurations can be
  saved and loaded as JSON.

## Installation

Note that installing this package will automatically install the required dependencies: `sympy`, `numpy`, and the GUI dependencies `PySide6` + `pyqtgraph`.

You can directly install the package in editable/development mode:
```bash
git clone https://github.com/jzli559/GaussianBeam.git
cd GaussianBeam
pip install -e .
```

Or install from PyPI (not yet published):
```bash
pip install gaussianbeam
```

## Quick Start

```python
from gaussianbeam.raytrace.main import Beam, Mode
from gaussianbeam.units import nm, um, mm, mrad

beam_initial = Beam(wl=1064*nm, w0=5*um)

beam_final = (
    beam_initial.copy()
    .prop(100*mm)
    .lens(f=50*mm)
    .prop(100*mm)
)

print(f"Final waist location : {beam_final.w0_loc/mm:.4f} mm")
```

All example scripts are in the `examples/` directory.
For complete numeric demos, see [py demo](examples/numeric_demo.py);
For complete symbolic demos, see [py demo](examples/symbolic_demo.py).

## GUI

Launch the interactive beam designer with either of:

```bash
gaussianbeam-gui
python examples/gui_demo.py
```

- **Elements panel** — add / insert / remove elements (free space, thin & thick
  lenses, curved & flat interfaces), reorder them, and edit parameters with
  SI-unit selectors (radii accept ∞).
- **Beam envelope plot** — the ±w(z) envelope along the propagation axis with
  element markers; every edit re-traces instantly.
- **Hover probe** — move the mouse over the plot to read the z position, beam
  radius w(z), and wavefront curvature R(z) (diverging / converging / plane).
- **System output** — output waist size and location, Rayleigh range, and
  divergence angle of the whole system.
- **Configuration** — save and load optical systems as JSON files.
