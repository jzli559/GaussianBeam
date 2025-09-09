# GaussianBeam

A Python toolkit for modeling **Gaussian beam propagation** using the **ABCD matrix formalism**.  
Supports both **symbolic (SymPy)** and **numeric (NumPy)** calculations, with intuitive chain-style propagation of optical elements.

## Features

- Numeric and symbolic Gaussian beam propagation.
- Abstracted optical elements: free space propagation, lenses, etc.
- Intuitive chain-style API.
- Supports unit handling: nm, um, mm, cm, m, mrad, ...

## Installation

Note that installing this package will automatically install the required dependencies: `sympy` and `numpy`.

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
