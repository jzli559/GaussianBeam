"""Metadata describing the GUI-editable optical element types.

Each element type declares its parameters with a display label, a ``kind``
(``"length"`` parameters get an SI-prefix unit selector, ``"index"``
parameters are plain dimensionless floats), and a default value.

All values are stored internally in SI units (meters).  Sign conventions
follow the core library: radius of curvature is positive if the center is
after the surface; focal length is negative for diverging lenses.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParamSpec:
    """Description of one editable element parameter."""

    name: str        # internal key used in ElementSpec.params
    label: str       # label shown in the parameter form (Qt rich text allowed)
    kind: str        # "length" | "index"
    default: float
    allow_inf: bool = False  # whether the form offers an "infinity" checkbox


@dataclass(frozen=True)
class ElementTypeSpec:
    """Description of one optical element type for the GUI."""

    type_name: str   # key used in configs / model
    label: str       # human-readable name
    params: tuple


ELEMENT_TYPES = {
    spec.type_name: spec
    for spec in (
        ElementTypeSpec(
            "FreeSpace", "Free space (FreeSpace)",
            (
                ParamSpec("d", "Length d", "length", 50e-3),
            ),
        ),
        ElementTypeSpec(
            "ThinLens", "Thin lens (ThinLens)",
            (
                ParamSpec("f", "Focal length f", "length", 50e-3),
            ),
        ),
        ElementTypeSpec(
            "ThickLens", "Thick lens (ThickLens)",
            (
                ParamSpec("n", "Lens index n", "index", 1.5),
                ParamSpec("R1", "Front radius R<sub>1</sub>", "length", 50e-3, allow_inf=True),
                ParamSpec("R2", "Back radius R<sub>2</sub>", "length", -50e-3, allow_inf=True),
                ParamSpec("t", "Center thickness t", "length", 5e-3),
            ),
        ),
        ElementTypeSpec(
            "CurvedInterface", "Curved interface",
            (
                ParamSpec("n2", "Index after n<sub>2</sub>", "index", 1.5),
                ParamSpec("R", "Radius R", "length", float("inf"), allow_inf=True),
            ),
        ),
        ElementTypeSpec(
            "FlatInterface", "Flat interface",
            (
                ParamSpec("n2", "Index after n<sub>2</sub>", "index", 1.5),
            ),
        ),
    )
}


def default_params(type_name: str) -> dict:
    """Return a fresh dict of default parameter values for an element type."""
    return {p.name: p.default for p in ELEMENT_TYPES[type_name].params}
