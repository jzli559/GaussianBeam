"""Optical system model for the GUI.

The model stores the initial beam parameters and an ordered list of
element descriptions (:class:`ElementSpec`).  It rebuilds a numeric
:class:`~gaussianbeam.raytrace.main.Beam` on demand and computes a sampled
propagation trace ``w(z)``, ``R(z)`` for plotting, plus per-segment
closed-form probes for the mouse hover readout.

Computation strategy (numeric mode only):

* Walk the element list, transforming the complex q parameter with the
  Mobius map ``q -> (A q + B) / (C q + D)`` of each element's ABCD matrix
  (matrices are taken from the core element classes to stay consistent).
* Inside a ``FreeSpace(d, n)`` segment, q evolves linearly,
  ``q(z) = q_in + (z - z0) / n``, so a whole segment is sampled with one
  vectorized NumPy expression.
* Beam radius and wavefront curvature follow from
  ``w^2 = -wl / (pi * Im(1/q))`` and ``R = 1 / Re(1/q)``
  (``R = inf`` means a plane wavefront; ``R > 0`` means the center of
  curvature is after the plane, i.e. a diverging beam).

All quantities are SI (meters, radians).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..raytrace.main import (
    Beam,
    CurvedInterface,
    FlatInterface,
    FreeSpace,
    ThickLens,
    ThinLens,
)
from .elements import ELEMENT_TYPES, default_params

# (name, factor to SI) pairs, ordered smallest to largest.
LENGTH_UNITS = (("nm", 1e-9), ("um", 1e-6), ("mm", 1e-3), ("cm", 1e-2), ("m", 1.0))


def format_length(x: float) -> str:
    """Format an SI length with an automatic SI prefix."""
    if x is None:
        return "—"
    if np.isinf(x):
        return "∞"
    ax = abs(x)
    if ax == 0:
        return "0 m"
    for name, factor in LENGTH_UNITS:
        if ax / factor < 1000:
            return f"{x / factor:.4g} {name}"
    return f"{x:.4g} m"


def _mobius(M: np.ndarray, q: complex) -> complex:
    """Apply the Mobius map of ABCD matrix M to q."""
    A, B, C, D = np.asarray(M, dtype=complex).flatten()
    return (A * q + B) / (C * q + D)


def q_to_wR(q, wl: float):
    """Beam radius ``w`` and wavefront curvature radius ``R`` from q.

    Works on scalars or NumPy arrays.  See module docstring for formulas.
    """
    q = np.asarray(q, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / q
        w = np.sqrt(np.clip(-wl / (np.pi * inv.imag), 0.0, None))
        R = np.where(inv.real == 0.0, np.inf, 1.0 / inv.real)
    return w, R


def element_matrix(spec: "ElementSpec") -> np.ndarray:
    """Numeric ABCD matrix of an element, built with the core classes."""
    p = spec.params
    if spec.type == "FreeSpace":
        return FreeSpace(p["d"], p.get("n", 1.0)).ABCD
    if spec.type == "ThinLens":
        return ThinLens(p["f"]).ABCD
    if spec.type == "ThickLens":
        return ThickLens(p["n0"], p["n"], p["R1"], p["R2"], p["t"]).ABCD
    if spec.type == "CurvedInterface":
        return CurvedInterface(p["n1"], p["n2"], p.get("R", np.inf)).ABCD
    if spec.type == "FlatInterface":
        return FlatInterface(p["n1"], p["n2"]).ABCD
    raise ValueError(f"Unknown element type: {spec.type!r}")


@dataclass
class ElementSpec:
    """One optical element: a type name plus SI parameter values."""

    type: str
    params: dict = field(default_factory=dict)

    @classmethod
    def create(cls, type_name: str) -> "ElementSpec":
        return cls(type_name, default_params(type_name))

    def summary(self) -> str:
        """One-line description for the element list widget."""
        label = ELEMENT_TYPES[self.type].label.split(" (")[0]
        parts = []
        for ps in ELEMENT_TYPES[self.type].params:
            v = self.params.get(ps.name, ps.default)
            if ps.kind == "length":
                parts.append(f"{ps.name}={format_length(v)}")
            else:
                parts.append(f"{ps.name}={v:.4g}")
        return f"{label}: {', '.join(parts)}"


@dataclass
class Segment:
    """A FreeSpace portion of the trace; enables closed-form probing."""

    z0: float        # start position (m)
    z1: float        # end position (m)
    q_in: complex    # q parameter entering the segment at z0
    n: float         # refractive index of the segment


@dataclass
class Marker:
    """Position marker for a non-FreeSpace element."""

    z0: float
    z1: float        # == z0 for thin elements, z0 + t for ThickLens
    label: str
    kind: str        # "lens" | "interface" | "thick"


@dataclass
class Trace:
    """Result of tracing a system: sampled curves plus probe segments."""

    z: np.ndarray
    w: np.ndarray
    R: np.ndarray
    segments: list
    markers: list
    total_length: float
    final: dict      # final beam properties (w0, zR, w0_loc, theta, w)
    wl: float

    def probe(self, z: float):
        """Closed-form (w, R) at z, or None if z is outside all segments."""
        for seg in self.segments:
            if seg.z0 <= z <= seg.z1:
                q = seg.q_in + (z - seg.z0) / seg.n
                w, R = q_to_wR(q, self.wl)
                return float(w), float(R)
        return None


@dataclass
class OpticalSystem:
    """Beam parameters plus an ordered list of optical elements."""

    wl: float = 632.8e-9
    w0: float = 0.3e-3
    elements: list = field(default_factory=list)

    @classmethod
    def default(cls) -> "OpticalSystem":
        """A small demo system: collimate, focus, propagate."""
        return cls(
            elements=[
                ElementSpec.create("FreeSpace"),
                ElementSpec("ThinLens", {"f": 50e-3}),
                ElementSpec("FreeSpace", {"d": 150e-3, "n": 1.0}),
            ],
        )

    # --- Computation ---

    def build_beam(self) -> Beam:
        """Rebuild a numeric core Beam from the element list."""
        beam = Beam(self.wl, self.w0)
        for spec in self.elements:
            p = spec.params
            if spec.type == "FreeSpace":
                beam.prop(p["d"], p.get("n", 1.0))
            elif spec.type == "ThinLens":
                beam.lens(p["f"])
            elif spec.type == "ThickLens":
                beam.thick_lens(p["n0"], p["n"], p["R1"], p["R2"], p["t"])
            elif spec.type == "CurvedInterface":
                beam.curved_interface(p["n1"], p["n2"], p.get("R", np.inf))
            elif spec.type == "FlatInterface":
                beam.flat_interface(p["n1"], p["n2"])
        return beam

    def trace(self, n_samples: int = 200) -> Trace:
        """Sample w(z), R(z) along the whole system.

        :param n_samples: sample points per FreeSpace segment
        """
        q = 1j * np.pi * self.w0**2 / self.wl  # waist at z = 0
        z = 0.0
        zs, ws, Rs = [], [], []
        segments, markers = [], []

        for spec in self.elements:
            if spec.type == "FreeSpace":
                d, n = spec.params["d"], spec.params.get("n", 1.0)
                t = np.linspace(0.0, d, n_samples)
                qq = q + t / n
                w, R = q_to_wR(qq, self.wl)
                zs.append(z + t)
                ws.append(w)
                Rs.append(R)
                segments.append(Segment(z, z + d, q, n))
                z += d
                q = q + d / n
            elif spec.type == "ThickLens":
                t = spec.params["t"]
                markers.append(Marker(z, z + t, spec.summary(), "thick"))
                q = _mobius(element_matrix(spec), q)
                z += t
            else:
                kind = "lens" if spec.type == "ThinLens" else "interface"
                markers.append(Marker(z, z, spec.summary(), kind))
                q = _mobius(element_matrix(spec), q)

        beam = self.build_beam()
        final = {
            "w0": beam.w0,
            "zR": beam.zR,
            "w0_loc": beam.w0_loc,
            "theta": beam.theta,
            "w": beam.w,
        }
        cat = lambda a: np.concatenate(a) if a else np.array([])
        return Trace(
            z=cat(zs), w=cat(ws), R=cat(Rs),
            segments=segments, markers=markers, total_length=z,
            final=final, wl=self.wl,
        )

    # --- Serialization ---

    _INF = "__inf__"
    _NINF = "__-inf__"

    def to_dict(self) -> dict:
        def enc(v):
            if v == np.inf:
                return self._INF
            if v == -np.inf:
                return self._NINF
            return float(v)

        return {
            "version": 1,
            "beam": {"wl": self.wl, "w0": self.w0},
            "elements": [
                {"type": s.type, "params": {k: enc(v) for k, v in s.params.items()}}
                for s in self.elements
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OpticalSystem":
        def dec(v):
            if v == cls._INF:
                return np.inf
            if v == cls._NINF:
                return -np.inf
            return float(v)

        beam = data.get("beam", {})
        system = cls(wl=float(beam.get("wl", 632.8e-9)),
                     w0=float(beam.get("w0", 0.3e-3)))
        for e in data.get("elements", []):
            spec = ElementSpec.create(e["type"])
            for k, v in e.get("params", {}).items():
                if k in spec.params:
                    spec.params[k] = dec(v)
            system.elements.append(spec)
        return system

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "OpticalSystem":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
