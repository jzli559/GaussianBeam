__all__ = []

_prefixes = {
    "f": -15,
    "p": -12,
    "n": -9,
    "u": -6,
    "m": -3,
    "c": -2,
    "_": 0,
    "k": 3,
    "M": 6,
    "G": 9,
    "T": 12
}

def _add_unit(unit: str, allowed_prefixes: str, bias: int = 0):
    for prefix, exp in _prefixes.items():
        if prefix in allowed_prefixes:
            name = unit if prefix == "_" else prefix + unit
            globals()[name] = 10.0 ** (exp + bias)
            __all__.append(name)

_add_unit("m", "fpnumc_k")
_add_unit("g", "num_k", bias=-3)
_add_unit("s", "fpnum_")
_add_unit("A", "num_")
_add_unit("K", "num_")
_add_unit("mol", "num_")
_add_unit("cd", "num_")

_add_unit("V", "num_k")
_add_unit("Ohm", "num_kM")
_add_unit("F", "pnum_")
_add_unit("H", "pnum_")
_add_unit("C", "pnum_")

_add_unit("Hz", "um_kMG")
_add_unit("rad", "um_")

_add_unit("W", "num_")
_add_unit("dBm", "_")

_add_unit("J", "m_k")
_add_unit("eV", "m_k")

_add_unit("N", "m_k")
_add_unit("Pa", "m_kM")
_add_unit("psi", "_")

if __name__ == "__main__":
    avail_units = []
    for u in sorted(__all__):
        avail_units.append(f"{u} = {globals()[u]:.1e}")
    print(f"All available units: {', '.join(avail_units)}")
