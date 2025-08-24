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

def _add_unit(unit: str, allowed_prefixes: str):
    for prefix, exp in _prefixes.items():
        if prefix in allowed_prefixes:
            name = unit if prefix == "_" else prefix + unit
            globals()[name] = 10.0 ** exp
            __all__.append(name)

_add_unit("s", "pnum_")
_add_unit("Hz", "um_kMG")
_add_unit("m", "fpnumc_k")
_add_unit("rad", "num_")

if __name__ == "__main__":
    avail_units = []
    for u in sorted(__all__):
        avail_units.append(f"{u} = {globals()[u]:.1e}")
    print(f"All available units: {', '.join(avail_units)}")
