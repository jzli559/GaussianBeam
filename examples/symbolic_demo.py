from gaussianbeam.raytrace.main import Beam, Mode
from gaussianbeam.units import nm, um, mm
import sympy as sp

def main():
    wl, w0, d1, f1, d2, f2 = sp.symbols("wl w0 d1 f1 d2 f2", real=True, positive=True)

    beam_initial = Beam(wl=wl, w0=w0, mode=Mode.SYMBOLIC)
    beam_final = beam_initial.copy().prop(d1,1).lens(f1).prop(d2,1).lens(f2)

    q1 = beam_initial.q_final
    q2 = beam_final.q_final
    q3 = q2.subs({wl:1762*nm, w0:5*um, d1:8*mm, f1:8*mm, d2:250*mm, f2:100*mm}).evalf()

    print("=== Symbolic Beam Propagation ===")
    print(f"Initial q         : {q1}")
    print(f"Final q           : {q2}")
    print(f"Final q (numeric) : {q3}")

if __name__ == "__main__":
    main()
