import os
import sys

if "gaussianbeam" not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)))))

from gaussianbeam.raytrace.main import Beam, Mode
from gaussianbeam.units import nm, um, mm, mrad
import sympy as sp

def test_numeric():
    beam_initial = Beam(wl=1762*nm, w0=5*um, mode=Mode.NUMERIC)
    beam_final = (
        beam_initial
        .prop(8*mm,1.0)
        .lens(8*mm)
        .prop(250*mm,1.0)
        .lens(100*mm)
    )
    print(beam_initial.q_final, beam_final.q_final)
    print(f"Final waist location: {beam_final.w0_loc/mm:.4f} mm")
    print(f"Final Rayleigh range: {beam_final.zR/mm:.4f} mm")
    print(f"Final waist diameter: {2*beam_final.w0/mm:.4f} mm")
    print(f"Final divergence 2θ: {2*beam_final.theta/mrad:.4f} mrad")
    print(f"Final beam diameter: {2*beam_final.w/mm:.4f} mm")

def test_symbolic():
    wl, w0, d1, f1, d2, f2 = sp.symbols("wl w0 d1 f1 d2 f2", real=True, positive=True)
    beam_initial = Beam(wl=wl, w0=w0, mode=Mode.SYMBOLIC)
    beam_final = beam_initial.copy().prop(d1,1).lens(f1).prop(d2,1).lens(f2)
    q1 = beam_initial.q_final
    q2 = beam_final.q_final
    q3 = q2.subs({wl:1762*nm, w0:5*um, d1:8*mm, f1:8*mm, d2:250*mm, f2:100*mm}).evalf()
    print(f"Initial q         : {q1}")
    print(f"Final q           : {q2}")
    print(f"Final q (numeric) : {q3}")

if __name__ == "__main__":
    test_numeric()
    test_symbolic()
