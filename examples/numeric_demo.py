from gaussianbeam.raytrace.main import Beam, Mode
from gaussianbeam.units import nm, um, mm, mrad

def main():
    beam_initial = Beam(wl=1762*nm, w0=5*um, mode=Mode.NUMERIC)
    beam_final = (
        beam_initial.copy()
        .prop(8*mm, 1.0)
        .lens(8*mm)
        .prop(250*mm, 1.0)
        .lens(100*mm)
    )

    print("=== Numeric Beam Propagation ===")
    print(f"Initial q         : {beam_initial.q_final}")
    print(f"Final q           : {beam_final.q_final}")
    print(f"Final waist location : {beam_final.w0_loc/mm:.4f} mm")
    print(f"Final Rayleigh range: {beam_final.zR/mm:.4f} mm")
    print(f"Final waist diameter: {2*beam_final.w0/mm:.4f} mm")
    print(f"Final divergence 2θ: {2*beam_final.theta/mrad:.4f} mrad")
    print(f"Final beam diameter : {2*beam_final.w/mm:.4f} mm")

if __name__ == "__main__":
    main()
