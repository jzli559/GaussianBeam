import numpy as np
import sympy as sp
from typing import Literal
from enum import Enum
import logging
from abc import ABC, abstractmethod

C = 299792458.0  # speed of light in vacuum (m/s, exact)

# ===========================================
# Optical Element
# ===========================================

class Mode(Enum):
    NUMERIC = 0
    SYMBOLIC = 1

class OpticalElement(ABC):
    def __init__(self, mode: Mode = Mode.NUMERIC):
        assert isinstance(mode, Mode), "mode must be an instance of Mode Enum"

        self.mode = mode

    @abstractmethod
    def matrix(self) -> np.ndarray | sp.Matrix:
        pass

    @property
    def ABCD(self):
        return self.matrix()


class FreeSpace(OpticalElement):
    def __init__(self, d: float | sp.Symbol, n: float | sp.Symbol= 1.0003,
                 mode: Mode = Mode.NUMERIC):
        """
        :param d: distance (m)
        :type d: float | sympy.Symbol
        :param n: refractive index
        :type n: float | sympy.Symbol
        :param mode: numeric or symbolic
        :type mode: Mode
        """

        self.logger = logging.getLogger(self.__class__.__name__)

        if mode == Mode.NUMERIC and not all(
            isinstance(param, (int, float)) for param in [d, n]
        ):
            mode = Mode.SYMBOLIC
            self.logger.warning("Non-numeric parameters detected, switched to symbolic mode.")
        super().__init__(mode)
        self.d = d
        self.n = n

    def matrix(self) -> np.ndarray | sp.Matrix:
        if self.mode == Mode.NUMERIC:
            return np.array([[1, self.d/self.n], [0, 1]], dtype=complex)
        else: # Mode.SYMBOLIC
            return sp.Matrix([[1, self.d/self.n], [0, 1]])


class CurvedInterface(OpticalElement):
    def __init__(self, n1: float | sp.Symbol, n2: float | sp.Symbol,
                 R: float | sp.Symbol = np.inf,
                 mode: Mode = Mode.NUMERIC):
        """
        :param n1: refractive index before interface
        :type n1: float | sympy.Symbol
        :param n2: refractive index after interface
        :type n2: float | sympy.Symbol
        :param R: radius of curvature (m), positive if center is after interface
        :type R: float | sympy.Symbol
        :param mode: numeric or symbolic
        :type mode: Mode
        """

        self.logger = logging.getLogger(self.__class__.__name__)

        if mode == Mode.NUMERIC and not all(
            isinstance(param, (int, float)) for param in [n1, n2, R]
        ):
            mode = Mode.SYMBOLIC
            self.logger.warning("Non-numeric parameters detected, switched to symbolic mode.")
        super().__init__(mode)
        self.n1 = n1
        self.n2 = n2
        self.R = R

    def matrix(self) -> np.ndarray | sp.Matrix:
        if self.mode == Mode.NUMERIC:
            C = (self.n1 - self.n2) / (self.R * self.n2)
            D = self.n1 / self.n2
            return np.array([[1, 0], [C, D]], dtype=complex)
        else: # Mode.SYMBOLIC
            self.R = sp.oo if self.R == np.inf else self.R
            C = (self.n1 - self.n2) / (self.R * self.n2)
            D = self.n1 / self.n2
            return sp.Matrix([[1, 0], [C, D]])


class FlatInterface(CurvedInterface):
    def __init__(self, n1: float | sp.Symbol, n2: float | sp.Symbol,
                 mode: Mode = Mode.NUMERIC):
        """
        :param n1: refractive index before interface
        :type n1: float | sympy.Symbol
        :param n2: refractive index after interface
        :type n2: float | sympy.Symbol
        :param mode: numeric or symbolic
        :type mode: Mode
        """

        super().__init__(n1, n2, R=np.inf, mode=mode)

    def matrix(self) -> np.ndarray | sp.Matrix:
        return super().matrix()


class ThinLens(OpticalElement):
    def __init__(self, f: float | sp.Symbol,
                 mode: Mode = Mode.NUMERIC):
        """
        :param f: focal length (m)
        :type f: float | sympy.Symbol
        :param mode: numeric or symbolic
        :type mode: Mode
        """

        self.logger = logging.getLogger(self.__class__.__name__)

        if mode == Mode.NUMERIC and not isinstance(f, (int, float)):
            mode = Mode.SYMBOLIC
            self.logger.warning("Non-numeric focal length detected, switched to symbolic mode.")
        super().__init__(mode)
        self.f = f

    def matrix(self) -> np.ndarray | sp.Matrix:
        if self.mode == Mode.NUMERIC:
            return np.array([[1, 0], [-1/self.f, 1]], dtype=complex)
        else: # Mode.SYMBOLIC
            return sp.Matrix([[1, 0], [-1/self.f, 1]])


class ThickLens(OpticalElement):
    def __init__(self, n0: float | sp.Symbol, n: float | sp.Symbol,
                 R1: float | sp.Symbol, R2: float | sp.Symbol,
                 t: float | sp.Symbol,
                 mode: Mode = Mode.NUMERIC):
        """
        :param n0: refractive index outside the lens
        :type n0: float | sympy.Symbol
        :param n: refractive index of the lens itself
        :type n: float | sympy.Symbol
        :param R1: radius of curvature of first surface (m), positive if center is after surface
        :type R1: float | sympy.Symbol
        :param R2: radius of curvature of second surface (m), positive if center is after surface
        :type R2: float | sympy.Symbol
        :param t: center thickness of the lens (m)
        :type t: float | sympy.Symbol
        :param mode: numeric or symbolic
        :type mode: Mode
        """

        self.logger = logging.getLogger(self.__class__.__name__)
        if mode == Mode.NUMERIC and not all(
            isinstance(param, (int, float)) for param in [R1, R2, n0, n, t]
        ):
            mode = Mode.SYMBOLIC
            self.logger.warning("Non-numeric parameters detected, switched to symbolic mode.")
        super().__init__(mode)
        self.R1 = R1
        self.R2 = R2
        self.n0 = n0
        self.n = n
        self.t = t

    def matrix(self) -> np.ndarray | sp.Matrix:
        if self.mode == Mode.SYMBOLIC:
            self.R1 = sp.oo if self.R1 == np.inf else self.R1
            self.R2 = sp.oo if self.R2 == np.inf else self.R2
        M1 = CurvedInterface(self.n0, self.n, self.R1, self.mode).ABCD
        M2 = FreeSpace(self.t, self.n, self.mode).ABCD
        M3 = CurvedInterface(self.n, self.n0, self.R2, self.mode).ABCD
        if self.mode == Mode.NUMERIC:
            return M3 @ M2 @ M1
        else: # Mode.SYMBOLIC
            return M3 * M2 * M1


# ===========================================
# Beam
# ===========================================

class Beam:
    def __init__(self, wl: float | sp.Symbol,
                 w0: float | sp.Symbol,
                 mode: Mode = Mode.NUMERIC,
                 n: float | sp.Symbol = 1.0003):
        """
        Initialize a Gaussian beam.

        :math:`q(z) = z + i z_R`

        The refractive index is a **chain state**: ``prop()`` travels through
        the *current medium*, and only interface elements change it. There is
        no per-element entry index to keep consistent by hand.

        :param wl: vacuum wavelength (m). Internally converted to the
            frequency ``freq = c / wl``, which is medium-independent; the
            wavelength in the current medium is derived as
            ``wl = c / (n * freq)`` (see the :attr:`wl` property).
        :type wl: float | sympy.Symbol
        :param w0: waist radius (m)
        :type w0: float | sympy.Symbol
        :param theta: divergence angle (rad)
        :type theta: float | sympy.Symbol
        :param mode: numeric or symbolic
        :type mode: Mode
        :param n: refractive index of the medium the initial beam lives in
            (default: air, 1.0003)
        :type n: float | sympy.Symbol
        """

        assert isinstance(mode, Mode), "mode must be an instance of Mode Enum"

        self.logger = logging.getLogger(self.__class__.__name__)

        if mode == Mode.NUMERIC and not all(
            isinstance(param, (int, float)) for param in [wl, w0]
        ):
            mode = Mode.SYMBOLIC
            self.logger.warning("Non-numeric parameters detected, switched to symbolic mode.")

        self.mode = mode
        self.elements: list[OpticalElement] = []
        self.wl_initial: float | sp.Symbol = wl  # vacuum wavelength
        self.w0_initial: float | sp.Symbol = w0  # waist radius
        self.freq: float | sp.Symbol = C / wl  # frequency (medium-independent)
        self.n_initial: float | sp.Symbol = n  # medium the initial beam lives in
        self.n_current: float | sp.Symbol = n  # current medium (chain state)
        self.q0_initial: complex | sp.Expr = None  # initial q parameter
        if self.mode == Mode.NUMERIC:
            self.q0_initial = 1j * np.pi * self.w0_initial**2 * n / self.wl_initial
        else: # Mode.SYMBOLIC
            self.q0_initial = sp.I * sp.pi * self.w0_initial**2 * n / self.wl_initial

    def copy(self) -> "Beam":
        """
        Create a copy of the beam.

        :return: a copy of the beam
        :rtype: Beam
        """

        new_beam = Beam(self.wl_initial, self.w0_initial, self.mode, n=self.n_initial)
        new_beam.elements = self.elements.copy()
        new_beam.n_current = self.n_current
        return new_beam

    # --- Elements ---

    def prop(self, d: float | sp.Symbol) -> "Beam":
        """
        Propagate a distance ``d`` through the current medium.

        :param d: distance (m)
        :type d: float | sympy.Symbol
        :return: self
        :rtype: Beam
        """

        self.elements.append(FreeSpace(d, self.n_current, self.mode))
        return self

    def curved_interface(self, n2: float | sp.Symbol,
                  R: float | sp.Symbol = np.inf) -> "Beam":
        """
        Add a curved interface from the current medium into ``n2``.
        The current medium becomes ``n2`` afterwards.

        :param n2: refractive index after interface
        :type n2: float | sympy.Symbol
        :param R: radius of curvature (m), positive if center is after interface
        :type R: float | sympy.Symbol
        :return: self
        :rtype: Beam
        """

        self.elements.append(CurvedInterface(self.n_current, n2, R, self.mode))
        self.n_current = n2
        return self

    def flat_interface(self, n2: float | sp.Symbol) -> "Beam":
        """
        Add a flat interface from the current medium into ``n2``.
        The current medium becomes ``n2`` afterwards.

        :param n2: refractive index after interface
        :type n2: float | sympy.Symbol
        :return: self
        :rtype: Beam
        """

        self.elements.append(FlatInterface(self.n_current, n2, self.mode))
        self.n_current = n2
        return self

    def interface(self, n2: float | sp.Symbol) -> "Beam":
        """
        Add a flat interface (alias for `self.flat_interface`).

        :param n2: refractive index after interface
        :type n2: float | sympy.Symbol
        :return: self
        :rtype: Beam
        """

        return self.flat_interface(n2)

    def thin_lens(self, f: float | sp.Symbol) -> "Beam":
        """
        Add a thin lens.

        :param f: focal length (m)
        :type f: float | sympy.Symbol
        :return: self
        :rtype: Beam
        """

        self.elements.append(ThinLens(f, self.mode))
        return self

    def lens(self, f: float | sp.Symbol) -> "Beam":
        """
        Add a thin lens (alias for `self.thin_lens`).

        :param f: focal length (m)
        :type f: float | sympy.Symbol
        :return: self
        :rtype: Beam
        """

        return self.thin_lens(f)

    def thick_lens(self, n: float | sp.Symbol,
                   R1: float | sp.Symbol, R2: float | sp.Symbol,
                   t: float | sp.Symbol) -> "Beam":
        """
        Add a thick lens immersed in the current medium (unchanged afterwards).

        :param n: refractive index of the lens itself
        :type n: float | sympy.Symbol
        :param R1: radius of curvature of first surface (m), positive if center is after surface
        :type R1: float | sympy.Symbol
        :param R2: radius of curvature of second surface (m), positive if center is after surface
        :type R2: float | sympy.Symbol
        :param t: center thickness of the lens (m)
        :type t: float | sympy.Symbol
        :return: self
        :rtype: Beam
        """

        self.elements.append(ThickLens(self.n_current, n, R1, R2, t, self.mode))
        return self

    # --- Computation ---

    def _total_matrix(self) -> np.ndarray | sp.Matrix:
        if self.mode == Mode.NUMERIC:
            M = np.eye(2, dtype=complex)
            for elem in self.elements:
                M = elem.ABCD @ M
        else: # Mode.SYMBOLIC
            M = sp.eye(2)
            for elem in self.elements:
                M = elem.ABCD * M
        return M

    def _compute(self) -> complex | sp.Expr:
        M = self._total_matrix()
        if self.mode == Mode.NUMERIC:
            A, B, C, D = M.flatten()
            qf = (A * self.q0_initial + B) / (C * self.q0_initial + D)
        else: # Mode.SYMBOLIC
            A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
            qf = (A * self.q0_initial + B) / (C * self.q0_initial + D)
            qf = sp.simplify(qf)
        return qf

    # --- Results ---
    #
    # Extraction conventions (reduced-angle ABCD formalism):
    #   1/q = 1/R - i * wl0 / (pi * n * w^2)
    # with wl0 the vacuum wavelength and n the CURRENT medium. Hence
    #   w^2 = -wl0 / (pi * n * Im(1/q)),   R = 1 / Re(1/q).

    @property
    def wl(self) -> float | sp.Expr:
        """
        Get the wavelength in the current medium, wl = c / (n * freq).

        :return: wavelength in the current medium (m)
        :rtype: float | sympy.Expr
        """

        return self.wl_initial / self.n_current

    @property
    def q_final(self) -> complex | sp.Expr:
        """
        Get the q parameter after propagation through all elements.

        :return: q parameter
        :rtype: complex | sympy.Expr
        """
        
        return self._compute()

    @property
    def w0_loc(self) -> float | sp.Expr:
        """
        Get the waist location referred to the final optical element.

        :return: waist position (m)
        :rtype: float | sympy.Expr
        """

        if self.mode == Mode.NUMERIC:
            return -np.real(self.q_final)
        else: # Mode.SYMBOLIC
            return -sp.re(self.q_final)

    @property
    def zR(self) -> float | sp.Expr:
        """
        Get the Rayleigh range after propagation through all elements
        (in the final medium).

        :return: Rayleigh range (m)
        :rtype: float | sympy.Expr
        """

        if self.mode == Mode.NUMERIC:
            return np.imag(self.q_final)
        else: # Mode.SYMBOLIC
            return sp.im(self.q_final)

    @property
    def w0(self) -> float | sp.Expr:
        """
        Get the waist radius after propagation through all elements.

        :return: waist radius (m)
        :rtype: float | sympy.Expr
        """

        if self.mode == Mode.NUMERIC:
            return np.sqrt(self.wl_initial * self.zR / (np.pi * self.n_current))
        else: # Mode.SYMBOLIC
            return sp.sqrt(self.wl_initial * self.zR / (sp.pi * self.n_current))

    @property
    def theta(self) -> float | sp.Expr:
        """
        Get the divergence half-angle after propagation through all elements
        (in the final medium).

        :return: divergence angle (rad)
        :rtype: float | sympy.Expr
        """

        if self.mode == Mode.NUMERIC:
            return self.wl_initial / (np.pi * self.n_current * self.w0)
        else: # Mode.SYMBOLIC
            return self.wl_initial / (sp.pi * self.n_current * self.w0)

    @property
    def w(self) -> float | sp.Expr:
        """
        Get the radius of beam after propagation through all elements.

        :return: radius of beam (m)
        :rtype: float | sympy.Expr
        """

        if self.mode == Mode.NUMERIC:
            return np.sqrt(-self.wl_initial / (np.pi * self.n_current * np.imag(1/self.q_final)))
        else: # Mode.SYMBOLIC
            return sp.sqrt(-self.wl_initial / (sp.pi * self.n_current * sp.im(1/self.q_final)))
