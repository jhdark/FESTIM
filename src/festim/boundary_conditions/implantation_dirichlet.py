from typing import Optional

import numpy as np
import ufl
from dolfinx import fem

from festim import k_B
from festim.boundary_conditions.dirichlet_bc import FixedConcentrationBC
from festim.helpers import (
    as_fenics_constant,
    as_mapped_function,
    get_interpolation_points,
)
from festim.species import Species
from festim.subdomain.surface_subdomain import SurfaceSubdomain


def dc_imp(
    mod, T, phi, R_p, D_0, E_D, Kr_0=None, E_Kr=None, Kd_0=None, E_Kd=None, P=None
):
    D = D_0 * mod.exp(-E_D / k_B / T)
    value = phi * R_p / D
    if Kr_0 is not None:
        Kr = Kr_0 * mod.exp(-E_Kr / k_B / T)
        if Kd_0 is not None:
            Kd = Kd_0 * mod.exp(-E_Kd / k_B / T)
            value += ((phi + Kd * P) / Kr) ** 0.5
        else:
            value += (phi / Kr) ** 0.5

    return value


class ImplantationDirichlet(FixedConcentrationBC):
    """Subclass of FixedConcentrationBC representing an approximation of an implanted
    flux of hydrogen.
    The details of the approximation can be found in
    https://www.nature.com/articles/s41598-020-74844-w

    c = phi*R_p/D + ((phi+Kd*P)/Kr)**0.5

    Args:
        subdomain: the surfaces of the BC
        species: The name of the species
        phi: implanted flux (H/m2/s)
        R_p: implantation depth (m)
        D_0: diffusion coefficient pre-exponential factor (m2/s)
        E_D: diffusion coefficient activation energy (eV)
        Kr_0: recombination coefficient pre-exponential
            factor (m^4/s). If None, instantaneous recombination will be
            assumed. Defaults to None.
        E_Kr: recombination coefficient activation
            energy (eV). Defaults to None.
        Kd_0: dissociation coefficient pre-exponential
            factor (m-2 s-1 Pa-1). If None, instantaneous dissociation will be
            assumed. Defaults to None.
        E_Kd: dissociation coefficient activation
            energy (eV). Defaults to None.
        P: partial pressure of H (Pa). Defaults to None.

    Attributes:
        subdomain: the surface of the BC
        species: The name of the species
        phi: implanted flux (H/m2/s)
        R_p: implantation depth (m)
        D_0: diffusion coefficient pre-exponential factor (m2/s)
        E_D: diffusion coefficient activation energy (eV)
        Kr_0: recombination coefficient pre-exponential
            factor (m^4/s). If None, instantaneous recombination will be
            assumed. Defaults to None.
        E_Kr: recombination coefficient activation
            energy (eV). Defaults to None.
        Kd_0: dissociation coefficient pre-exponential
            factor (m-2 s-1 Pa-1). If None, instantaneous dissociation will be
            assumed. Defaults to None.
        E_Kd: dissociation coefficient activation
            energy (eV). Defaults to None.
        P: partial pressure of H (Pa). Defaults to None.

        phi_fenics: implanted flux in a fenics format
        R_p_fenics: implantation depth in a fenics format
        P_fenics:partial pressure of H in a fenics format
        time_dependent: True if the value of the bc is time dependent
        temperature_dependent: True if the value of the bc is temperature dependent
        bc_expr: The expression of the boundary condition that is used to
            update the `value_fenics`
        value_fenics: The value of the boundary condition in fenics format


    """

    subdomain: SurfaceSubdomain
    species: Species
    phi: float
    R_p: float
    D_0: float
    E_D: float
    Kr_0: Optional[float]
    E_Kr: Optional[float]
    Kd_0: Optional[float]
    E_Kd: Optional[float]
    P: Optional[float]

    def __init__(
        self,
        subdomain,
        species,
        phi,
        R_p,
        D_0,
        E_D,
        Kr_0=None,
        E_Kr=None,
        Kd_0=None,
        E_Kd=None,
        P=None,
    ) -> None:
        super().__init__(subdomain=subdomain, value=None, species=species)

        self.phi = phi
        self.R_p = R_p
        self.D_0 = D_0
        self.E_D = E_D
        self.Kr_0 = Kr_0
        self.E_Kr = E_Kr
        self.Kd_0 = Kd_0
        self.E_Kd = E_Kd
        self.P = P

    def create_value(
        self,
        function_space: fem.FunctionSpace,
        temperature: float | fem.Constant,
        t: float | fem.Constant,
    ):
        # handle conversion of phi
        if isinstance(
            self.phi, float | int | fem.Constant | fem.Expression | ufl.core.expr.Expr
        ):
            self.phi_fenics = self.phi
        elif callable(self.phi):
            self.phi_fenics = as_mapped_function(
                value=self.phi,
                function_space=function_space,
                t=t,
                temperature=temperature,
            )

        # handle conversion of R_p
        if isinstance(
            self.R_p, float | int | fem.Constant | fem.Expression | ufl.core.expr.Expr
        ):
            self.R_p_fenics = self.R_p
        elif callable(self.R_p):
            self.R_p_fenics = as_mapped_function(
                value=self.R_p,
                function_space=function_space,
                t=t,
                temperature=temperature,
            )

        # handle conversion of P
        if self.P is None:
            self.P_fenics = self.P
        elif isinstance(
            self.R_p, float | int | fem.Constant | fem.Expression | ufl.core.expr.Expr
        ):
            self.R_p_fenics = self.R_p
        elif callable(self.R_p):
            self.R_p_fenics = as_mapped_function(
                value=self.R_p,
                function_space=function_space,
                t=t,
                temperature=temperature,
            )
            args = [
                self.phi,
                self.R_p,
                self.D_0,
                self.E_D,
                self.Kr_0,
                self.E_Kr,
                self.Kd_0,
                self.E_Kd,
                self.P,
            ]
            mod = (
                np
                if all(isinstance(arg, int | float | type(None)) for arg in args)
                else ufl
            )
            value_BC = dc_imp(
                mod=mod,
                T=temperature,
                phi=self.phi_fenics,
                R_p=self.R_p_fenics,
                D_0=self.D_0,
                E_D=self.E_D,
                Kr_0=self.Kr_0,
                E_Kr=self.E_Kr,
                Kd_0=self.Kd_0,
                E_Kd=self.E_Kd,
                P=self.P_fenics,
            )
        if isinstance(value_BC, float | int):
            self.value_fenics = as_fenics_constant(
                value=value_BC, mesh=function_space.mesh
            )
        else:
            self.value_fenics = fem.Function(function_space)
            self.bc_expr = fem.Expression(
                value_BC,
                get_interpolation_points(function_space.element),
            )
            self.value_fenics.interpolate(self.bc_expr)
