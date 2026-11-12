"""Hydride precipitation and dissolution reactions.

The three mechanisms of the Hydride Nucleation-Growth-Dissolution (HNGD) model
(Passelaigue et al., J. Nucl. Mater. 544 (2021) 152683), each written as a
:class:`~festim.reactions.reaction.ReactionBase` between a mobile solute species
(hydrogen in solid solution) and an immobile hydride species (precipitated
hydrogen).
"""

from collections.abc import Callable

import ufl

from festim.reactions.reaction import ReactionBase
from festim.species import Species
from festim.subdomain.volume_subdomain import VolumeSubdomain


class HydrideReaction(ReactionBase):
    """Base class for reactions exchanging hydrogen between a solute species and a
    hydride species.

    Subclasses implement :meth:`rate`, the net rate as a ufl expression of the
    temperature and of the solute and hydride concentrations. The rate is wrapped
    in a callable whose species arguments are mapped to ``solute`` and ``hydride``
    through ``arg_to_species``, so it is evaluated by festim.Value like any other
    species-dependent reaction rate.

    Arguments:
        solute: The mobile species holding the hydrogen in solid solution.
        hydride: The immobile species holding the precipitated hydrogen.
        volume: The volume subdomain where the reaction takes place.
        solubility: The terminal solid solubility the reaction drives the solute
            towards, a float or a callable of the temperature ``T``.
        rate_constant: The kinetic rate constant of the reaction, a float or a
            callable of the temperature ``T``.
        dissolves: True if the reaction moves hydrogen from the hydride into solid
            solution, False if it moves it from solid solution into the hydride.

    Attributes:
        solute: The mobile species holding the hydrogen in solid solution.
        hydride: The immobile species holding the precipitated hydrogen.
        solubility: The terminal solid solubility.
        rate_constant: The kinetic rate constant of the reaction.
    """

    solute: Species
    hydride: Species
    solubility: float | Callable
    rate_constant: float | Callable

    def __init__(
        self,
        solute: Species,
        hydride: Species,
        volume: VolumeSubdomain,
        solubility: float | Callable,
        rate_constant: float | Callable,
        dissolves: bool,
    ) -> None:
        self.solute = solute
        self.hydride = hydride
        self.solubility = solubility
        self.rate_constant = rate_constant

        # the lambda reads the parameters through self, so changing them after
        # construction is picked up when the problem is initialised
        super().__init__(
            reaction_rate=lambda T, c_s, c_h: self.rate(T, c_s, c_h),
            volume=volume,
            reactant=hydride if dissolves else solute,
            product=solute if dissolves else hydride,
            arg_to_species={"c_s": solute, "c_h": hydride},
        )

    @property
    def solute(self) -> Species:
        return self._solute

    @solute.setter
    def solute(self, value: Species) -> None:
        if not isinstance(value, Species):
            raise TypeError(
                f"solute must be a festim.Species, not {type(value).__name__}"
            )
        self._solute = value

    @property
    def hydride(self) -> Species:
        return self._hydride

    @hydride.setter
    def hydride(self, value: Species) -> None:
        if not isinstance(value, Species):
            raise TypeError(
                f"hydride must be a festim.Species, not {type(value).__name__}"
            )
        self._hydride = value

    @property
    def solubility(self) -> float | Callable:
        return self._solubility

    @solubility.setter
    def solubility(self, value: float | Callable) -> None:
        self._solubility = _check_temperature_function("solubility", value)

    @property
    def rate_constant(self) -> float | Callable:
        return self._rate_constant

    @rate_constant.setter
    def rate_constant(self, value: float | Callable) -> None:
        self._rate_constant = _check_temperature_function("rate_constant", value)

    def rate(self, T, c_s, c_h) -> ufl.core.expr.Expr:
        """The net reaction rate.

        Args:
            T: The temperature.
            c_s: The solute concentration.
            c_h: The hydride concentration.

        Returns:
            The net rate, always non-negative, at which hydrogen moves from the
            reactant to the product.
        """
        raise NotImplementedError

    def _evaluate(self, value: float | Callable, T) -> ufl.core.expr.Expr | float:
        """A float parameter as is, or a callable parameter evaluated at ``T``."""
        return value(T) if callable(value) else value


class HydrideDissolution(HydrideReaction):
    """Dissolution of hydrides into solid solution.

    When the solute concentration is below the dissolution solubility
    :math:`\\text{TSS}_D` and hydrides are present, hydrogen moves from the hydride
    into solid solution at the rate

    .. math::

        R = K_D \\max\\left(\\min\\left(\\text{TSS}_D - c_s, c_h\\right), 0\\right)

    The driving force is capped by the hydride concentration, as in the reference
    HNGD implementation (github.com/FloPasselaigue/HNGD), so a hydride cannot
    release more hydrogen than it holds: the rate falls continuously to zero as
    the last hydride dissolves, instead of switching off at :math:`c_h = 0`.

    Arguments:
        solute: The mobile species holding the hydrogen in solid solution.
        hydride: The immobile species holding the precipitated hydrogen.
        volume: The volume subdomain where the reaction takes place.
        solubility: The dissolution terminal solid solubility
            :math:`\\text{TSS}_D`, a float or a callable of the temperature ``T``.
        rate_constant: The dissolution rate constant :math:`K_D`, a float or a
            callable of the temperature ``T``.

    Examples:

        .. testcode:: HydrideDissolution

            import festim as F
            import ufl

            material = F.Material(D_0=1, E_D=0)
            volume = F.VolumeSubdomain(id=1, material=material)

            solute = F.Species("C_SS")
            hydride = F.Species("C_prec", mobile=False)

            reaction = F.HydrideDissolution(
                solute=solute,
                hydride=hydride,
                volume=volume,
                solubility=lambda T: 1.02e5 * ufl.exp(-0.37 / (F.k_B * T)),
                rate_constant=lambda T: 1.11e3 * ufl.exp(-0.46 / (F.k_B * T)),
            )
            print(reaction)

        .. testoutput:: HydrideDissolution

            C_prec --> C_SS
    """

    def __init__(
        self,
        solute: Species,
        hydride: Species,
        volume: VolumeSubdomain,
        solubility: float | Callable,
        rate_constant: float | Callable,
    ) -> None:
        super().__init__(
            solute, hydride, volume, solubility, rate_constant, dissolves=True
        )

    def rate(self, T, c_s, c_h) -> ufl.core.expr.Expr:
        TSS_D = self._evaluate(self.solubility, T)
        K_D = self._evaluate(self.rate_constant, T)
        return K_D * ufl.max_value(ufl.min_value(TSS_D - c_s, c_h), 0)


class HydrideNucleation(HydrideReaction):
    """Nucleation of new hydrides from a supersaturated solid solution.

    When the solute concentration exceeds the precipitation solubility
    :math:`\\text{TSS}_P`, hydrogen moves from solid solution into the hydride at
    the rate

    .. math::

        R = K_N \\left(c_s - \\text{TSS}_P\\right)

    Arguments:
        solute: The mobile species holding the hydrogen in solid solution.
        hydride: The immobile species holding the precipitated hydrogen.
        volume: The volume subdomain where the reaction takes place.
        solubility: The precipitation terminal solid solubility
            :math:`\\text{TSS}_P`, a float or a callable of the temperature ``T``.
        rate_constant: The nucleation rate constant :math:`K_N`, a float or a
            callable of the temperature ``T``.

    Examples:

        .. testcode:: HydrideNucleation

            import festim as F
            import ufl

            material = F.Material(D_0=1, E_D=0)
            volume = F.VolumeSubdomain(id=1, material=material)

            solute = F.Species("C_SS")
            hydride = F.Species("C_prec", mobile=False)

            reaction = F.HydrideNucleation(
                solute=solute,
                hydride=hydride,
                volume=volume,
                solubility=lambda T: 3.08e4 * ufl.exp(-0.26 / (F.k_B * T)),
                rate_constant=0.017,
            )
            print(reaction)

        .. testoutput:: HydrideNucleation

            C_SS --> C_prec
    """

    def __init__(
        self,
        solute: Species,
        hydride: Species,
        volume: VolumeSubdomain,
        solubility: float | Callable,
        rate_constant: float | Callable,
    ) -> None:
        super().__init__(
            solute, hydride, volume, solubility, rate_constant, dissolves=False
        )

    def rate(self, T, c_s, c_h) -> ufl.core.expr.Expr:
        TSS_P = self._evaluate(self.solubility, T)
        K_N = self._evaluate(self.rate_constant, T)
        return ufl.conditional(ufl.gt(c_s, TSS_P), K_N * (c_s - TSS_P), 0)


class HydrideGrowth(HydrideReaction):
    """Growth of existing hydrides, following the Johnson-Mehl-Avrami-Kolmogorov
    (JMAK) kinetics.

    When the solute concentration exceeds the dissolution solubility
    :math:`\\text{TSS}_D` and hydrides are present, hydrogen moves from solid
    solution into the hydride at the rate

    .. math::

        R = K_G \\left(c_\\text{tot} - \\text{TSS}_D\\right)
            p \\left(1 - x\\right) \\left(-\\ln(1 - x)\\right)^{1 - 1/p}

    where :math:`c_\\text{tot} = c_s + c_h` is the total hydrogen concentration,
    :math:`x = c_h / (c_\\text{tot} - \\text{TSS}_D)` the precipitation
    advancement and :math:`p` the Avrami exponent. Growth needs existing hydrides,
    so the hydride concentration must be seeded with a small positive value if no
    nucleation reaction is present to create them. Growth is only active where the
    hydride concentration exceeds ``hydride_threshold``: after a complete
    dissolution the hydride concentration decays towards zero without reaching it,
    and the residue must not act as a growth seed.

    The advancement :math:`x` (and the check that hydrides exist) is evaluated
    with the concentrations of the previous time step, the rest of the rate being
    implicit. Near :math:`x = 0` the JMAK factor behaves as :math:`x^{1 - 1/p}`,
    whose infinite slope makes a fully implicit Newton step overshoot to a
    negative hydride concentration, switching growth off so that it never starts.

    Arguments:
        solute: The mobile species holding the hydrogen in solid solution.
        hydride: The immobile species holding the precipitated hydrogen.
        volume: The volume subdomain where the reaction takes place.
        solubility: The dissolution terminal solid solubility
            :math:`\\text{TSS}_D`, a float or a callable of the temperature ``T``.
        rate_constant: The growth rate constant :math:`K_G`, a float or a callable
            of the temperature ``T``.
        avrami_exponent: The Avrami exponent :math:`p`. Defaults to 2.5 (platelet
            hydrides).
        hydride_threshold: The hydride concentration above which hydrides are
            considered present and growth is active, in the units of the hydride
            concentration. Defaults to 0. Set it well below any intended seed but
            above the residue left by dissolution (e.g. 1e-8 wt.ppm).

    Attributes:
        avrami_exponent: The Avrami exponent.
        hydride_threshold: The hydride concentration above which growth is active.

    Examples:

        .. testcode:: HydrideGrowth

            import festim as F
            import ufl

            material = F.Material(D_0=1, E_D=0)
            volume = F.VolumeSubdomain(id=1, material=material)

            solute = F.Species("C_SS")
            hydride = F.Species("C_prec", mobile=False)

            reaction = F.HydrideGrowth(
                solute=solute,
                hydride=hydride,
                volume=volume,
                solubility=lambda T: 1.02e5 * ufl.exp(-0.37 / (F.k_B * T)),
                rate_constant=1.71e-4,
            )
            print(reaction)

        .. testoutput:: HydrideGrowth

            C_SS --> C_prec
    """

    avrami_exponent: float
    hydride_threshold: float

    # floor on 1 - x, keeping ln(1 - x) finite at full precipitation
    _ONE_MINUS_X_MIN = 1e-14

    def __init__(
        self,
        solute: Species,
        hydride: Species,
        volume: VolumeSubdomain,
        solubility: float | Callable,
        rate_constant: float | Callable,
        avrami_exponent: float = 2.5,
        hydride_threshold: float = 0.0,
    ) -> None:
        self.avrami_exponent = avrami_exponent
        self.hydride_threshold = hydride_threshold
        super().__init__(
            solute, hydride, volume, solubility, rate_constant, dissolves=False
        )

    @property
    def avrami_exponent(self) -> float:
        return self._avrami_exponent

    @avrami_exponent.setter
    def avrami_exponent(self, value: float) -> None:
        if not isinstance(value, float | int) or isinstance(value, bool):
            raise TypeError(
                f"avrami_exponent must be a float, not {type(value).__name__}"
            )
        if value <= 0:
            raise ValueError(f"avrami_exponent must be positive, not {value}")
        self._avrami_exponent = value

    @property
    def hydride_threshold(self) -> float:
        return self._hydride_threshold

    @hydride_threshold.setter
    def hydride_threshold(self, value: float) -> None:
        if not isinstance(value, float | int) or isinstance(value, bool):
            raise TypeError(
                f"hydride_threshold must be a float, not {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(f"hydride_threshold must be non-negative, not {value}")
        self._hydride_threshold = value

    def rate(self, T, c_s, c_h) -> ufl.core.expr.Expr:
        TSS_D = self._evaluate(self.solubility, T)
        K_G = self._evaluate(self.rate_constant, T)
        p = self.avrami_exponent

        excess = c_s + c_h - TSS_D  # hydrogen above TSS_D, precipitated or not

        # the advancement x is lagged to the previous time step (see docstring)
        c_s_n = self._previous(self.solute)
        c_h_n = self._previous(self.hydride)
        excess_n = c_s_n + c_h_n - TSS_D
        one_minus_x = ufl.max_value(self._ONE_MINUS_X_MIN, 1 - c_h_n / excess_n)
        jmak = p * one_minus_x * (-ufl.ln(one_minus_x)) ** (1 - 1 / p)

        present = ufl.gt(c_h_n, self.hydride_threshold)
        active = ufl.And(ufl.gt(c_s, TSS_D), present)
        return ufl.conditional(active, K_G * excess * jmak, 0)

    def _previous(self, species: Species) -> ufl.core.expr.Expr:
        """The concentration of a species at the previous time step."""
        if species.prev_solution is not None:
            return species.prev_solution
        # discontinuous case: one previous solution per subdomain
        return species.subdomain_to_prev_solution[self.volume]


def _check_temperature_function(name: str, value: float | Callable) -> float | Callable:
    """Check that a parameter is a float or a callable of the temperature only."""
    if isinstance(value, bool) or not isinstance(value, float | int | Callable):
        raise TypeError(
            f"{name} must be a float or a callable of T, not {type(value).__name__}"
        )
    if callable(value):
        code = value.__code__
        arguments = code.co_varnames[: code.co_argcount]
        if arguments != ("T",):
            raise ValueError(
                f"{name} must be a callable of the temperature T only, not of "
                f"{arguments}"
            )
    return value
