import numpy as np
import pytest
import ufl

import festim as F

volume = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=None)
HYDRIDE_CLASSES = [F.HydrideDissolution, F.HydrideNucleation, F.HydrideGrowth]


def make(cls, solubility=100.0, rate_constant=2.0, **kwargs):
    """A hydride reaction between fresh solute and hydride species."""
    solute = F.Species("C_SS")
    hydride = F.Species("C_prec", mobile=False)
    return cls(
        solute=solute,
        hydride=hydride,
        volume=volume,
        solubility=solubility,
        rate_constant=rate_constant,
        **kwargs,
    )


def growth_rate(T, c_s, c_h, c_s_n=None, c_h_n=None, **kwargs):
    """HydrideGrowth.rate evaluated to a float, with the previous-step
    concentrations defaulting to the current ones."""
    reaction = make(F.HydrideGrowth, **kwargs)
    reaction.solute.prev_solution = ufl.as_ufl(c_s if c_s_n is None else c_s_n)
    reaction.hydride.prev_solution = ufl.as_ufl(c_h if c_h_n is None else c_h_n)
    return float(reaction.rate(T, ufl.as_ufl(c_s), ufl.as_ufl(c_h)))


# ---------------------------------------------------------------- structure


@pytest.mark.parametrize(
    "cls, expected",
    [
        (F.HydrideDissolution, "C_prec --> C_SS"),
        (F.HydrideNucleation, "C_SS --> C_prec"),
        (F.HydrideGrowth, "C_SS --> C_prec"),
    ],
)
def test_direction(cls, expected):
    """Test that dissolution moves hydrogen from the hydride to the solute and
    nucleation/growth the other way."""
    assert str(make(cls)) == expected


@pytest.mark.parametrize("cls", HYDRIDE_CLASSES)
def test_arg_to_species_maps_solute_and_hydride(cls):
    """Test that the rate's species arguments are mapped to solute and hydride."""
    reaction = make(cls)
    assert reaction.arg_to_species == {
        "c_s": reaction.solute,
        "c_h": reaction.hydride,
    }


@pytest.mark.parametrize("cls", HYDRIDE_CLASSES)
def test_is_a_reaction_base(cls):
    """Test that the hydride reactions can be used wherever a ReactionBase can."""
    assert isinstance(make(cls), F.ReactionBase)


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("cls", HYDRIDE_CLASSES)
@pytest.mark.parametrize("attribute", ["solute", "hydride"])
def test_species_must_be_species(cls, attribute):
    """Test that solute and hydride reject non-Species values."""
    reaction = make(cls)
    with pytest.raises(TypeError, match=f"{attribute} must be a festim.Species"):
        setattr(reaction, attribute, "C_SS")


@pytest.mark.parametrize("cls", HYDRIDE_CLASSES)
@pytest.mark.parametrize("attribute", ["solubility", "rate_constant"])
@pytest.mark.parametrize("value", ["1.0", True, None])
def test_parameter_type_error(cls, attribute, value):
    """Test that solubility and rate_constant reject non-float, non-callable values."""
    reaction = make(cls)
    with pytest.raises(TypeError, match=f"{attribute} must be a float or a callable"):
        setattr(reaction, attribute, value)


@pytest.mark.parametrize("cls", HYDRIDE_CLASSES)
@pytest.mark.parametrize("attribute", ["solubility", "rate_constant"])
@pytest.mark.parametrize("value", [lambda x: 1.0, lambda T, t: 1.0, lambda: 1.0])
def test_parameter_must_depend_on_T_only(cls, attribute, value):
    """Test that a callable parameter must be a function of T alone."""
    reaction = make(cls)
    with pytest.raises(ValueError, match="callable of the temperature T only"):
        setattr(reaction, attribute, value)


@pytest.mark.parametrize("value", [2.5, 1, lambda T: 1.0])
def test_parameter_accepts_floats_and_callables_of_T(value):
    """Test that floats, ints and callables of T are accepted."""
    reaction = make(F.HydrideNucleation)
    reaction.solubility = value
    assert reaction.solubility is value


@pytest.mark.parametrize("value", ["2.5", True, None])
def test_avrami_exponent_type_error(value):
    """Test that avrami_exponent rejects non-numeric values."""
    with pytest.raises(TypeError, match="avrami_exponent must be a float"):
        make(F.HydrideGrowth, avrami_exponent=value)


@pytest.mark.parametrize("value", [0, -2.5])
def test_avrami_exponent_must_be_positive(value):
    """Test that avrami_exponent rejects non-positive values."""
    with pytest.raises(ValueError, match="avrami_exponent must be positive"):
        make(F.HydrideGrowth, avrami_exponent=value)


def test_base_rate_not_implemented():
    """Test that HydrideReaction.rate must be implemented by subclasses."""
    reaction = F.HydrideReaction(
        solute=F.Species("A"),
        hydride=F.Species("B", mobile=False),
        volume=volume,
        solubility=1.0,
        rate_constant=1.0,
        dissolves=True,
    )
    with pytest.raises(NotImplementedError):
        reaction.rate(300.0, 1.0, 1.0)


# ---------------------------------------------------------------- dissolution


@pytest.mark.parametrize(
    "c_s, c_h, expected",
    [
        (40.0, 250.0, 2.0 * (100.0 - 40.0)),  # undersaturated, plenty of hydride
        (95.0, 3.0, 2.0 * 3.0),  # capped by the hydride left
        (100.0, 250.0, 0.0),  # at the solubility
        (150.0, 250.0, 0.0),  # supersaturated
        (40.0, 0.0, 0.0),  # no hydride
        (40.0, -1.0, 0.0),  # negative hydride (Newton overshoot)
    ],
)
def test_dissolution_rate(c_s, c_h, expected):
    """Test R = K_D max(min(TSS_D - c_s, c_h), 0)."""
    reaction = make(F.HydrideDissolution)
    rate = reaction.rate(500.0, ufl.as_ufl(c_s), ufl.as_ufl(c_h))
    assert np.isclose(float(rate), expected)


def test_dissolution_evaluates_callables_of_T():
    """Test that solubility and rate constant callables are evaluated at T."""
    reaction = make(
        F.HydrideDissolution,
        solubility=lambda T: 0.2 * T,
        rate_constant=lambda T: 0.01 * T,
    )
    rate = reaction.rate(ufl.as_ufl(500.0), ufl.as_ufl(40.0), ufl.as_ufl(250.0))
    assert np.isclose(float(rate), 5.0 * (100.0 - 40.0))


# ---------------------------------------------------------------- nucleation


@pytest.mark.parametrize(
    "c_s, expected",
    [(540.0, 2.0 * (540.0 - 100.0)), (100.0, 0.0), (60.0, 0.0)],
)
def test_nucleation_rate(c_s, expected):
    """Test R = K_N (c_s - TSS_P) above TSS_P and 0 below."""
    reaction = make(F.HydrideNucleation)
    rate = reaction.rate(500.0, ufl.as_ufl(c_s), ufl.as_ufl(0.0))
    assert np.isclose(float(rate), expected)


# ---------------------------------------------------------------- growth


def jmak(c_s, c_h, TSS, K, p):
    """Reference JMAK growth rate (Passelaigue et al. 2021, Eq. 3)."""
    excess = c_s + c_h - TSS
    x = c_h / excess
    return K * excess * p * (1 - x) * (-np.log(1 - x)) ** (1 - 1 / p)


@pytest.mark.parametrize(
    "c_s, c_h, p",
    [(288.0, 1e-6, 2.5), (200.0, 88.0, 2.5), (150.0, 138.0, 2.5), (200.0, 88.0, 1.5)],
)
def test_growth_rate_matches_jmak(c_s, c_h, p):
    """Test that the growth rate is the JMAK rate of Eq. 3."""
    rate = growth_rate(500.0, c_s, c_h, avrami_exponent=p, solubility=144.0)
    assert np.isclose(rate, jmak(c_s, c_h, 144.0, 2.0, p))


@pytest.mark.parametrize(
    "c_s, c_h_n",
    [(100.0, 50.0), (144.0, 50.0), (200.0, 0.0), (200.0, -1.0)],
)
def test_growth_inactive(c_s, c_h_n):
    """Test that growth is off at or below TSS_D, or without existing hydride."""
    assert growth_rate(500.0, c_s, max(c_h_n, 0.0), c_h_n=c_h_n, solubility=144.0) == 0


def test_growth_advancement_uses_previous_step():
    """Test that x is taken from the previous-step concentrations, while the
    excess factor (c_tot - TSS_D) uses the current ones."""
    rate = growth_rate(500.0, 200.0, 88.0, c_s_n=230.0, c_h_n=58.0, solubility=144.0)
    x_n = 58.0 / (230.0 + 58.0 - 144.0)
    expected = 2.0 * (288.0 - 144.0) * 2.5 * (1 - x_n) * (-np.log(1 - x_n)) ** 0.6
    assert np.isclose(rate, expected)


def test_growth_rate_finite_at_full_precipitation():
    """Test that the rate stays finite when all excess hydrogen has precipitated
    (x = 1)."""
    assert np.isfinite(growth_rate(500.0, 144.0 + 1e-9, 144.0, solubility=144.0))


@pytest.mark.parametrize("value", ["1e-8", True, None])
def test_hydride_threshold_type_error(value):
    """Test that hydride_threshold rejects non-numeric values."""
    with pytest.raises(TypeError, match="hydride_threshold must be a float"):
        make(F.HydrideGrowth, hydride_threshold=value)


def test_hydride_threshold_must_be_non_negative():
    """Test that hydride_threshold rejects negative values."""
    with pytest.raises(ValueError, match="hydride_threshold must be non-negative"):
        make(F.HydrideGrowth, hydride_threshold=-1e-8)


@pytest.mark.parametrize(
    "c_h_n, expected_active", [(6e-13, False), (1e-8, False), (1e-6, True)]
)
def test_hydride_threshold_switches_growth(c_h_n, expected_active):
    """Test that growth is off at or below hydride_threshold (e.g. the residue of
    a complete dissolution) and on above it (e.g. a 1e-6 wt.ppm seed)."""
    rate = growth_rate(500.0, 288.0, c_h_n, solubility=144.0, hydride_threshold=1e-8)
    assert (rate > 0) == expected_active
