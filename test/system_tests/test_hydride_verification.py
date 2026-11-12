"""Verification of the hydride reactions against the analytical solutions of the
HNGD verification cases (Passelaigue et al., J. Nucl. Mater. 544 (2021) 152683,
Section 3.1.1, Eqs. 10-12).

Each case runs a single mechanism on a uniform, flux-free 1D domain, so the
problem is effectively 0D. Concentrations are in wt.ppm. The time steps are the
largest giving < 1% error with backward Euler (see hydride_scripts/time_schemes).
"""

import numpy as np
import pytest

import festim as F

K_D = 1.11e3 * np.exp(-0.46 / (F.k_B * 550))
K_N = 2.75e-5 * np.exp(
    -(-0.566 + 4e-4 * 600 - 2e-7 * 600**2 + 3e-10 * 600**3) / (F.k_B * 600)
)
K_G, P = 1.71e-4, 2.5

# name: (class, T, solubility, rate constant, C_SS(0), C_prec(0), final time,
#        analytical C_SS(t))
CASES = {
    "dissolution": (
        F.HydrideDissolution,
        550,
        44.0,
        K_D,
        0.0,
        250.0,
        100.0,
        lambda t: 44.0 * (1 - np.exp(-K_D * t)),
    ),
    "nucleation": (
        F.HydrideNucleation,
        600,
        195.0,
        K_N,
        540.0,
        0.0,
        400.0,
        lambda t: 195.0 + (540.0 - 195.0) * np.exp(-K_N * t),
    ),
    "growth": (
        F.HydrideGrowth,
        650,
        144.0,
        K_G,
        288.0,
        1e-6,
        8000.0,
        lambda t: 144.0 + 144.0 * np.exp(-((K_G * t) ** P)),
    ),
}


def run(
    name: str, dt: float, problem_class: type = F.HydrogenTransportProblem
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one verification case with a fixed time step.

    Returns:
        The times and the C_SS and C_prec values at each step (uniform fields).
    """
    reaction_class, T, solubility, K, c_s_0, c_h_0, final_time, _ = CASES[name]
    model = problem_class()
    model.mesh = F.Mesh1D(vertices=np.linspace(0, 1, 3))
    volume = F.VolumeSubdomain1D(
        id=1, borders=[0, 1], material=F.Material(D_0=1, E_D=0)
    )
    model.subdomains = [volume]
    discontinuous = problem_class is F.HydrogenTransportProblemDiscontinuous
    subdomains = {"subdomains": [volume]} if discontinuous else {}
    solute = F.Species("C_SS", **subdomains)
    hydride = F.Species("C_prec", mobile=False, **subdomains)
    model.species = [solute, hydride]
    model.reactions = [
        reaction_class(
            solute=solute,
            hydride=hydride,
            volume=volume,
            solubility=solubility,
            rate_constant=K,
        )
    ]
    model.temperature = T
    model.initial_conditions = [
        F.InitialConcentration(species=solute, value=c_s_0, volume=volume),
        F.InitialConcentration(species=hydride, value=c_h_0, volume=volume),
    ]
    # the reaction residuals are small in wt.ppm units: rely on rtol only
    model.settings = F.Settings(
        atol=1e-15, rtol=1e-10, final_time=final_time, stepsize=dt
    )
    model.petsc_options = {"snes_stol": 0.0}
    model.show_progress_bar = False
    model.initialise()

    u = volume.u if discontinuous else model.u
    t, c_s, c_h = [], [], []
    while float(model.t) < final_time - 1e-9:
        model.iterate()
        t.append(float(model.t))
        c_s.append(u.sub(0).collapse().x.array.mean())
        c_h.append(u.sub(1).collapse().x.array.mean())
    return np.array(t), np.array(c_s), np.array(c_h)


def max_error(
    name: str, dt: float, problem_class: type = F.HydrogenTransportProblem
) -> float:
    """Max |C_SS - exact| / C_tot over the run."""
    t, c_s, _ = run(name, dt, problem_class)
    _, _, _, _, c_s_0, c_h_0, _, exact = CASES[name]
    return np.max(np.abs(c_s - exact(t))) / (c_s_0 + c_h_0)


@pytest.mark.parametrize(
    "problem_class",
    [F.HydrogenTransportProblem, F.HydrogenTransportProblemDiscontinuous],
)
@pytest.mark.parametrize(
    "name, dt", [("dissolution", 2.0), ("nucleation", 4.0), ("growth", 20.0)]
)
def test_hydride_matches_analytical(name, dt, problem_class):
    """Test that each HNGD mechanism reproduces its analytical solution to 1%, in
    both the continuous and the discontinuous problem."""
    assert max_error(name, dt, problem_class) < 0.01


@pytest.mark.parametrize(
    "name, dt", [("dissolution", 4.0), ("nucleation", 16.0), ("growth", 200.0)]
)
def test_hydride_converges_first_order(name, dt):
    """Test that halving dt roughly halves the error (backward Euler)."""
    order = np.log2(max_error(name, dt) / max_error(name, dt / 2))
    assert 0.6 < order < 1.3


@pytest.mark.parametrize("name", list(CASES))
def test_hydride_conserves_hydrogen(name):
    """Test that C_SS + C_prec is conserved to round-off."""
    _, _, _, _, c_s_0, c_h_0, _, _ = CASES[name]
    _, c_s, c_h = run(name, dt=50.0 if name == "growth" else 5.0)
    assert np.allclose(c_s + c_h, c_s_0 + c_h_0, rtol=1e-12, atol=0)
