import numpy as np
import pytest

import festim as F

test_mesh = F.Mesh1D(vertices=np.linspace(0, 1, 100))
test_mat = F.Material(D_0=1, E_D=0, thermal_conductivity=1, heat_capacity=1, density=1)
test_subdomains = [F.VolumeSubdomain1D(id=1, borders=[0, 1], material=test_mat)]
test_H = F.Species("H")


@pytest.mark.parametrize(
    "object",
    ["coucou", 1, 1.0],
)
def test_error_rasied_when_wrong_hydrogen_problem_given(object):
    """Test TypeError is raised when an object that isnt a
    festim.HydrogenTransportProblem object is given to hydrogen_problem"""

    test_heat_problem = F.HeatTransferProblem()

    with pytest.raises(
        TypeError,
        match="hydrogen_problem must be a festim.HydrogenTransportProblem object",
    ):
        F.CoupledHeatTransferHydrogenTransport(
            heat_problem=test_heat_problem, hydrogen_problem=object
        )


@pytest.mark.parametrize(
    "object",
    ["coucou", 1, 1.0],
)
def test_error_rasied_when_wrong_heat_problem_given(object):
    """Test TypeError is raised when an object that isnt a
    festim.HeatTransferProblem object is given to heat_problem"""

    test_hydrogen_problem = F.HydrogenTransportProblem()

    with pytest.raises(
        TypeError,
        match="heat_problem must be a festim.HeatTransferProblem object",
    ):
        F.CoupledHeatTransferHydrogenTransport(
            heat_problem=object, hydrogen_problem=test_hydrogen_problem
        )


def test_initial_dt_values_are_the_same():
    test_heat_problem = F.HeatTransferProblem(
        mesh=test_mesh,
        subdomains=test_subdomains,
        settings=F.Settings(atol=1, rtol=1, transient=True, stepsize=0.5, final_time=5),
    )

    test_hydrogen_problem = F.HydrogenTransportProblem(
        mesh=test_mesh,
        subdomains=test_subdomains,
        species=[test_H],
        settings=F.Settings(atol=1, rtol=1, transient=True, stepsize=1.5, final_time=5),
    )

    test_coupled_problem = F.CoupledHeatTransferHydrogenTransport(
        heat_problem=test_heat_problem,
        hydrogen_problem=test_hydrogen_problem,
    )

    test_coupled_problem.initialise()

    assert np.isclose(
        float(test_coupled_problem.heat_problem.dt),
        float(test_coupled_problem.hydrogen_problem.dt),
    )


def test_final_time_values_are_the_same():
    test_heat_problem = F.HeatTransferProblem(
        mesh=test_mesh,
        subdomains=test_subdomains,
        settings=F.Settings(atol=1, rtol=1, transient=True, stepsize=1, final_time=5),
    )

    test_hydrogen_problem = F.HydrogenTransportProblem(
        mesh=test_mesh,
        subdomains=test_subdomains,
        species=[test_H],
        settings=F.Settings(atol=1, rtol=1, transient=True, stepsize=1, final_time=10),
    )

    test_coupled_problem = F.CoupledHeatTransferHydrogenTransport(
        heat_problem=test_heat_problem,
        hydrogen_problem=test_hydrogen_problem,
    )

    test_coupled_problem.initialise()

    assert np.isclose(
        float(test_coupled_problem.heat_problem.settings.final_time),
        float(test_coupled_problem.hydrogen_problem.settings.final_time),
    )


def test_error_is_raised_when_meshes_not_the_same():
    alt_mesh = F.Mesh1D(vertices=np.linspace(0, 2, 20))

    test_heat_problem = F.HeatTransferProblem(
        mesh=test_mesh,
        subdomains=test_subdomains,
        settings=F.Settings(atol=1, rtol=1, transient=True, stepsize=1, final_time=1),
    )

    test_hydrogen_problem = F.HydrogenTransportProblem(
        mesh=alt_mesh,
        subdomains=test_subdomains,
        species=[test_H],
        settings=F.Settings(atol=1, rtol=1, transient=True, stepsize=1, final_time=1),
    )

    test_coupled_problem = F.CoupledHeatTransferHydrogenTransport(
        heat_problem=test_heat_problem,
        hydrogen_problem=test_hydrogen_problem,
    )

    with pytest.raises(
        ValueError,
        match="The meshes of the heat transfer and hydrogen transport problems must be the same",
    ):
        test_coupled_problem.initialise()
