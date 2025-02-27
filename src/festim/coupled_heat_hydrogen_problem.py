import numpy as np
import tqdm.autonotebook
from dolfinx import fem

from festim.heat_transfer_problem import HeatTransferProblem
from festim.helpers import as_fenics_constant
from festim.hydrogen_transport_problem import (
    HTransportProblemDiscontinuous,
    HTransportProblemPenalty,
    HydrogenTransportProblem,
)
from festim.problem_change_of_var import HydrogenTransportProblemDiscontinuousChangeVar


def nmm_interpolate(f_out: fem.function, f_in: fem.function):
    """Non Matching Mesh Interpolate: interpolate one function (f_in) from one mesh into
    another function (f_out) with a mismatching mesh

    args:
        f_out: function to interpolate into
        f_in: function to interpolate from

    notes:
    https://fenicsproject.discourse.group/t/gjk-error-in-interpolation-between-non-matching-second-ordered-3d-meshes/16086/6
    """

    dim = f_out.function_space.mesh.topology.dim
    index_map = f_out.function_space.mesh.topology.index_map(dim)
    ncells = index_map.size_local + index_map.num_ghosts
    cells = np.arange(ncells, dtype=np.int32)
    interpolation_data = fem.create_interpolation_data(
        f_out.function_space, f_in.function_space, cells, padding=1e-11
    )
    f_out.interpolate_nonmatching(f_in, cells, interpolation_data=interpolation_data)


class CoupledHeatTransferHydrogenTransport:
    """
    Coupled heat transfer and hydrogen transport problem

    Args:
        heat_problem: the heat transfer problem
        hydrogen_problem: the hydrogen transport problem

    Attributes:
        heat_problem: the heat transfer problem
        hydrogen_problem: the hydrogen transport problem
        non_matching_meshes: True if the meshes in the heat_problem and hydorgen_problem
            are not matching

    Examples:
        .. highlight:: python
        .. code-block:: python

            import festim as F

            my_heat_transfer_model = F.HeatTransferProblem(
                mesh=F.Mesh(...),
                subdomains=[F.Subdomain(...)],
                ...
            )

            my_h_transport_model = F.HydrogenTransportProblem(
                mesh=F.Mesh(...),
                subdomains=[F.Subdomain(...)],
                species=[F.Species(name="H"), F.Species(name="Trap")],
                ...
            )

            coupled_problem = F.CoupledHeatTransferHydrogenTransport(
                heat_problem=my_heat_transfer_model,
                hydrogen_problem=my_h_transport_model,
            )


    """

    heat_problem: HeatTransferProblem
    hydrogen_problem: HydrogenTransportProblem

    non_matching_meshes: bool

    def __init__(
        self,
        heat_problem: HeatTransferProblem,
        hydrogen_problem: HydrogenTransportProblem,
    ) -> None:
        self.heat_problem = heat_problem
        self.hydrogen_problem = hydrogen_problem

    @property
    def heat_problem(self):
        return self._heat_problem

    @heat_problem.setter
    def heat_problem(self, value):
        if not isinstance(value, HeatTransferProblem):
            raise TypeError("heat_problem must be a festim.HeatTransferProblem object")
        self._heat_problem = value

    @property
    def hydrogen_problem(self):
        return self._hydrogen_problem

    @hydrogen_problem.setter
    def hydrogen_problem(self, value):
        if isinstance(
            value,
            HTransportProblemDiscontinuous
            | HTransportProblemPenalty
            | HydrogenTransportProblemDiscontinuousChangeVar,
        ):
            raise NotImplementedError(
                "Coupled heat transfer - hydorgen transport simulations with "
                "HydrogenTransportProblemDiscontinuousChangeVar, "
                "HTransportProblemPenalty or"
                "HydrogenTransportProblemDiscontinuousChangeVar, "
                "not currently supported"
            )
        elif not isinstance(value, HydrogenTransportProblem):
            raise TypeError(
                "hydrogen_problem must be a festim.HydrogenTransportProblem object"
            )
        self._hydrogen_problem = value

    @property
    def non_matching_meshes(self):
        return self.heat_problem.mesh.mesh != self.hydrogen_problem.mesh.mesh

    def initialise(self):
        if (
            self.heat_problem.settings.transient
            and self.hydrogen_problem.settings.transient
        ):
            # make sure both problems have the same initial time step and final time,
            # use minimal initial value of the two and maximal final time of the two
            min_initial_dt = min(
                self.heat_problem.settings.stepsize.initial_value,
                self.hydrogen_problem.settings.stepsize.initial_value,
            )
            self.heat_problem.settings.stepsize.initial_value = min_initial_dt
            self.hydrogen_problem.settings.stepsize.initial_value = min_initial_dt

            if (
                self.heat_problem.settings.final_time
                != self.hydrogen_problem.settings.final_time
            ):
                raise ValueError(
                    "Final time values in the heat transfer and hydrogen transport "
                    "model must be the same"
                )

        self.heat_problem.initialise()

        self.heat_problem.show_progress_bar = False

        if self.non_matching_meshes:
            V = fem.functionspace(self.hydrogen_problem.mesh.mesh, ("P", 1))
            T_func = fem.Function(V)

            nmm_interpolate(T_func, self.heat_problem.u)

            self.hydrogen_problem.temperature = T_func
        else:
            self.hydrogen_problem.temperature = self.heat_problem.u

        self.hydrogen_problem.initialise()

    def iterate(self):
        self.heat_problem.iterate()

        if self.non_matching_meshes:
            nmm_interpolate(
                self.hydrogen_problem.temperature_fenics, self.heat_problem.u
            )

        self.hydrogen_problem.iterate()

        # use the same time step for both problems, use minimum of the two
        next_dt_value = min(
            float(self.hydrogen_problem.dt), float(self.heat_problem.dt)
        )
        self.heat_problem.dt = as_fenics_constant(
            value=next_dt_value, mesh=self.heat_problem.mesh.mesh
        )
        self.hydrogen_problem.dt = as_fenics_constant(
            value=next_dt_value, mesh=self.hydrogen_problem.mesh.mesh
        )

    def run(self):
        if (
            self.heat_problem.settings.transient
            and self.hydrogen_problem.settings.transient
        ):
            if self.hydrogen_problem.show_progress_bar:
                self.hydrogen_problem.progress_bar = tqdm.autonotebook.tqdm(
                    desc=f"Solving {self.__class__.__name__}",
                    total=self.hydrogen_problem.settings.final_time,
                    unit_scale=True,
                )

            while (
                self.hydrogen_problem.t.value
                < self.hydrogen_problem.settings.final_time
            ):
                self.iterate()

            if self.hydrogen_problem.show_progress_bar:
                self.hydrogen_problem.progress_bar.refresh()
                self.hydrogen_problem.progress_bar.close()
        else:
            # Solve steady-state
            self.heat_problem.run()

            if self.non_matching_meshes:
                nmm_interpolate(
                    self.hydrogen_problem.temperature_fenics, self.heat_problem.u
                )

            self.hydrogen_problem.initialise()
            self.hydrogen_problem.run()
