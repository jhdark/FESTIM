import math

import ufl
from dolfinx import fem
from scifem import assemble_scalar

from festim.exports.surface_quantity import SurfaceQuantity
from festim.mesh import CoordinateSystem
from festim.species import Species
from festim.subdomain.surface_subdomain import SurfaceSubdomain


class SurfaceFlux(SurfaceQuantity):
    """Computes the flux of a field on a given surface.

    Args:
        field: species for which the surface flux is computed
        surface: surface subdomain
        filename: name of the file to which the surface flux is exported

    Attributes:
        see `festim.SurfaceQuantity`
    """

    field: Species
    surface: SurfaceSubdomain
    filename: str
    coordinate_system: CoordinateSystem

    title: str
    value: float
    data: list[float]

    @property
    def title(self):
        return f"{self.field.name} flux surface {self.surface.id}"

    def __init__(
        self, field: Species, surface: SurfaceSubdomain, filename: str | None = None
    ) -> None:
        super().__init__(field=field, surface=surface, filename=filename)

    def compute(
        self, u: fem.Function | ufl.indexed.Indexed, ds: ufl.Measure, entity_maps=None
    ):
        """Computes the value of the flux at the surface.

        Args:
            u: field for which the flux is computed
            ds: surface measure of the model
            entity_maps: entity maps relating parent mesh and submesh
        """

        # obtain mesh normal from integration domain
        mesh = ds.ufl_domain()
        n = ufl.FacetNormal(mesh)

        match self.coordinate_system:
            case CoordinateSystem.CARTESIAN:
                weight = 1
            case CoordinateSystem.CYLINDRICAL:
                r = ufl.SpatialCoordinate(mesh)[0]
                # TODO: full coverage assumed; expose as a constructor parameter
                # (e.g. azimuth_range) to support partial coverage
                coverage = 2 * math.pi  # radians
                weight = coverage * r
            case CoordinateSystem.SPHERICAL:
                r = ufl.SpatialCoordinate(mesh)[0]
                # TODO: full coverage assumed; expose as constructor parameters
                # (e.g. azimuth_range, polar_range) to support partial coverage
                coverage = 4 * math.pi  # steradians
                weight = coverage * r**2
            case _:
                raise NotImplementedError(
                    f"Unknown coordinate system {self.coordinate_system!s}"
                )

        self.value = assemble_scalar(
            fem.form(
                -weight * self.D * ufl.dot(ufl.grad(u), n) * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )
        self.data.append(self.value)
