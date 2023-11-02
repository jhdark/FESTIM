import festim as F
import numpy as np
import ufl
from dolfinx.mesh import meshtags
from dolfinx import fem
import pytest
import os


def surface_flux_export_compute():
    """Test that the surface flux export computes the correct value"""

    # BUILD
    L = 4.0
    D = 1.5
    my_mesh = F.Mesh1D(np.linspace(0, L, 10000))
    dummy_surface = F.SurfaceSubdomain1D(id=1, x=4)

    # define mesh ds measure
    facet_indices = np.array(
        dummy_surface.locate_boundary_facet_indices(my_mesh.mesh, 0),
        dtype=np.int32,
    )
    tags_facets = np.array(
        [1],
        dtype=np.int32,
    )
    facet_meshtags = meshtags(my_mesh.mesh, 0, facet_indices, tags_facets)
    ds = ufl.Measure("ds", domain=my_mesh.mesh, subdomain_data=facet_meshtags)

    # give function to species
    V = fem.FunctionSpace(my_mesh.mesh, ("CG", 1))
    u = fem.Function(V)
    u.interpolate(lambda x: 2 * x[0] ** 2 + 1)

    my_species = F.Species("H")
    my_species.solution = u

    my_export = F.SurfaceFlux(
        filename="my_surface_flux.csv",
        field=my_species,
        surface_subdomain=dummy_surface,
    )
    my_export.D = D

    # RUN
    my_export.compute(n=my_mesh.n, ds=ds)

    # TEST
    expected_value = -D * 4 * dummy_surface.x
    computed_value = my_export.value

    assert np.isclose(computed_value, expected_value, rtol=1e-2)


@pytest.mark.parametrize(
    "input, expected_value",
    [("export.csv", True), (None, False)],
)
def test_write_to_file_attribute(input, expected_value):
    """Test that the write_to_file attribute is correctly set when a filename is given"""
    my_export = F.SurfaceQuantity(
        filename=input,
        field="H",
        surface_subdomain=1,
    )

    assert my_export.write_to_file is expected_value


def test_title_generation(tmp_path):
    """Test that the title is made to be written to the header"""
    my_export = F.SurfaceFlux(
        filename=os.path.join(tmp_path, "my_export.csv"),
        field=F.Species("TEST"),
        surface_subdomain=F.SurfaceSubdomain1D(id=35, x=1),
    )
    assert my_export.title == "Flux surface 35: TEST"


def test_filename_setter_raises_TypeError(tmp_path):
    """Test that a TypeError is raised when the filename is not a string"""

    with pytest.raises(TypeError):
        F.SurfaceFlux(
            filename=os.path.join(tmp_path, 1),
            field=F.Species("test"),
            surface_subdomain=F.SurfaceSubdomain1D(id=1, x=0),
        )


def test_filename_setter_raises_ValueError(tmp_path):
    """Test that a ValueError is raised when the filename does not end with .csv"""

    with pytest.raises(ValueError):
        F.SurfaceFlux(
            filename=os.path.join(tmp_path, "my_export.xdmf"),
            field=F.Species("test"),
            surface_subdomain=F.SurfaceSubdomain1D(id=1, x=0),
        )


def test_writer(tmp_path):
    """Test that the writes values at each timestep"""
    my_export = F.SurfaceFlux(
        filename=os.path.join(tmp_path, "my_export.csv"),
        field=F.Species("test"),
        surface_subdomain=F.SurfaceSubdomain1D(id=1, x=0),
    )
    my_export.value = 2.0

    for i in range(10):
        my_export.write(i)

    # computed value should be range + 1 for the header
    computed_value = len(np.genfromtxt(my_export.filename, delimiter=","))

    expected_value = 11

    assert computed_value == expected_value
