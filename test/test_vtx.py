import festim as F
import dolfinx
from mpi4py import MPI
import numpy as np

import pytest

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
V = dolfinx.fem.FunctionSpace(mesh, ("Lagrange", 1))


def test_vtx_export_one_function(tmpdir):
    """Test can add one function to a vtx export"""
    u = dolfinx.fem.Function(V)
    sp = F.Species("H")
    sp.solution = u
    filename = str(tmpdir.join("my_export.bp"))
    my_export = F.VTXExport(filename, field=sp)
    my_export.define_writer(mesh.comm)

    for t in range(10):
        u.interpolate(lambda x: t * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2))

        my_export.write(t)


def test_vtx_export_two_functions(tmpdir):
    """Test can add two functions to a vtx export"""
    u = dolfinx.fem.Function(V)
    v = dolfinx.fem.Function(V)

    sp1 = F.Species("1")
    sp2 = F.Species("2")
    sp1.solution = u
    sp2.solution = v
    filename = str(tmpdir.join("my_export.bp"))
    my_export = F.VTXExport(filename, field=[sp1, sp2])

    my_export.define_writer(mesh.comm)

    for t in range(10):
        u.interpolate(lambda x: t * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2))
        v.interpolate(lambda x: t * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2))

        my_export.write(t)


def test_vtx_integration_with_h_transport_problem(tmpdir):
    my_model = F.HydrogenTransportProblem()
    my_model.mesh = F.Mesh1D(vertices=np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    my_mat = F.Material(D_0=1, E_D=0, name="mat")
    my_model.subdomains = [
        F.VolumeSubdomain1D(1, borders=[0.0, 4.0], material=my_mat),
        F.SurfaceSubdomain1D(1, x=0.0),
        F.SurfaceSubdomain1D(2, x=4.0),
    ]
    my_model.species = [F.Species("H")]
    my_model.temperature = 500

    filename = str(tmpdir.join("my_export.bp"))
    my_export = F.VTXExport(filename, field=my_model.species[0])
    my_model.exports = [my_export]

    my_model.initialise()

    for t in range(10):
        my_export.write(t)


def test_field_attribute_is_always_list():
    """Test that the field attribute is always a list"""
    my_export = F.VTXExport("my_export.bp", field=F.Species("H"))
    assert isinstance(my_export.field, list)

    my_export = F.VTXExport("my_export.bp", field=[F.Species("H")])
    assert isinstance(my_export.field, list)


@pytest.mark.parametrize("field", ["H", 1, [F.Species("H"), 1]])
def test_field_attribute_raises_error_when_invalid_type(field):
    """Test that the field attribute raises an error if the type is not festim.Species or list"""
    with pytest.raises(TypeError):
        F.VTXExport("my_export.bp", field=field)


def test_filename_raises_error_with_wrong_extension():
    """Test that the filename attribute raises an error if the extension is not .bp"""
    with pytest.raises(ValueError):
        F.VTXExport("my_export.txt", field=[F.Species("H")])


def test_filename_raises_error_when_wrong_type():
    """Test that the filename attribute raises an error if the extension is not .bp"""
    with pytest.raises(TypeError):
        F.VTXExport(1, field=[F.Species("H")])
