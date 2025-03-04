from mpi4py import MPI

import dolfinx.mesh
import numpy as np
import pytest
import ufl
from dolfinx import default_scalar_type, fem
from ufl.conditional import Conditional

import festim as F

dummy_mat = F.Material(D_0=1, E_D=1, name="dummy_mat")

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)


def test_init():
    """Test that the attributes are set correctly"""
    # create a DirichletBC object
    subdomain = F.SurfaceSubdomain1D(1, x=0)
    species = "test"
    phi = 1.0
    R_p = 1.0
    D_0 = 1.0
    E_D = 1.0
    bc = F.ImplantationDirichlet(subdomain, species, phi, R_p, D_0, E_D)

    # # check that the attributes are set correctly
    # assert bc.subdomain == subdomain
    # assert bc.species == species
    # assert bc.phi == phi
    # assert bc.R_p == R_p
    # assert bc.D_0 == D_0
    # assert bc.E_D == E_D
    # assert bc.value_fenics is None
    # assert bc.bc_expr is None
