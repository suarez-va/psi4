import pytest
import psi4
from addons import using

pytestmark = [pytest.mark.psi, pytest.mark.api, pytest.mark.quick, *using("einsums")]


@pytest.fixture
def h2o_cwfn():
    mol = psi4.geometry("""
        O
        H 1 1.0
        H 1 1.0 2 101.5
        symmetry c1
    """)
    psi4.set_options({"basis": "sto-3g"})
    basis = psi4.core.BasisSet.build(mol, "ORBITAL", psi4.core.get_global_option("BASIS"))
    return psi4.core.ComplexWavefunction(mol, basis)


def test_complexoeprop_mulliken_not_implemented(h2o_cwfn):
    """ComplexOEProp is a skeleton: requesting Mulliken charges should raise Not Implemented."""
    oe = psi4.core.ComplexOEProp(h2o_cwfn)
    oe.set_title("SCF")
    oe.add("MULLIKEN_CHARGES")
    with pytest.raises(RuntimeError, match="Not Implemented!"):
        oe.compute()
