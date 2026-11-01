#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2025 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#

import os

import numpy as np

from psi4 import core

from ... import p4util
from ...p4util.exceptions import ValidationError

def run_rt_scf(wfn, **kwargs):
    from scipy.linalg import expm

    # Checking for spin reference
    verbose = core.get_option('SCF', "PRINT")
    reference = core.get_option('SCF', 'REFERENCE')
    if reference == "CGHF":
        print(reference)
        pass
    else:
        raise ValidationError("SCF: Unknown reference (%s)." % reference)

    S_ = wfn.S()
    X_ = wfn.get_X()
    Xinv_ = core.ComplexMatrix.doublet(X_, S_, True, False)

    Cao_ = wfn.C()
    Coao_ = core.ComplexMatrix.doublet(Xinv_, Cao_, False, False)
    Coao_ndt_ = Coao_.clone()
    Coao_pdt_ = Coao_.clone()

    Fao_ = wfn.F()
    Foao_ = core.ComplexMatrix.triplet(X_, Fao_, X_, True, False, False)

    # real-time propagation user options
    t0 = 0.0
    dt = float(kwargs.get("timestep", 1.0))
    tf = float(kwargs.get("totaltime", 10.0))
    Nprint = int(kwargs.get("printfreq", 1))
    Nsteps = round(tf / dt)

    # real-time propagation loop
    core.print_out("\n  ==> Real-Time <==\n\n")
    core.print_out("     Time         Total Energy\n\n")
    for i in range(Nsteps):
        # Current time
        t = t0 + i*dt

        # Printing observables at current time to output
        if np.mod(i, Nprint) == 0:
            ene = wfn.compute_E()
            core.print_out("%11.6f %20.14f\n" % (t, ene))
            #p4util.oeprop(wfn, 'MULLIKEN_CHARGES', title='SCF')

        U_p2dt_ = core.ComplexMatrix.expm(Foao_, -2j*dt)
        Coao_pdt_ = core.ComplexMatrix.doublet(U_p2dt_, Coao_ndt_, False, False)
        Coao_ndt_ = Coao_.clone()
        Coao_ = Coao_pdt_.clone()
        Cao_ = core.ComplexMatrix.doublet(X_, Coao_, False, False)
        wfn.set_C(Cao_)
        wfn.form_D()
        wfn.form_G()
        wfn.form_F()
        #Fao_ = wfn.F()
        Foao_ = core.ComplexMatrix.triplet(X_, Fao_, X_, True, False, False)

    core.print_out("  Time propagation completed.\n\n")


#Cao = X @ Coao; wfn.set_C(core.ComplexMatrix.from_array(Cao, name="MO coefficients"))
#def run_rt_scf_old(name, **kwargs):
#    import psi4
#    from scipy.linalg import expm
#    #core.print_out("\n  Start real-time code\n\n")
#    # SCF starting for initial condition starting point and access to wfn object
#    ene, wfn = psi4.energy('scf', return_wfn=True)
#
#    # Checking for spin reference
#    verbose = core.get_option('SCF', "PRINT")
#    reference = core.get_option('SCF', 'REFERENCE')
#    if reference == "CGHF":
#        print(reference)
#        pass
#    else:
#        raise ValidationError("SCF: Unknown reference (%s)." % reference)
#
#    wfn.form_Shalf()
#    wfn.form_H()
#    #enuc = wfn.get_energies("Nuclear")
#    #X_ = wfn.get_X(); Xinv = X_.to_array(); X = np.linalg.inv(Xinv)
#    X_ = wfn.get_X(); X = X_.to_array(); Xinv = np.linalg.inv(X)
#    C_ = wfn.C(); Cao = C_.to_array(); Coao = Xinv @ Cao; Coao_ndt = Coao.copy(); Coao_pdt = Coao.copy()
#    D_ = wfn.D(); Dao = D_.to_array()
#    H_ = wfn.H(); Hao = H_.to_array()
#    F_ = wfn.F(); Fao = F_.to_array(); Foao = X.conj().T @ Fao @ X
#
#    #nao = round(S.shape[0]/2); print([S[:nao,:nao],S[nao:,:nao]])
#    # real-time propagation user options
#    t0 = 0.0
#    dt = float(kwargs.get("timestep", 1.0))
#    tf = float(kwargs.get("totaltime", 10.0))
#    Nprint = int(kwargs.get("printfreq", 1))
#    Nsteps = round(tf / dt)
#
#    # real-time propagation loop
#    core.print_out("\n  ==> Real-Time <==\n\n")
#    core.print_out("     Time         Total Energy\n\n")
#    for i in range(Nsteps):
#        # Current time
#        t = t0 + i*dt
#
#        # Printing observables at current time to output
#        if np.mod(i, Nprint) == 0:
#            core.print_out("%11.6f %20.14f\n" % (t, ene))
#            ene = wfn.compute_E()
#            #ene = 0.5*np.einsum('ij,ji->', Hao + Fao, Dao) + enuc
#
#        U_p2dt = expm(-2j*Foao*dt)
#        Coao_pdt = U_p2dt @ Coao_ndt
#
#        Coao_ndt = Coao.copy()
#        Coao = Coao_pdt.copy()
#        Cao = X @ Coao
#        wfn.set_C(core.ComplexMatrix.from_array(Cao, name="MO coefficients"))
#        wfn.form_D()
#        wfn.form_G()
#        wfn.form_F()
#
#        C_ = wfn.C(); Cao = C_.to_array()
#        D_ = wfn.D(); Dao = D_.to_array()
#        F_ = wfn.F(); Fao = F_.to_array(); Foao = X.conj().T @ Fao @ X
#
#    core.print_out("  Time propagation completed.\n\n")


