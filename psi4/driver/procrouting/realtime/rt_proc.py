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

import psi4

from psi4 import core

from ...p4util.exceptions import ValidationError


def run_rt_scf(name, **kwargs):
    #core.print_out("\n  Start real-time code\n\n")
    # SCF starting for initial condition starting point and access to wfn object
    ene, wfn = psi4.energy('scf', return_wfn=True)

    # Checking for spin reference
    verbose = core.get_option('SCF', "PRINT")
    reference = core.get_option('SCF', 'REFERENCE')
    if reference == "RHF":
        pass
    else:
        raise ValidationError("SCF: Unknown reference (%s)." % reference)

    # Determining number of basis functions nbf, creating overlap matrix S,
    # diagonalizing to get evecs and evals for orthoganalization done later 
    nbf = wfn.basisset().nbf()
    mints = core.MintsHelper(wfn.basisset())
    S = mints.ao_overlap()
    S_evecs = core.Matrix("S_evecs", nbf, nbf)
    S_evals = core.Vector("S_evals", nbf)
    S.diagonalize(S_evecs, S_evals, core.DiagonalizeOrder.Descending)

    # Creating Lowdin orthogonalization matrices X and Xinv.
    # we need access to X and Xinv repeatedly during each propagation time-step.
    X = core.Matrix("X", nbf, nbf)
    Xinv = core.Matrix("Xinv", nbf, nbf)
    X.np[:] = (S_evecs.np * 1/np.sqrt(S_evals.np)) @ S_evecs.np.T
    Xinv.np[:] = (S_evecs.np * np.sqrt(S_evals.np)) @ S_evecs.np.T

    # Molecular orbital coefficients in the non-orthogonalized basis C_ao,
    # and the orthogonalized basis C_oao. SharedMatrix initially but will
    # need to made complex at some point for real-time propagation. 
    Ca_ao = wfn.Ca_subset("AO", "ALL")
    Cb_ao = wfn.Cb_subset("AO", "ALL")
    Ca_oao = Ca_ao.clone()
    Cb_oao = Cb_ao.clone()
    Ca_oao.gemm(False, False, 1.0, Xinv, Ca_ao, 0.0)
    Cb_oao.gemm(False, False, 1.0, Xinv, Cb_ao, 0.0)

    # Quick test to make sure orthoganalization matricies are correct
    #res = core.Matrix("res", nbf, nbf)
    #res.gemm(True, False, 1.0, Ca_oao, Ca_oao, 0.0)
    #print(res.np)

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
            core.print_out("%11.6f %20.14f\n" % (t, ene))

        # Here is where we need fock contraction over complex orbitals
        # Real-time propagation step from t to t + dt
        '''
        Equation of motion (orthogonal basis):
            dC(t)/dt = -i*F(t)*C(t)

        0. Approximate F(t+0.5*dt).

        1. Compute U(t+0.5*dt) from F(t+0.5*dt) using:
            U(t+0.5*dt) = exp(-i*dt*F(t+0.5*dt)).

        2. Compute C(t+dt) from U(t+0.5*dt) using:
            C(t+dt) = U(t+0.5*dt)C(t).

        3. Compute F(t+dt) from C(t+dt) (einsums), interpolate new F(t+0.5*dt) using:
            F(t+0.5*dt) = 0.5*(F(t+dt)+F(t)).

        4. Repeat steps 1-3 until convergence to some tolerance.
        '''
    core.print_out("  Time propagation completed.\n\n")


