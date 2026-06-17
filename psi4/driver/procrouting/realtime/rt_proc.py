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


# Validation error, 90% of them, go find and use that to filer name
def run_rt_scf(name, **kwargs):
    ene, wfn = psi4.energy('scf', return_wfn=True)
    mints = core.MintsHelper(wfn.basisset())
    S = mints.ao_overlap()
    core.print_out("\n  Real time code here!!! \n\n")
    print("hello world!")
    core.print_out(S.np)

