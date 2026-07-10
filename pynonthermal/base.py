#!/usr/bin/env python3
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pynonthermal.constants import EV
from pynonthermal.constants import H
from pynonthermal.constants import ME
from pynonthermal.constants import QE
from numba import njit

DATADIR = Path(__file__).absolute().parent / "data"

@njit(fastmath=True,cache=True)
def jit_electronlossfunction(energy_ev: float, n_e_cgs: float) -> float:
    if energy_ev <= 0:
        return 0.0
    
    energy = energy_ev * EV
    omegap = 5.64e4 * math.sqrt(n_e_cgs)
    zetae = H * omegap / 2.0 / math.pi

    if energy_ev > 14.0:
        if 2.0 * energy <= zetae:
            return 0.0
        lossfunc = n_e_cgs * 2.0 * math.pi * (QE**4) / energy * math.log(2.0 * energy / zetae)
    else:
        v = math.sqrt(2.0 * energy / ME)
        eulergamma = 0.577215664901532
        val_inside_log = (ME * (v**3)) / (eulergamma * (QE**2) * omegap)
        if val_inside_log <= 0:
            return 0.0
        lossfunc = n_e_cgs * 2.0 * math.pi * (QE**4) / energy * math.log(val_inside_log)

    return lossfunc / EV


def electronlossfunction(energy_ev: float, n_e_cgs: float) -> float:
    # free-electron plasma loss rate (as in Kozma & Fransson 1992)
    # - dE / dX [eV / cm]
    # returns a positive number

    # return math.log(energy_ev) / energy_ev
    n_e = n_e_cgs
    energy = energy_ev * EV  # convert eV to erg

    # omegap = math.sqrt(4 * math.pi * n_e_cgs * pow(QE, 2) / ME)
    omegap = 5.64e4 * math.sqrt(n_e_cgs)
    zetae = H * omegap / 2 / math.pi

    if energy_ev > 14:
        assert 2 * energy > zetae
        lossfunc = n_e * 2 * math.pi * QE**4 / energy * math.log(2 * energy / zetae)
    else:
        v = math.sqrt(2 * energy / ME)  # velocity in m/s
        eulergamma = 0.577215664901532
        lossfunc = n_e * 2 * math.pi * QE**4 / energy * math.log(ME * pow(v, 3) / (eulergamma * pow(QE, 2) * omegap))

    # lossfunc is now [erg / cm]
    return lossfunc / EV  # return as [eV / cm]


def get_n_tot(ions: Sequence[tuple[int, int]], ionpopdict: dict[tuple[int, int], float]) -> float:
    # total number density of all nuclei [cm^-3]
    n_tot = 0.0
    for Z, ion_stage in ions:
        n_tot += ionpopdict[(Z, ion_stage)]
    return n_tot


def get_Zbar(ions: Sequence[tuple[int, int]], ionpopdict: dict[tuple[int, int], float]) -> float:
    # number density-weighted average atomic number
    # i.e. protons per nucleus
    Zbar = 0.0
    n_tot = get_n_tot(ions, ionpopdict)
    for Z, ion_stage in ions:
        n_ion = ionpopdict[(Z, ion_stage)]
        Zbar += Z * n_ion / n_tot

    return Zbar


def get_Zboundbar(ions: Sequence[tuple[int, int]], ionpopdict: dict[tuple[int, int], float]) -> float:
    # number density-weighted average number of bound electrons per nucleus
    Zboundbar = 0.0
    n_tot = get_n_tot(ions, ionpopdict)
    for Z, ion_stage in ions:
        n_ion = ionpopdict[(Z, ion_stage)]
        Zboundbar += (Z - ion_stage + 1) * n_ion / n_tot

    return Zboundbar


def get_energyindex_lteq(en_ev: float, engrid: npt.NDArray[np.float64]) -> int:
    # find energy bin lower boundary is less than or equal to search value
    # assert en_ev >= engrid[0]
    deltaen = engrid[1] - engrid[0]
    # assert en_ev < (engrid[-1] + deltaen)

    index = math.floor((en_ev - engrid[0]) / deltaen)

    return 0 if index < 0 else min(index, len(engrid) - 1)


def get_energyindex_gteq(en_ev: float, engrid: npt.NDArray[np.float64]) -> int:
    # find energy bin lower boundary is greater than or equal to search value
    deltaen = engrid[1] - engrid[0]

    index = math.ceil((en_ev - engrid[0]) / deltaen)

    return 0 if index < 0 else min(index, len(engrid) - 1)
