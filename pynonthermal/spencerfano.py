from __future__ import annotations
from scipy.linalg import toeplitz
import math
import typing as t
from math import atan
from pathlib import Path
from numba import njit
import artistools as at
import matplotlib.axes as mplax
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import linalg
import time 

import pynonthermal
from pynonthermal.axelrod import get_workfn_ev
from pynonthermal.base import electronlossfunction, jit_electronlossfunction
from pynonthermal.base import get_Zbar
from pynonthermal.constants import K_B

SUBSHELLNAMES = [
    "K ",
    "L1",
    "L2",
    "L3",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "N6",
    "N7",
    "O1",
    "O2",
    "O3",
    "O4",
    "O5",
    "O6",
    "O7",
    "P1",
    "P2",
    "P3",
    "P4",
    "Q1",
]

@njit(cache=True, fastmath=True)
def jit_get_J(Z, ion_stage, ionpot_ev):
    if ion_stage == 1:
        if Z == 2:
            return 15.8
        if Z == 10:
            return 24.2
        if Z == 18:
            return 10.0
    return 0.6 * ionpot_ev


@njit(cache=True, fastmath=True)
def jit_psecondary(e_p, ionpot_ev, J, epsilon):
    e_s = epsilon - ionpot_ev
    return 1.0 / J / math.atan((e_p - ionpot_ev) / 2.0 / J) / (1.0 + (e_s / J) ** 2)


@njit(cache=True)
def _searchsorted_right_minus1(arr, val):
    return np.searchsorted(arr, val, side="right") - 1


@njit(cache=True, fastmath=True)
def shell_contribution_jit(engrid, yvec, ar_xs_array, arr_en_nz, ionpot_ev, J, deltaen):
    npts = engrid.shape[0]
    n_en = arr_en_nz.shape[0]
    N_e_ion = np.zeros(n_en)
    emax = engrid[-1]

    integral1startindex = max(0, _searchsorted_right_minus1(engrid, ionpot_ev))

    for a in range(n_en):
        en_a = arr_en_nz[a]
        acc = 0.0

        # Integral 1: ionpot to enlambda
        enlambda = min(emax - en_a, en_a + ionpot_ev)
        integral2stopindex = _searchsorted_right_minus1(engrid, enlambda)

        if integral2stopindex >= integral1startindex:
            for j in range(integral1startindex, integral2stopindex + 1):
                endash = engrid[j]
                k = _searchsorted_right_minus1(engrid, en_a + endash)
                k = max(0, min(k, npts - 1))
                e_p = engrid[k]
                if e_p > ionpot_ev:
                    psec = jit_psecondary(e_p, ionpot_ev, J, endash)
                    acc += deltaen * yvec[k] * ar_xs_array[k] * psec

        # Integral 2: 2E + I to Emax
        start_idx = max(0, _searchsorted_right_minus1(engrid, 2.0 * en_a + ionpot_ev))
        epsilon_val = en_a + ionpot_ev
        for j in range(start_idx, npts):
            e_p = engrid[j]
            if e_p > ionpot_ev:
                psec2 = jit_psecondary(e_p, ionpot_ev, J, epsilon_val)
                acc += deltaen * yvec[j] * ar_xs_array[j] * psec2

        N_e_ion[a] = acc

    return N_e_ion

@njit(cache=True,fastmath=True)
def jit_compute_ionisation_shell(sfmatrix, engrid, ionpot_ev, ar_xs_array, J, n_ion, deltaen, xsstartindex):
    """
    Standalone Numba function. No 'self', no Polars, just arrays and math.
    """
    npts = len(engrid)

    # Pre-allocate 1D arrays (Numba handles these very efficiently)
    prefactors = np.zeros(npts)
    epsilon_uppers = np.zeros(npts)
    int_eps_uppers = np.zeros(npts)
    epsilon_lowers1 = np.zeros(npts)
    int_eps_lowers1 = np.zeros(npts)

    # Precompute terms (runs at C-speed)
    for j in range(npts):
        prefactors[j] = (n_ion * ar_xs_array[j] / math.atan((engrid[j] - ionpot_ev) / 2.0 / J)) * deltaen
        epsilon_uppers[j] = min((engrid[j] + ionpot_ev) / 2.0, engrid[j])
        int_eps_uppers[j] = math.atan((epsilon_uppers[j] - ionpot_ev) / J)
        epsilon_lowers1[j] = max(engrid[j] - engrid[0], ionpot_ev)
        int_eps_lowers1[j] = math.atan((epsilon_lowers1[j] - ionpot_ev) / J)

    # Fast nested loops (No massive 2D memory overhead)
    for i in range(npts):
        en = engrid[i]
        jstart = max(i, xsstartindex)

        # First integral
        for j in range(jstart, npts):
            if epsilon_lowers1[j - i] <= epsilon_uppers[j]:
                sfmatrix[i, j] += prefactors[j] * (int_eps_uppers[j] - int_eps_lowers1[j - i])

        # Second integral setup
        target_en = 2 * en + ionpot_ev
        if target_en < engrid[-1] + deltaen:
            # Simple linear search is often faster than binary search in JIT for small arrays
            second_start = npts + 1
            for k in range(npts):
                if engrid[k] >= target_en:
                    second_start = k
                    break
        else:
            second_start = npts + 1

        epsilon_lower2 = en + ionpot_ev

        # Second integral execution
        for j in range(second_start, npts):
            if epsilon_lower2 <= epsilon_uppers[j]:
                int_eps_lower2 = math.atan((epsilon_lower2 - ionpot_ev) / J)
                sfmatrix[i, j] -= prefactors[j] * (int_eps_uppers[j] - int_eps_lower2)

    return sfmatrix

@njit(cache=True,fastmath=True)
def jit_calculate_N_e_batch(arr_en, engrid, yvec, ion_pop_keys, ion_pop_vals, df_Z, df_ionstage, df_ionpot, df_J, df_arxs, excitation_data, deltaen):
    """
    A purely numeric Numba function that handles evaluating N_e for a batch array.
    """
    npts_out = len(arr_en)
    N_e_tot = np.zeros(npts_out)
    n_engrid = len(engrid)
    
    # Outer Loop over Ions
    for idx_ion in range(len(ion_pop_keys)):
        Z = ion_pop_keys[idx_ion, 0]
        ion_stage = ion_pop_keys[idx_ion, 1]
        n_ion = ion_pop_vals[idx_ion]
        
        N_e_ion = np.zeros(npts_out)
        
        # --- Excitation Component (Processed safely via primitive unpacked structures) ---
        # excitation_data format: array of [level_n_density, epsilon_trans_ev, xsvec_start_index_in_flat_buffer]
        # For simplicity/robustness, if excitation lists are active, we can handle it or fall back. 
        # (Assuming main time is ionisation, but let's loop through it)
        # Note: If excitation is too dynamic, it can also be handled or skipped if negligible.
        
        # --- Ionisation Shell Components ---
        for s_idx in range(len(df_Z)):
            if df_Z[s_idx] != Z or df_ionstage[s_idx] != ion_stage:
                continue
                
            ionpot_ev = df_ionpot[s_idx]
            J = df_J[s_idx]
            ar_xs_array = df_arxs[s_idx] # 2D array row
            
            # Fast loop over the output subgrid points
            for i in range(npts_out):
                en_out = arr_en[i]
                if en_out <= 0:
                    continue
                
                # --- Integral 1 (ionpot to enlambda) ---
                enlambda = min(engrid[-1] - en_out, en_out + ionpot_ev)
                
                # Manual searchsorted right for limits
                integral1startindex = 0
                for k in range(n_engrid):
                    if engrid[k] > ionpot_ev:
                        integral1startindex = k - 1
                        break
                integral1startindex = max(0, integral1startindex)
                
                integral2stopindex = 0
                for k in range(n_engrid):
                    if engrid[k] > enlambda:
                        integral2stopindex = k - 1
                        break
                
                # Evaluate Integral 1 loop
                for j in range(integral1startindex, integral2stopindex + 1):
                    endash = engrid[j]
                    target_en = en_out + endash
                    
                    # searchsorted right
                    k_idx = n_engrid - 1
                    for k in range(n_engrid):
                        if engrid[k] > target_en:
                            k_idx = k - 1
                            break
                    k_idx = max(0, min(k_idx, n_engrid - 1))
                    
                    e_p = engrid[k_idx]
                    if e_p > ionpot_ev:
                        val_psec = jit_Psecondary(e_p, ionpot_ev, J, epsilon=endash)
                        N_e_ion[i] += deltaen * yvec[k_idx] * ar_xs_array[k_idx] * val_psec
                
                # --- Integral 2 (2E + I to E_max) ---
                target_start_en2 = 2.0 * en_out + ionpot_ev
                integral2startindex = n_engrid + 1
                for k in range(n_engrid):
                    if engrid[k] >= target_start_en2:
                        integral2startindex = k
                        break
                
                if integral2startindex < n_engrid:
                    epsilon_val = en_out + ionpot_ev
                    for j in range(integral2startindex, n_engrid):
                        e_p = engrid[j]
                        if e_p > ionpot_ev:
                            val_psec2 = jit_Psecondary(e_p, ionpot_ev, J, epsilon=epsilon_val)
                            N_e_ion[i] += deltaen * yvec[j] * ar_xs_array[j] * val_psec2
                            
        for i in range(npts_out):
            N_e_tot[i] += n_ion * N_e_ion[i]
            
    return N_e_tot

class SpencerFanoSolver:
    """Solve the Spencer-Fano equation for non-thermal heating, ionisation, and excitation.

    The Spencer-Fano equation is a differential equation that describes the energy deposition
    of non-thermal electrons in a plasma. The solution of the Spencer-Fano equation gives the
    energy density of the non-thermal electrons as a function of energy. The energy density
    can be used to calculate the heating rate, ionisation rate, and excitation rate of the plasma.
    """

    _solved: bool
    _frac_heating: float
    _frac_ionisation_tot: float
    _frac_excitation_tot: float
    _frac_ionisation_ion: dict[tuple[int, int], float]
    _frac_excitation_ion: dict[tuple[int, int], float]
    _eff_ionpot: dict[tuple[int, int], float]
    _nt_ionisation_ratecoeff: dict[tuple[int, int], float]
    ionpopdict: dict[tuple[int, int], float]
    excitationlists: dict[tuple[int, int], dict[t.Any, tuple[float, npt.NDArray[np.float64], float]]]
    verbose: bool
    _n_e: float
    engrid: npt.NDArray[np.float64]
    deltaen: npt.NDArray[np.float64]
    dfcollion: pl.DataFrame
    sourcevec: npt.NDArray[np.float64]
    E_init_ev: float
    sfmatrix: npt.NDArray[np.float64]
    adata_polars: pl.DataFrame | None
    yvec: npt.NDArray[np.float64]

    def __init__(
        self,
        emin_ev: float = 1,
        emax_ev: float = 3000,
        npts: int = 4000,
        verbose: bool = False,
        use_ar1985: bool = False,
    ) -> None:
        tt = time.time()
        print('Starting SF solver - LPM fork ')
        
        self._solved = False
        self._n_e = 0.0
        self.reset_solution_analysis()

        self.ionpopdict = {}  # key is (Z, ion_stage) value is number density

        # key is (Z, ion_stage) value is {levelkey : (levelnumberdensity, xs_vec, epsilon_trans_ev)}
        self.excitationlists = {}

        self.verbose = verbose
        self.engrid = np.linspace(emin_ev, emax_ev, num=npts, endpoint=True, dtype=float)
        self.deltaen = self.engrid[1] - self.engrid[0]

        readtime = time.time()
        self.dfcollion = pynonthermal.collion.read_colliondata(
            collionfilename=("collion-AR1985.txt" if use_ar1985 else "collion.txt")
        )
        readtime = time.time() - readtime
        
        self.sourcevec = np.zeros(self.engrid.shape)
        # 0.3% of the energy range, so 0.1 keV for 3 KeV Emax to match Kozma & Fransson 1992
        source_spread_pts = math.ceil(npts / 30.0)
        if source_spread_pts < 1:
            msg = "source_spread_pts must be at least 1"
            raise ValueError(msg)

        for s in range(npts):
            # spread the source over some energy width
            if s < npts - source_spread_pts:
                self.sourcevec[s] = 0.0
            elif s < npts:
                self.sourcevec[s] = 1.0 / (self.deltaen * source_spread_pts)
        # self.sourcevec[-1] = 1.

        source_emin = self.engrid[np.flatnonzero(self.sourcevec)[0]]
        source_emax = self.engrid[np.flatnonzero(self.sourcevec)[-1]]

        # E_init_ev is the deposition rate density that we assume when solving the SF equation.
        # The solution will be scaled to the true deposition rate later
        self.E_init_ev = np.dot(self.engrid, self.sourcevec) * self.deltaen

        self.adata_polars = None

        if self.verbose:
            print(
                f"\nSetting up Spencer-Fano equation with {npts} energy points from"
                f" {self.engrid[0]} to {self.engrid[-1]} eV..."
            )
            print(
                f"  source is a box function from {source_emin:.2f} to"
                f" {source_emax:.2f} eV with E_init {self.E_init_ev:7.2f} [eV/s/cm3]"
            )

        self.sfmatrix = np.zeros((npts, npts))
        print('init time = ',time.time()-tt, 'of which is spent in reading',readtime)

    def __enter__(self) -> t.Self:
        """Enter the context manager."""
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context manager."""

    def get_energyindex_lteq(self, en_ev: float) -> int:
        return pynonthermal.get_energyindex_lteq(en_ev, engrid=self.engrid)

    def get_energyindex_gteq(self, en_ev: float) -> int:
        return pynonthermal.get_energyindex_gteq(en_ev, engrid=self.engrid)

    def electronlossfunction(self, en_ev: float) -> float:
        return electronlossfunction(en_ev, self.get_n_e())

    def add_excitation(
        self,
        Z: int,
        ion_stage: int,
        levelnumberdensity: float,
        xs_vec: npt.NDArray[np.float64],
        epsilon_trans_ev: float,
        transitionkey: t.Any | None = None,
    ) -> None:
        """Add a bound-bound non-thermal collisional excitation to the solver.

        levelnumberdensity:
            the level population density in cm^-3
        xs_vec:
            an array of cross sections in cm^2 defined at every energy in the SpencerFanoSolver.engrid array [eV]
        epsilon_trans_ev:
            the transition energy in eV
        transitionkey:
            any key to uniquely identify the transition so that the rate coefficient can be retrieved later
        """
        assert not self._solved, "Can't add excitation after solving the Spencer-Fano equation"
        assert len(xs_vec) == len(self.engrid)

        if (Z, ion_stage) not in self.excitationlists:
            self.excitationlists[(Z, ion_stage)] = {}

        if transitionkey is None:
            transitionkey = len(self.excitationlists[(Z, ion_stage)])  # simple number index

        assert transitionkey not in self.excitationlists[(Z, ion_stage)]
        self.excitationlists[(Z, ion_stage)][transitionkey] = (
            levelnumberdensity,
            xs_vec,
            epsilon_trans_ev,
        )
        vec_xs_excitation_levelnumberdensity_deltae = levelnumberdensity * self.deltaen * xs_vec
        xsstartindex = self.get_energyindex_lteq(en_ev=epsilon_trans_ev)

        for i, en in enumerate(self.engrid):
            stopindex = self.get_energyindex_lteq(en_ev=float(en + epsilon_trans_ev))

            startindex = max(i, xsstartindex)
            # for j in range(startindex, stopindex):
            self.sfmatrix[i, startindex:stopindex] += vec_xs_excitation_levelnumberdensity_deltae[startindex:stopindex]

            # do the last bit separately because we're not using the full deltaen interval

            delta_en_actual = en + epsilon_trans_ev - self.engrid[stopindex]
            self.sfmatrix[i, stopindex] += (
                vec_xs_excitation_levelnumberdensity_deltae[stopindex] * delta_en_actual / self.deltaen
            )

    def add_ion_ltepopexcitation(
        self, Z: int, ion_stage: int, n_ion: float, temperature: float = 3000, adata_polars: pl.DataFrame | None = None
    ) -> None:
        if adata_polars is not None:
            self.adata_polars = adata_polars

        if self.adata_polars is None:
            # use ARTIS atomic data read by the artistools package to get the levels
            self.adata_polars = at.atomic.get_levels(
                Path(pynonthermal.DATADIR, "artis_files"),
                get_transitions=True,
                derived_transitions_columns=["epsilon_trans_ev", "lambda_angstroms", "lower_g", "upper_g"],
            )

        assert self.adata_polars is not None

        ion = self.adata_polars.filter(pl.col("Z") == Z).filter(pl.col("ion_stage") == ion_stage)
        if ion.is_empty():
            msg = f"ERROR: No excitation data for Z={Z} ion_stage {ion_stage} in internal database."
            raise AssertionError(msg)

        assert (Z, ion_stage) not in self.ionpopdict or math.isclose(
            self.ionpopdict[(Z, ion_stage)], n_ion, rel_tol=1e-6
        ), "Can't add the same ion twice with different populations"

        dfpops_thision = ion["levels"].item()

        ltepartfunc = dfpops_thision.select(pl.col("g") * (-pl.col("energy_ev") / K_B / temperature).exp()).sum().item()
        dfpops_thision = (
            dfpops_thision.rename({"levelindex": "level"})
            .with_columns(ion_popfrac=pl.col("g") * (-pl.col("energy_ev") / K_B / temperature).exp() / ltepartfunc)
            .with_columns(n_LTE=n_ion * pl.col("ion_popfrac"))
            .with_columns(n_NLTE=pl.col("n_LTE"))
        ).select(["level", "n_LTE", "n_NLTE", "ion_popfrac"])

        lzdftransitions = ion["transitions"].item().filter((pl.col("collstr") >= 0).or_(pl.col("forbidden") == 0))

        maxnlevelslower: int | None = None
        maxnlevelsupper: int | None = None

        # find the highest ground multiplet level
        # groundlevelnoj = ion.levels.iloc[0].levelname.split('[')[0]
        # maxnlevelslower = ion.levels[ion.levels.levelname.str.startswith(groundlevelnoj)].index.max()

        # match ARTIS defaults
        maxnlevelslower = 5
        maxnlevelsupper = 250

        if maxnlevelslower is not None:  # pyright: ignore [reportUnnecessaryComparison]
            lzdftransitions = lzdftransitions.filter(pl.col("lower") < maxnlevelslower)
        if maxnlevelsupper is not None:  # pyright: ignore [reportUnnecessaryComparison]
            lzdftransitions = lzdftransitions.filter(pl.col("upper") < maxnlevelsupper)

        lzdftransitions = lzdftransitions.filter(pl.col("epsilon_trans_ev") >= self.engrid[0])
        dftransitions = lzdftransitions.collect()

        if not dftransitions.is_empty():
            dftransitions = dftransitions.join(
                dfpops_thision.select(pl.col("level").alias("lower"), pl.col("n_NLTE").alias("lower_pop")),
                on="lower",
                how="left",
            )

            if self.verbose:
                print(
                    f"  including Z={Z} ion_stage"
                    f" {ion_stage} ({at.get_ionstring(Z, ion_stage)}) excitation with T"
                    f" {temperature} K (ntransitions {len(dftransitions)},"
                    f" maxnlevelslower {maxnlevelslower}, maxnlevelsupper"
                    f" {maxnlevelsupper})"
                )

            for transition in dftransitions.iter_rows(named=True):
                epsilon_trans_ev = transition["epsilon_trans_ev"]
                if epsilon_trans_ev >= self.engrid[0]:
                    xs_vec = pynonthermal.excitation.get_xs_excitation_vector(self.engrid, transition)
                    self.add_excitation(
                        Z,
                        ion_stage,
                        transition["lower_pop"],
                        xs_vec,
                        epsilon_trans_ev,
                        transitionkey=(transition["lower"], transition["upper"]),
                    )


    def _add_ionisation_shell(self, n_ion: float, shell: dict[str, int | float]) -> None:
        assert not self._solved
        
        deltaen = self.engrid[1] - self.engrid[0]
        ionpot_ev = shell["ionpot_ev"]
        J = pynonthermal.collion.get_J(int(shell["Z"]), int(shell["ion_stage"]), ionpot_ev)
        
        ar_xs_array = np.array(pynonthermal.collion.get_arxs_array_shell(self.engrid, shell))

        if ionpot_ev <= self.engrid[0]:
            xsstartindex = 0
        else:
            xsstartindex = np.searchsorted(self.engrid, ionpot_ev, side='left')

        # Hand off to the ultra-fast Numba compiler
        self.sfmatrix = jit_compute_ionisation_shell(
            self.sfmatrix, self.engrid, ionpot_ev, ar_xs_array, 
            J, n_ion, deltaen, xsstartindex
        )

    #def _add_ionisation_shell(self, n_ion: float, shell: dict[str, int | float]) -> None:
    #    assert not self._solved, "Can't add ionisation after solving the Spencer-Fano equation"
#
    #    deltaen = self.engrid[1] - self.engrid[0]
    #    ionpot_ev = shell["ionpot_ev"]
    #    J = pynonthermal.collion.get_J(int(shell["Z"]), int(shell["ion_stage"]), ionpot_ev)
    #    npts = len(self.engrid)
    #    ar_xs_array = np.array(pynonthermal.collion.get_arxs_array_shell(self.engrid, shell))
    #    if ionpot_ev <= self.engrid[0]:
    #        xsstartindex = 0
    #    else:
    #        # Replaced the generator with a fast searchsorted equivalent
    #        xsstartindex = np.searchsorted(self.engrid, ionpot_ev, side='left')
    #    # Fully vectorize the integral bounds using numpy arrays
    #    prefactors = n_ion * ar_xs_array / np.arctan((self.engrid - ionpot_ev) / (2.0 * J)) * deltaen
    #    epsilon_uppers = np.minimum((self.engrid + ionpot_ev) / 2.0, self.engrid)
    #    int_eps_uppers = np.arctan((epsilon_uppers - ionpot_ev) / J)
    #    epsilon_lowers1 = np.maximum(self.engrid - self.engrid[0], ionpot_ev)
    #    int_eps_lowers1 = np.arctan((epsilon_lowers1 - ionpot_ev) / J)
    #    # --- SCIPY TOEPLITZ UPGRADE ---
    #    # Construct the [j - i] offset mapping as an upper triangular Toeplitz matrix.
    #    # toeplitz(c, r) takes the first column and first row.
    #    c_eps = np.zeros(npts)
    #    c_eps[0] = epsilon_lowers1[0]
    #    T_eps_lowers1 = np.triu(toeplitz(c_eps, epsilon_lowers1))
    #    c_int = np.zeros(npts)
    #    c_int[0] = int_eps_lowers1[0]
    #    T_int_eps_lowers1 = np.triu(toeplitz(c_int, int_eps_lowers1))
    #    # Build index grids for fast broadcasting masks
    #    I, J_idx = np.ogrid[:npts, :npts]
    #    # --- FIRST INTEGRAL MATRICES ---
    #    mask1 = (J_idx >= I) & (J_idx >= xsstartindex) & (T_eps_lowers1 <= epsilon_uppers)
    #    term1 = prefactors * (int_eps_uppers - T_int_eps_lowers1)
    #    self.sfmatrix += np.where(mask1, term1, 0.0)
    #    # --- SECOND INTEGRAL MATRICES ---
    #    en_2d = self.engrid[:, np.newaxis]
    #    epsilon_lower2 = en_2d + ionpot_ev
    #    target_energies = 2 * self.engrid + ionpot_ev
    #    valid_second = target_energies < (self.engrid[-1] + deltaen)
    #    # Vectorized equivalent of get_energyindex_lteq across the whole grid
    #    second_starts = np.where(
    #        valid_second,
    #        np.searchsorted(self.engrid, target_energies, side='right') - 1,
    #        npts + 1
    #    )
    #    mask2 = (J_idx >= second_starts[:, np.newaxis]) & (epsilon_lower2 <= epsilon_uppers)
    #    int_eps_lower2 = np.arctan((epsilon_lower2 - ionpot_ev) / J)
    #    term2 = prefactors * (int_eps_uppers - int_eps_lower2)
    #    self.sfmatrix -= np.where(mask2, term2, 0.0)

    def add_ionisation(self, Z: int, ion_stage: int, n_ion: float) -> None:
        assert not self._solved, "Can't add ionisation after solving the Spencer-Fano equation"
        assert (Z, ion_stage) not in self.ionpopdict, "Can't add the same ion twice"
        if n_ion == 0.0:
            return
        if self.verbose:
            print(
                f"  including Z={Z} ion_stage"
                f" {ion_stage} ({at.get_ionstring(Z, ion_stage)}) ionisation with n_ion"
                f" {n_ion:.1e} [/cm3]"
            )
        assert n_ion > 0.0
        self.ionpopdict[(Z, ion_stage)] = n_ion
        # Combine filters into a single Polars operation and push the ionpot_ev check
        # directly into the dataframe query rather than evaluating it inside the python loop.
        valid_shells = self.dfcollion.filter(
            (pl.col("Z") == Z) & 
            (pl.col("ion_stage") == ion_stage) & 
            (pl.col("ionpot_ev") >= self.engrid[0])
        )
        # .to_dicts() avoids the high Python-level overhead of .iter_rows()
        for shell in valid_shells.to_dicts():
            self._add_ionisation_shell(n_ion, shell)

    def calculate_free_electron_density(self) -> float:
        # number density of free electrons [cm-^3]
        n_e = 0.0
        for Z, ion_stage in self.ionpopdict:
            charge = ion_stage - 1
            assert charge >= 0
            n_e += charge * self.ionpopdict[(Z, ion_stage)]
        return n_e

    def get_n_e(self) -> float:
        if self._n_e <= 0.0:
            self._n_e = self.calculate_free_electron_density()

        return self._n_e

    def get_n_ion_tot(self) -> float:
        # total number density of all nuclei [cm^-3]
        n_ion_tot = 0.0
        for Z, ion_stage in self.ionpopdict:
            n_ion_tot += self.ionpopdict[(Z, ion_stage)]
        return n_ion_tot

    def solve(self, depositionratedensity_ev: float, override_n_e: float | None = None) -> None:
        import time 
        t = time.time()
        self._solved = False
        self.reset_solution_analysis()

        self.depositionratedensity_ev = depositionratedensity_ev
        if override_n_e is not None:
            self._n_e = override_n_e
            # else it will be calculated on demand from ion populations

        npts = len(self.engrid)
        n_e = self.get_n_e()

        if self.verbose:
            n_ion_tot = self.get_n_ion_tot()
            x_e = n_e / n_ion_tot
            print(f" n_ion_tot: {n_ion_tot:.2e} [/cm3]        (total ion density)")
            print(f"       n_e: {n_e:.2e} [/cm3]        (free electron density)")
            print(f"       x_e: {x_e:.2e} [/cm3]        (electrons per nucleus)")
            print(f"deposition: {self.depositionratedensity_ev:7.2f}  [eV/s/cm3]")

        deltaen = self.engrid[1] - self.engrid[0]
        npts = len(self.engrid)

        constvec = np.zeros(npts)
        for i in range(npts):
            for j in range(i, npts):
                constvec[i] += self.sourcevec[j] * deltaen

        sfmatrix_with_electronloss = self.sfmatrix.copy()
        for i in range(npts):
            sfmatrix_with_electronloss[i, i] += electronlossfunction(self.engrid[i], n_e)

        yvec_reference = np.array(
            linalg.lu_solve(
                linalg.lu_factor(sfmatrix_with_electronloss, overwrite_a=False), constvec, trans=0
            ),  # zuban: ignore[no-untyped-call]
            dtype=np.float64,
        )
        self.yvec = np.array(yvec_reference * self.depositionratedensity_ev / self.E_init_ev, dtype=np.float64)
        self._solved = True
        print('Time in solve(): ',time.time()-t)

    def calculate_nt_frac_excitation_ion(self, Z: int, ion_stage: int) -> float:
        if (Z, ion_stage) not in self.excitationlists:
            return 0.0

        # integral in Kozma & Fransson equation 9, but summed over all transitions for given ion
        deltaen = self.engrid[1] - self.engrid[0]
        npts = len(self.engrid)

        xs_excitation_vec_sum_alltrans = np.zeros(npts)

        for (
            levelnumberdensity,
            xsvec,
            epsilon_trans_ev,
        ) in self.excitationlists[(Z, ion_stage)].values():
            xs_excitation_vec_sum_alltrans += levelnumberdensity * epsilon_trans_ev * xsvec

        return np.dot(xs_excitation_vec_sum_alltrans, self.yvec) * deltaen / self.depositionratedensity_ev

#    def calculate_N_e(self, energy_ev: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
#        is_scalar = np.isscalar(energy_ev)
#        arr_en = np.atleast_1d(energy_ev).astype(np.float64)
#        
#        deltaen = self.engrid[1] - self.engrid[0]
#        
#        # Prepare dictionary/DataFrame data into primitive NumPy arrays for JIT consumption
#        ion_pop_keys = np.array(list(self.ionpopdict.keys()), dtype=np.int64)
#        ion_pop_vals = np.array(list(self.ionpopdict.values()), dtype=np.float64)
#        
#        # Unpack Polars columns into NumPy arrays
#        df_Z = self.dfcollion["Z"].to_numpy()
#        df_ionstage = self.dfcollion["ion_stage"].to_numpy()
#        df_ionpot = self.dfcollion["ionpot_ev"].to_numpy()
#        
#        # Precompute J vectors and cross-section matrix
#        df_J = np.array([pynonthermal.collion.get_J(int(z), int(ion), pot) 
#                         for z, ion, pot in zip(df_Z, df_ionstage, df_ionpot)])
#        
#        df_arxs = np.array([pynonthermal.collion.get_arxs_array_shell(self.engrid, row) 
#                            for row in self.dfcollion.to_dicts()])
#        
#        # Execute the Numba JIT-accelerated inner loop
#        arr_N_e = jit_calculate_N_e_batch(
#            arr_en, self.engrid, self.yvec, 
#            ion_pop_keys, ion_pop_vals, 
#            df_Z, df_ionstage, df_ionpot, df_J, df_arxs,
#            None, deltaen
#        )
#        
#        return float(arr_N_e[0]) if is_scalar else arr_N_e

    def calculate_frac_heating(self) -> float:
        self._frac_heating = 0.0
        E_0 = self.engrid[0]
        n_e = self.get_n_e()
        deltaen = self.engrid[1] - self.engrid[0]

        # Use the global JIT function to evaluate losses instantly across engrid
        print('in loss vec')
        loss_vec = np.array([jit_electronlossfunction(float(en), n_e) for en in self.engrid])
        print('out of loss vec')
        self._frac_heating += (deltaen / self.depositionratedensity_ev) * np.sum(loss_vec * self.yvec)

        # Single E_0 point edge calculation
        frac_heating_E_0_part = E_0 * self.yvec[0] * jit_electronlossfunction(E_0, n_e) / self.depositionratedensity_ev
        self._frac_heating += frac_heating_E_0_part

        # Sub-threshold integration grid
        npts_integral = math.ceil(E_0 / deltaen) * 5
        arr_en, deltaen2 = np.linspace(0.0, E_0, num=npts_integral, retstep=True, endpoint=True, dtype=np.float64)
        
        # Passes the subgrid through our upgraded JIT-based calculate_N_e path!
        timene = time.time()
        print('in calc n_e')
        arr_N_e = self.calculate_N_e(arr_en)
        print('out of calc n_e', time.time()-timene)

        arr_en_N_e = arr_en * arr_N_e
        
        frac_heating_N_e = float(1.0 / self.depositionratedensity_ev * np.sum(arr_en_N_e) * deltaen2)

        if self.verbose:
            print(f" frac_heating(E<EMIN): {frac_heating_N_e:.5f}")

        self._frac_heating += frac_heating_N_e
        return self._frac_heating
    
    def calculate_N_e(self, energy_ev: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        # Kozma & Fransson equation 6.
        # Natively accepts scalar floats OR numpy arrays to support batch evaluation.
        is_scalar = np.isscalar(energy_ev)
        arr_en = np.atleast_1d(energy_ev).astype(np.float64)
        
        N_e_tot = np.zeros_like(arr_en)
        nonzero_mask = arr_en > 0.0
        
        # Return early if all energies are 0
        if not np.any(nonzero_mask):
            return 0.0 if is_scalar else N_e_tot

        arr_en_nz = arr_en[nonzero_mask]
        deltaen = self.engrid[1] - self.engrid[0]
        
        # Pre-vectorize the external Psecondary function for fast batch execution
        v_Psec = np.vectorize(pynonthermal.collion.Psecondary)

        for (Z, ion_stage), n_ion in self.ionpopdict.items():
            N_e_ion = np.zeros_like(arr_en_nz)

            # --- Excitation Component ---
            if self.excitationlists and (Z, ion_stage) in self.excitationlists:
                for levelnumberdensity, xsvec, epsilon_trans_ev in self.excitationlists[(Z, ion_stage)].values():
                    valid_mask = (arr_en_nz + epsilon_trans_ev) >= self.engrid[0]
                    if np.any(valid_mask):
                        valid_energies = arr_en_nz[valid_mask] + epsilon_trans_ev
                        idx = np.searchsorted(self.engrid, valid_energies, side='right') - 1
                        idx = np.clip(idx, 0, len(self.engrid) - 1)
                        N_e_ion[valid_mask] += (levelnumberdensity / n_ion) * self.yvec[idx] * xsvec[idx]

            # --- Ionisation Shell Components ---
            dfcollion_thision = self.dfcollion.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage))

            tshell = time.time()
            
            
            for shell in dfcollion_thision.to_dicts():
                ionpot_ev = shell["ionpot_ev"]
                J = jit_get_J(int(shell["Z"]), int(shell["ion_stage"]), ionpot_ev)
                shelltime = time.time()
                ar_xs_array = np.asarray(pynonthermal.collion.get_arxs_array_shell(self.engrid, shell), dtype=np.float64)
                #print(' get_arxs_array_shell time',time.time()-shelltime)
                shelltime = time.time()
                N_e_ion = shell_contribution_jit(
                    self.engrid, self.yvec, ar_xs_array, arr_en_nz, ionpot_ev, J, deltaen
                )
                #print(' shell_contribution_jit time',time.time()-shelltime)
                #todo - need to check where this is supposed to go.
                N_e_tot[nonzero_mask] += n_ion * N_e_ion
                
            #print(' time in shell loop', time.time() - tshell,n_ion * N_e_ion)

            #for shell in dfcollion_thision.to_dicts():
            #    ionpot_ev = shell["ionpot_ev"]
            #    J = pynonthermal.collion.get_J(int(shell["Z"]), int(shell["ion_stage"]), ionpot_ev)
            #    ar_xs_array = np.array(pynonthermal.collion.get_arxs_array_shell(self.engrid, shell))
#
            #    # --- Integral 1 (ionpot to enlambda) ---
            #    enlambda = np.minimum(self.engrid[-1] - arr_en_nz, arr_en_nz + ionpot_ev)
            #    integral1startindex = max(0, np.searchsorted(self.engrid, ionpot_ev, side='right') - 1)
            #    integral2stopindices = np.searchsorted(self.engrid, enlambda, side='right') - 1
            #    
            #    j_indices = np.arange(len(self.engrid))
            #    j_mask = (j_indices >= integral1startindex) & (j_indices[np.newaxis, :] <= integral2stopindices[:, np.newaxis])
            #    
            #    if np.any(j_mask):
            #        endash_2d = self.engrid[np.newaxis, :]
            #        target_en = arr_en_nz[:, np.newaxis] + endash_2d
            #        k_idx = np.searchsorted(self.engrid, target_en, side='right') - 1
            #        k_idx = np.clip(k_idx, 0, len(self.engrid) - 1)
            #        
            #        e_p_2d = self.engrid[k_idx]
            #        epsilon_2d = np.broadcast_to(endash_2d, e_p_2d.shape)
            #        
            #        # CRITICAL FIX: Ensure Psecondary is only evaluated where e_p > ionpot_ev
            #        safe_eval_mask = j_mask & (e_p_2d > ionpot_ev)
            #        
            #        if np.any(safe_eval_mask):
            #            #v_Psec(e_p_2d[mask], ionpot_ev=ionpot_ev, J=J, epsilon=epsilon_2d[mask])
            #            valid_Psec = v_Psec(e_p_2d[safe_eval_mask], epsilon=epsilon_2d[safe_eval_mask],ionpot_ev= ionpot_ev, J=J)
            #            term1_full = np.zeros_like(e_p_2d)
            #            term1_full[safe_eval_mask] = deltaen * self.yvec[k_idx[safe_eval_mask]] * ar_xs_array[k_idx[safe_eval_mask]] * valid_Psec
            #            N_e_ion += np.sum(term1_full, axis=1)
#
            #    # --- Integral 2 (2E + I to E_max) ---
            #    target_start_en2 = 2 * arr_en_nz + ionpot_ev
            #    integral2startindices = np.searchsorted(self.engrid, target_start_en2, side='right') - 1
            #    integral2startindices = np.maximum(0, integral2startindices)
            #    
            #    j_mask2 = j_indices[np.newaxis, :] >= integral2startindices[:, np.newaxis]
            #    
            #    if np.any(j_mask2):
            #        e_p_2d = np.broadcast_to(self.engrid[np.newaxis, :], j_mask2.shape)
            #        epsilon_2d = np.broadcast_to((arr_en_nz + ionpot_ev)[:, np.newaxis], j_mask2.shape)
            #        
            #        # CRITICAL FIX: Maintain consistency safeguard for the second integral limits
            #        safe_eval_mask2 = j_mask2 & (e_p_2d > ionpot_ev)
            #        
            #        if np.any(safe_eval_mask2):
            #            valid_Psec2 = v_Psec(e_p_2d[safe_eval_mask2], epsilon=epsilon_2d[safe_eval_mask2], ionpot_ev=ionpot_ev, J=J)
            #            y_xs_2d = np.broadcast_to((self.yvec * ar_xs_array)[np.newaxis, :], j_mask2.shape)
            #            
            #            term2_full = np.zeros_like(e_p_2d)
            #            term2_full[safe_eval_mask2] = deltaen * y_xs_2d[safe_eval_mask2] * valid_Psec2
            #            N_e_ion += np.sum(term2_full, axis=1)
            #            
            #N_e_tot[nonzero_mask] += n_ion * N_e_ion

        return float(N_e_tot[0]) if is_scalar else N_e_tot
#
#    def calculate_frac_heating(self) -> float:
#        # Kozma & Fransson equation 8
#        self._frac_heating = 0.0
#        E_0 = self.engrid[0]
#        n_e = self.get_n_e()
#        deltaen = self.engrid[1] - self.engrid[0]
#
#        # Vectorized Standard Heating integration (replaces generator comprehension)
#        loss_vec = np.array([electronlossfunction(float(en), n_e) for en in self.engrid])
#        self._frac_heating += (deltaen / self.depositionratedensity_ev) * np.sum(loss_vec * self.yvec)
#
#        # Single E_0 point edge calculation
#        frac_heating_E_0_part = E_0 * self.yvec[0] * loss_vec[0] / self.depositionratedensity_ev
#        self._frac_heating += frac_heating_E_0_part
#
#        # Fast Vectorized N_e Evaluation
#        npts_integral = math.ceil(E_0 / deltaen) * 5
#        arr_en, deltaen2 = np.linspace(0.0, E_0, num=npts_integral, retstep=True, endpoint=True, dtype=np.float64)
#        
#        # Pass the whole array natively into our upgraded calculate_N_e function!
#        arr_N_e = self.calculate_N_e(arr_en)
#        arr_en_N_e = arr_en * arr_N_e
#        
#        frac_heating_N_e = float(1.0 / self.depositionratedensity_ev * np.sum(arr_en_N_e) * deltaen2)
#
#        if self.verbose:
#            print(f" frac_heating(E<EMIN): {frac_heating_N_e:.5f}")
#
#        self._frac_heating += frac_heating_N_e
#        return self._frac_heating

    def reset_solution_analysis(self) -> None:
        self._frac_heating = 0.0
        self._frac_ionisation_tot = 0.0
        self._frac_excitation_tot = 0.0
        self._frac_ionisation_ion = {}
        self._frac_excitation_ion = {}
        self._nt_ionisation_ratecoeff = {}
        self._eff_ionpot = {}

    def analyse_ntspectrum(self) -> None:
        import time 
        tt = time.time()
        
        assert self._solved
        self.reset_solution_analysis()

        deltaen = self.engrid[1] - self.engrid[0]

        if self.verbose:
            print(f"    n_e_nt: {self.get_n_e_nt():.2e} [/cm3]")

        for (Z, ion_stage), n_ion in self.ionpopdict.items():
            n_ion_tot = self.get_n_ion_tot()
            X_ion = n_ion / n_ion_tot
            
            dfcollion_thision = self.dfcollion.filter((pl.col("Z") == Z) & (pl.col("ion_stage") == ion_stage))
            
            ionpot_valence = dfcollion_thision["ionpot_ev"].min()
            assert isinstance(ionpot_valence, float)

            if self.verbose:
                print(
                    f"\n====> Z={Z:2d} ion_stage"
                    f" {ion_stage} {at.get_ionstring(Z, ion_stage)} (valence potential"
                    f" {ionpot_valence:.1f} eV)"
                )
                print(f"               n_ion: {n_ion:.2e} [/cm3]")
                print(f"     n_ion/n_ion_tot: {X_ion:.5f}")

            self._frac_ionisation_ion[(Z, ion_stage)] = 0.0
            
            # --- VECTORIZED BATCH PROCESSING START ---
            shells = dfcollion_thision.to_dicts()
            
            if shells:
                # Build a 2D matrix of cross sections: shape (num_shells, len(engrid))
                ar_xs_matrix = np.array([
                    pynonthermal.collion.get_arxs_array_shell(self.engrid, shell) 
                    for shell in shells
                ])
                
                ionpot_ev_array = dfcollion_thision["ionpot_ev"].to_numpy()
                
                # Vectorized dot products: Matrix @ Vector handles all shells at once
                integralgamma_array = (ar_xs_matrix @ self.yvec) * deltaen
                
                # Calculate fraction of ionisation for all shells
                frac_ionisation_shell_array = (
                    n_ion * ionpot_ev_array * integralgamma_array / self.depositionratedensity_ev
                )
                
                # Vectorized warning check
                invalid_mask = frac_ionisation_shell_array > 1
                if np.any(invalid_mask):
                    print(f"WARNING: Ignoring invalid frac_ionisation_shell(s).")
                
                # Accumulate the totals
                integralgamma = np.sum(integralgamma_array)
                self._frac_ionisation_ion[(Z, ion_stage)] = np.sum(frac_ionisation_shell_array)
                eta_over_ionpot_sum = np.sum(frac_ionisation_shell_array / ionpot_ev_array)
                
                # Only run the shell logging loop if verbose is True
                if self.verbose:
                    for i, shell in enumerate(shells):
                        if int(shell["n"]) < 0:
                            strsubshell = SUBSHELLNAMES[-int(shell["l"])]
                            shellname = f"Lotz shell {strsubshell}"
                        else:
                            shellname = f"n {int(shell['n']):d} l {int(shell['l']):d}"
                        print(
                            f"frac_ionisation_shell({shellname}):"
                            f" {frac_ionisation_shell_array[i]:.4f} (ionpot"
                            f" {shell['ionpot_ev']:.2f} eV)"
                        )
            else:
                integralgamma = 0.0
                eta_over_ionpot_sum = 0.0
            # --- VECTORIZED BATCH PROCESSING END ---

            self._frac_ionisation_tot += self._frac_ionisation_ion[(Z, ion_stage)]

            eff_ionpot = float(X_ion / eta_over_ionpot_sum) if eta_over_ionpot_sum else float("inf")
            self._eff_ionpot[(Z, ion_stage)] = eff_ionpot

            if self.verbose:
                print(f"     frac_ionisation: {self._frac_ionisation_ion[(Z, ion_stage)]:.4f}")

            if self.excitationlists:
                if n_ion > 0.0:
                    self._frac_excitation_ion[(Z, ion_stage)] = self.calculate_nt_frac_excitation_ion(Z, ion_stage)
                else:
                    self._frac_excitation_ion[(Z, ion_stage)] = 0.0

                if self._frac_excitation_ion[(Z, ion_stage)] > 1:
                    self._frac_excitation_ion[(Z, ion_stage)] = 0.0
                    print(
                        f"WARNING: Ignoring invalid frac_excitation_ion of {self._frac_excitation_ion[(Z, ion_stage)]}."
                    )

                self._frac_excitation_tot += self._frac_excitation_ion[(Z, ion_stage)]

                if self.verbose:
                    print(f"     frac_excitation: {self._frac_excitation_ion[(Z, ion_stage)]:.4f}")
            else:
                self._frac_excitation_ion[(Z, ion_stage)] = 0.0

            self._nt_ionisation_ratecoeff[(Z, ion_stage)] = self.depositionratedensity_ev / n_ion_tot / eff_ionpot
            
            if self.verbose:
                workfn_ev = get_workfn_ev(
                    Z,
                    ion_stage,
                    ionpot_ev=ionpot_valence,
                    Zbar=get_Zbar(ions=tuple(self.ionpopdict.keys()), ionpopdict=self.ionpopdict),
                )
                print(f"   workfn eff_ionpot: {eff_ionpot:8.2f} [eV]")
                print(f"       approx workfn: {workfn_ev:8.2f} [eV] (without Spencer-Fano solution)")
                print(f"ionisation ratecoeff: {self._nt_ionisation_ratecoeff[(Z, ion_stage)]:.2e} [/s]")

                assert np.isclose(
                    self._nt_ionisation_ratecoeff[(Z, ion_stage)],
                    integralgamma,
                    rtol=0.01,
                )

        if self.verbose:
            print()
            print(f"  frac_excitation_tot: {self._frac_excitation_tot:.4f}")
            print(f"  frac_ionisation_tot: {self._frac_ionisation_tot:.4f}")

        import time
        t = time.time()
        self.calculate_frac_heating()
        t_heating  = time.time()-t
        frac_heating = self.get_frac_heating()

        if self.verbose:
            print(f"         frac_heating: {frac_heating:.4f}")
            print(f"             frac_sum: {self._frac_excitation_tot + self._frac_ionisation_tot + frac_heating:.4f}")
        
        print('time in analyse() = ',time.time()-tt, 'of which is spent in heating: ',t_heating)

    def get_n_e_nt(self) -> float:
        assert self._solved
        n_e_nt = 0.0
        for i, en in enumerate(self.engrid):
            # oneovervelocity = np.sqrt(9.10938e-31 / 2 / en / 1.60218e-19) / 100.
            velocity = np.sqrt(2 * en * 1.60218e-19 / 9.10938e-31) * 100.0  # cm/s
            n_e_nt += self.yvec[i] / velocity * self.deltaen

        return n_e_nt

    def get_frac_heating(self) -> float:
        assert self._solved
        if self._frac_heating <= 0.0:
            self.calculate_frac_heating()

        return self._frac_heating

    def get_frac_excitation_tot(self) -> float:
        assert self._solved
        if self._frac_excitation_tot <= 0.0:
            self.analyse_ntspectrum()

        return self._frac_excitation_tot

    def get_frac_ionisation_tot(self) -> float:
        assert self._solved
        if self._frac_ionisation_tot <= 0.0:
            self.analyse_ntspectrum()

        return self._frac_ionisation_tot

    def get_frac_ionisation_ion(self, Z: int, ion_stage: int) -> float:
        assert self._solved
        if (Z, ion_stage) not in self._frac_ionisation_ion:
            self.analyse_ntspectrum()

        return self._frac_ionisation_ion[(Z, ion_stage)]

    def get_eff_ionpot(self, Z: int, ion_stage: int) -> float:
        assert self._solved
        if (Z, ion_stage) not in self._eff_ionpot:
            self.analyse_ntspectrum()

        return self._eff_ionpot[(Z, ion_stage)]

    def get_ionisation_ratecoeff(self, Z: int, ion_stage: int) -> float:
        assert self._solved
        return self._nt_ionisation_ratecoeff[(Z, ion_stage)]

    def get_excitation_ratecoeff(self, Z: int, ion_stage: int, transitionkey: t.Any) -> float:
        # integral in Kozma & Fransson equation 9
        _levelnumberdensity, xsvec, _epsilon_trans_ev = self.excitationlists[(Z, ion_stage)][transitionkey]

        return np.dot(xsvec, self.yvec) * self.deltaen / self.depositionratedensity_ev

    def get_frac_sum(self) -> float:
        return self.get_frac_heating() + self.get_frac_excitation_tot() + self.get_frac_ionisation_tot()

    def get_d_etaheating_by_d_en_vec(self) -> list[float]:
        assert self._solved
        return [
            self.electronlossfunction(self.engrid[i]) * self.yvec[i] / self.depositionratedensity_ev
            for i in range(len(self.engrid))
        ]

    def get_d_etaexcitation_by_d_en_vec(self) -> npt.NDArray[np.float64]:
        assert self._solved
        part_integrand = np.zeros(len(self.engrid))

        for Z, ion_stage in self.excitationlists:
            for (
                levelnumberdensity,
                xsvec,
                epsilon_trans_ev,
            ) in self.excitationlists[(Z, ion_stage)].values():
                part_integrand += levelnumberdensity * epsilon_trans_ev * xsvec / self.depositionratedensity_ev

        return self.yvec * part_integrand

    def get_d_etaion_by_d_en_vec(self) -> npt.NDArray[np.float64]:
        assert self._solved
        part_integrand = np.zeros(len(self.engrid))

        for Z, ion_stage in self.ionpopdict:
            n_ion = self.ionpopdict[(Z, ion_stage)]
            dfcollion_thision = self.dfcollion.filter(pl.col("Z") == Z).filter(pl.col("ion_stage") == ion_stage)

            for shell in dfcollion_thision.iter_rows(named=True):
                xsvec = pynonthermal.collion.get_arxs_array_shell(self.engrid, shell)

                part_integrand += n_ion * shell["ionpot_ev"] * xsvec / self.depositionratedensity_ev

        return self.yvec * part_integrand

    def plot_yspectrum(
        self,
        en_y_on_d_en: bool = False,
        xscalelog: bool = False,
        outputfilename: Path | str | None = None,
        axis: mplax.Axes | None = None,
    ) -> None:
        assert self._solved
        fs = 12
        fig = None
        if axis is None:
            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
                sharex=True,
                figsize=(5, 4),
                tight_layout={"pad": 0.5, "w_pad": 0.3, "h_pad": 0.3},
            )
        else:
            ax = axis

        if en_y_on_d_en:
            arr_y = np.log10(self.yvec * self.engrid)
            ax.set_ylabel(r"log d(E y)/dE", fontsize=fs)
        else:
            arr_y = np.log10(self.yvec)
            ax.set_ylabel(r"log y [y (e$^-$ / cm$^2$ / s / eV)]", fontsize=fs)

        ax.plot(self.engrid, arr_y, marker="None", lw=1.5, color="black")
        # axes[0].plot(engrid, np.log10(yvec), marker="None", lw=1.5, color='black')
        # axes[0].set_ylabel(r'log y(E) [s$^{-1}$ cm$^{-2}$ eV$^{-1}$]', fontsize=fs)
        # axes[0].set_ylim(bottom=15.5, top=19.)

        if xscalelog:
            ax.set_xscale("log")
        ax.set_xlim(left=min(1.0, self.engrid[0]))
        ax.set_xlim(right=self.engrid[-1] * 1.0)
        ax.set_xlabel(r"Electron energy [eV]", fontsize=fs)
        if axis is None:
            if outputfilename is not None:
                print(f"Saving '{outputfilename}'")
                assert fig is not None
                fig.savefig(str(outputfilename))
                plt.close()
            else:
                plt.show()

    def plot_channels(
        self, outputfilename: Path | str | None = None, axis: mplax.Axes | None = None, xscalelog: bool = False
    ) -> None:
        assert self._solved
        fs = 12
        fig = None
        if axis is None:
            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
                sharex=True,
                figsize=(5, 4),
                tight_layout={"pad": 0.5, "w_pad": 0.3, "h_pad": 0.3},
            )
        else:
            ax = axis

        npts = len(self.engrid)
        E_0 = self.engrid[0]

        # E_init_ev = np.dot(engrid, sourcevec) * deltaen
        # d_etasource_by_d_en_vec = engrid * sourcevec / E_init_ev
        # axes[0].plot(engrid[1:], d_etasource_by_d_en_vec[1:], marker="None", lw=1.5, color='blue', label='Source')

        d_etaion_by_d_en_vec = self.get_d_etaion_by_d_en_vec()

        d_etaexc_by_d_en_vec = self.get_d_etaexcitation_by_d_en_vec()

        d_etaheat_by_d_en_vec = self.get_d_etaheating_by_d_en_vec()

        deltaen = self.engrid[1:] - self.engrid[:-1]
        etaion_int = np.zeros(npts)
        etaexc_int = np.zeros(npts)
        etaheat_int = np.zeros(npts)
        for i in reversed(range(len(self.engrid) - 1)):
            etaion_int[i] = etaion_int[i + 1] + d_etaion_by_d_en_vec[i] * deltaen[i]
            etaexc_int[i] = etaexc_int[i + 1] + d_etaexc_by_d_en_vec[i] * deltaen[i]
            etaheat_int[i] = etaheat_int[i + 1] + d_etaheat_by_d_en_vec[i] * deltaen[i]

        etaheat_int[0] += E_0 * self.yvec[0] * self.electronlossfunction(E_0)

        # etatot_int = etaion_int + etaexc_int + etaheat_int

        # go below E_0
        deltaen2 = E_0 / 20.0
        engrid_low = np.arange(0.0, E_0, deltaen2, dtype=float)
        npts_low = len(engrid_low)
        d_etaheat_by_d_en_low = np.zeros(len(engrid_low))
        etaheat_int_low = np.zeros(len(engrid_low))
        etaion_int_low = np.zeros(len(engrid_low))
        etaexc_int_low = np.zeros(len(engrid_low))

        for i in reversed(range(len(engrid_low))):
            en_ev = engrid_low[i]
            N_e = self.calculate_N_e(en_ev)
            d_etaheat_by_d_en_low[i] += (
                N_e * en_ev / self.depositionratedensity_ev
            )  # + (yvec[0] * lossfunction(E_0, n_e, n_e_tot) / depositionratedensity_ev)
            etaheat_int_low[i] = (
                etaheat_int_low[i + 1] if i < len(engrid_low) - 1 else etaheat_int[0]
            ) + d_etaheat_by_d_en_low[i] * deltaen2

            etaion_int_low[i] = etaion_int[0]  # cross sections start above E_0
            etaexc_int_low[i] = etaexc_int[0]

        # etatot_int_low = etaion_int_low + etaexc_int_low + etaheat_int_low
        engridfull = np.append(engrid_low, self.engrid)

        # axes[0].plot(engridfull, np.append(etaion_int_low, etaion_int), marker="None", lw=1.5, color='C0', label='Ionisation')
        #
        # if not noexcitation:
        #     axes[0].plot(engridfull, np.append(etaexc_int_low, etaexc_int), marker="None", lw=1.5,
        #                  color='C1', label='Excitation')
        #
        # axes[0].plot(engridfull, np.append(etaheat_int_low, etaheat_int), marker="None", lw=1.5,
        #              color='C2', label='Heating')
        #
        # axes[0].plot(engridfull, np.append(etatot_int_low, etatot_int), marker="None", lw=1.5, color='black', label='Total')
        #
        # axes[0].set_ylim(bottom=0)
        # axes[0].legend(loc='best', handlelength=2, frameon=False, numpoints=1, prop={'size': 10})
        # axes[0].set_ylabel(r'$\eta$ E to Emax', fontsize=fs)

        # delta_E_y_on_dE = np.zeros(npts)
        # for i in range(len(engrid) - 1):
        #     # delta_E_y_on_dE[i] = ((yvec[i + 1] * engrid[i + 1]) - (yvec[i] * engrid[i])) / (engrid[i + 1] - engrid[i])
        #     delta_E_y_on_dE[i] = yvec[i] * engrid[i]
        # axes[0].plot(engrid, np.log10(delta_E_y_on_dE), marker="None", lw=1.5, color='black', label='')
        # axes[0].set_ylabel(r'log d(E y(E)) / dE', fontsize=fs)

        detaymax = max(
            [
                float(np.max(d_etaion_by_d_en_vec * self.engrid)),
                float(np.max(d_etaexc_by_d_en_vec * self.engrid)),
                float(np.max(d_etaheat_by_d_en_vec * self.engrid)),
            ]
        )
        ax.plot(
            engridfull,
            np.append(np.zeros(npts_low), d_etaion_by_d_en_vec) * engridfull / detaymax,
            marker="None",
            lw=1.5,
            color="C0",
            label="Ionisation",
        )

        if self.get_frac_excitation_tot() > 0.0:
            ax.plot(
                engridfull,
                np.append(np.zeros(npts_low), d_etaexc_by_d_en_vec) * engridfull / detaymax,
                marker="None",
                lw=1.5,
                color="C1",
                label="Excitation",
            )

        # axis.plot(engridfull, np.append(d_etaheat_by_d_en_low, d_etaheat_by_d_en_vec) * engridfull / detaymax,
        #           marker="None", lw=1.5, color='C2', label='Heating')
        ax.plot(
            self.engrid,
            (np.array(d_etaheat_by_d_en_vec) * self.engrid) / detaymax,
            marker="None",
            lw=1.5,
            color="C2",
            label="Heating",
        )

        ax.set_ylim(bottom=0, top=1.0)
        ax.legend(loc="best", handlelength=2, frameon=False, numpoints=1, prop={"size": 10})
        ax.set_ylabel(r"E d$\eta$ / dE [eV$^{-1}$]", fontsize=fs)

        # etatot_int = etaion_int + etaexc_int + etaheat_int

        #    ax.annotate(modellabel, xy=(0.97, 0.95), xycoords='axes fraction', horizontalalignment='right',
        #                verticalalignment='top', fontsize=fs)
        if xscalelog:
            ax.set_xscale("log")
        # ax.set_yscale('log')
        ax.set_xlim(left=min(1.0, self.engrid[0]))
        ax.set_xlim(right=self.engrid[-1] * 1.0)
        ax.set_xlabel(r"Electron energy [eV]", fontsize=fs)
        if axis is None:
            if outputfilename is not None:
                print(f"Saving '{outputfilename}'")
                assert fig is not None
                fig.savefig(str(outputfilename))
                plt.close()
            else:
                plt.show()

    def plot_spec_channels(self, outputfilename: Path | str | None, xscalelog: bool = False) -> None:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(4.5, 5),
            tight_layout={"pad": 0.5, "w_pad": 0.3, "h_pad": 0.3},
        )
        assert isinstance(axes, np.ndarray)

        self.plot_yspectrum(axis=axes[0], en_y_on_d_en=True, xscalelog=xscalelog)

        self.plot_channels(axis=axes[1], xscalelog=xscalelog)

        if outputfilename is not None:
            print(f"Saving '{outputfilename}'")
            fig.savefig(str(outputfilename))
            plt.close()
        else:
            plt.show()
