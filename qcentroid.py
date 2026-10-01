"""QCentroid solver entry point — Football Team Creator.

QUBO-based football squad search solved locally with D-Wave Ocean's
quantum-inspired samplers: simulated annealing (`neal`) and tabu search
(`dwave-tabu`).

Do not rename this file and do not change the run() signature — the platform's
execution engine looks specifically for qcentroid.py::run().
"""

import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: the platform has no display, only additional_output/ files
import matplotlib.pyplot as plt

import dimod
import neal
import tabu

logger = logging.getLogger("qcentroid-user-log")

ADDITIONAL_OUTPUT_DIR = "additional_output"

# Set K = {GK, DF, MF, FW} — fixed order used by every array-valued field
POSITIONS: List[str] = ["GK", "DF", "MF", "FW"]
K = len(POSITIONS)

# Position–statistic incidence M_k, fixed by the club
POSITION_STATS: Dict[str, List[str]] = {
    "GK": ["save", "rating"],
    "DF": ["tack", "pass", "rating"],
    "MF": ["pass", "goal", "xg", "rating"],
    "FW": ["goal", "xg", "rating"],
}

# Secondary-position policy used to build E_i (preferred ∪ secondary positions).
# Outfield players are eligible for tactically adjacent roles; goalkeepers are
# only ever eligible for GK, and no outfielder can be eligible for GK.
SECONDARY_POSITIONS: Dict[str, List[str]] = {
    "GK": [],
    "DF": ["MF"],
    "MF": ["DF", "FW"],
    "FW": ["MF"],
}

KNOWN_INPUT_DATA_KEYS = {
    "NC", "NT", "N", "W_max", "B0", "L_k", "U_k", "rho_k", "Sigma_max", "Delta_max",
    "mu_min", "alpha", "A_hi", "A_lo", "ai", "wi", "vi", "ci", "mui", "ri", "pi",
    "ti", "gi", "xi_xg", "si", "p_pref",
}

KNOWN_SOLVER_PARAMS = {
    "sampler", "num_reads_sa", "num_sweeps_sa", "num_restarts_tabu", "tenure",
    "timeout_tabu", "seed", "penalty_scale", "repair_solution", "generate_plots",
}


# --- Input guards ----------------------------------------------------------
# Catches typos (e.g. "W_max" -> "Wmax") that .get(key, default) would
# otherwise silently ignore by falling back to a default with no error.
def validate_input_data_keys(input_data: dict) -> None:
    unknown = set(input_data) - KNOWN_INPUT_DATA_KEYS
    if unknown:
        raise ValueError(
            f"Unknown input_data key(s): {sorted(unknown)}. "
            f"Valid keys are: {sorted(KNOWN_INPUT_DATA_KEYS)}."
        )


def validate_solver_params(solver_params: dict) -> None:
    unknown = set(solver_params) - KNOWN_SOLVER_PARAMS
    if unknown:
        raise ValueError(
            f"Unknown solver_params key(s): {sorted(unknown)}. "
            f"Valid keys are: {sorted(KNOWN_SOLVER_PARAMS)}."
        )


# Like dict.get(key, default), but logs when the key is absent and the default
# is applied — so a caller who omitted a param can see it in the Execution logs
# instead of silently losing control over that value.
def get_with_default(source: dict, key: str, default, label: str):
    if key in source:
        return source[key]
    logger.info(f"{label}: '{key}' not provided, using default {default!r}")
    return default


# --- Problem container -----------------------------------------------------
class Problem:
    """Holds every set and parameter of the mathematical definition."""

    def __init__(self) -> None:
        self.NC: int = 0
        self.NT: int = 0
        self.P: int = 0
        self.N: int = 0
        self.W_max: float = 0.0
        self.B0: float = 0.0
        self.L_k: List[int] = []
        self.U_k: List[int] = []
        self.rho_k: List[int] = []
        self.Sigma_max: int = 0
        self.Delta_max: int = 0
        self.mu_min: float = 0.0
        self.alpha: float = 0.7
        self.A_hi: float = 20.0
        self.A_lo: float = 30.0
        self.ai: List[int] = []
        self.wi: List[float] = []
        self.vi: List[float] = []   # length NC (current players only)
        self.ci: List[float] = []   # length NT (candidates only)
        self.mui: List[float] = []
        self.ri: List[float] = []
        self.pi: List[int] = []
        self.ti: List[int] = []
        self.gi: List[int] = []
        self.xg: List[float] = []
        self.si: List[int] = []
        self.p_pref: List[str] = []
        # derived
        self.E: List[List[int]] = []          # eligible position indices per player
        self.R: List[List[int]] = []          # reference group per position
        self.Q: List[Dict[int, float]] = []   # Q_{i,k}
        self.q_perf: List[Dict[int, float]] = []
        self.q_pot: List[float] = []

    # Helper: sale value of current player i (0 for candidates)
    def value_of(self, i: int) -> float:
        return float(self.vi[i]) if i < self.NC else 0.0

    # Helper: transfer fee of candidate i (0 for current players)
    def fee_of(self, i: int) -> float:
        return float(self.ci[i - self.NC]) if i >= self.NC else 0.0


# --- Data loading — maps every parameter of the formulation ----------------
def _as_float_list(raw: Sequence, name: str, expected: int) -> List[float]:
    values = [float(v) for v in raw]
    if len(values) != expected:
        raise ValueError(f"'{name}' has length {len(values)}, expected {expected}.")
    return values


def _as_int_list(raw: Sequence, name: str, expected: int) -> List[int]:
    values = [int(v) for v in raw]
    if len(values) != expected:
        raise ValueError(f"'{name}' has length {len(values)}, expected {expected}.")
    return values


def load_problem_data(input_data: dict) -> Problem:
    """Reads input_data into a Problem object (sets C, T, P and all parameters)."""
    required = [
        "NC", "NT", "N", "W_max", "B0", "L_k", "U_k", "rho_k", "Sigma_max",
        "Delta_max", "mu_min", "alpha", "A_hi", "A_lo", "ai", "wi", "vi", "ci",
        "mui", "ri", "pi", "ti", "gi", "xi_xg", "si", "p_pref",
    ]
    missing = [k for k in required if k not in input_data]
    if missing:
        raise ValueError(f"Missing required input_data key(s): {missing}.")

    p = Problem()
    p.NC = int(input_data["NC"])
    p.NT = int(input_data["NT"])
    p.P = p.NC + p.NT
    p.N = int(input_data["N"])
    p.W_max = float(input_data["W_max"])
    p.B0 = float(input_data["B0"])
    p.L_k = _as_int_list(input_data["L_k"], "L_k", K)
    p.U_k = _as_int_list(input_data["U_k"], "U_k", K)
    p.rho_k = _as_int_list(input_data["rho_k"], "rho_k", K)
    p.Sigma_max = int(input_data["Sigma_max"])
    p.Delta_max = int(input_data["Delta_max"])
    p.mu_min = float(input_data["mu_min"])
    p.alpha = float(input_data["alpha"])
    p.A_hi = float(input_data["A_hi"])
    p.A_lo = float(input_data["A_lo"])

    p.ai = _as_int_list(input_data["ai"], "ai", p.P)
    p.wi = _as_float_list(input_data["wi"], "wi", p.P)
    p.vi = _as_float_list(input_data["vi"], "vi", p.NC)
    p.ci = _as_float_list(input_data["ci"], "ci", p.NT)
    p.mui = _as_float_list(input_data["mui"], "mui", p.P)
    p.ri = _as_float_list(input_data["ri"], "ri", p.P)
    p.pi = _as_int_list(input_data["pi"], "pi", p.P)
    p.ti = _as_int_list(input_data["ti"], "ti", p.P)
    p.gi = _as_int_list(input_data["gi"], "gi", p.P)
    p.xg = _as_float_list(input_data["xi_xg"], "xi_xg", p.P)
    p.si = _as_int_list(input_data["si"], "si", p.P)

    prefs = [str(s).strip().upper() for s in input_data["p_pref"]]
    if len(prefs) != p.P:
        raise ValueError(f"'p_pref' has length {len(prefs)}, expected {p.P}.")
    bad = sorted({s for s in prefs if s not in POSITIONS})
    if bad:
        raise ValueError(f"Unknown preferred position(s) {bad}; valid values are {POSITIONS}.")
    p.p_pref = prefs

    if p.A_lo <= p.A_hi:
        raise ValueError("A_lo must be strictly greater than A_hi.")
    if not (0.0 <= p.alpha <= 1.0):
        raise ValueError("alpha must lie in [0, 1].")
    if any(m <= 0 for m in p.mui):
        raise ValueError("All 'mui' values must be strictly positive (per-90 normalization).")
    if any(w <= 0 for w in p.wi):
        raise ValueError("All 'wi' values must be strictly positive.")
    return p


# --- Input-feasibility cross-checks (verified before the run) -------------
def check_input_feasibility(p: Problem) -> List[str]:
    """Implements the 'Input-feasibility' block of the formulation.

    Hard inconsistencies raise; soft data-quality issues are logged and returned.
    """
    if not (sum(p.L_k) <= p.N <= sum(p.U_k)):
        raise ValueError(
            f"Infeasible bounds: sum(L_k)={sum(p.L_k)} <= N={p.N} <= sum(U_k)={sum(p.U_k)} violated."
        )
    if not (p.NC - p.Delta_max <= p.N <= p.NC + p.Sigma_max):
        raise ValueError(
            f"Infeasible transaction limits: NC-Delta_max={p.NC - p.Delta_max} <= N={p.N} "
            f"<= NC+Sigma_max={p.NC + p.Sigma_max} violated."
        )
    if sum(p.rho_k) > p.Sigma_max:
        raise ValueError(f"sum(rho_k)={sum(p.rho_k)} exceeds Sigma_max={p.Sigma_max}.")
    for k in range(K):
        if p.rho_k[k] > p.U_k[k]:
            raise ValueError(f"rho_k[{POSITIONS[k]}]={p.rho_k[k]} exceeds U_k={p.U_k[k]}.")
        if p.L_k[k] > p.U_k[k]:
            raise ValueError(f"L_k[{POSITIONS[k]}]={p.L_k[k]} exceeds U_k={p.U_k[k]}.")

    warnings: List[str] = []
    for k in range(K):
        if len(p.R[k]) < 5:
            warnings.append(
                f"Reference group |R_{POSITIONS[k]}|={len(p.R[k])} < 5: percentiles are unreliable."
            )
    below = [i for i in range(p.P) if p.mui[i] < p.mu_min]
    if below:
        warnings.append(
            f"{len(below)} player(s) have minutes below mu_min={p.mu_min}; "
            f"their statistics are still used (indices: {below[:10]}{'...' if len(below) > 10 else ''})."
        )
    for w in warnings:
        logger.info(f"Input-feasibility warning: {w}")
    return warnings


# --- Sets E_i and R_k ------------------------------------------------------
def build_eligibility(p: Problem) -> None:
    """Builds E_i (eligible positions) and R_k (reference groups)."""
    p.E = []
    for i in range(p.P):
        pref = p.p_pref[i]
        eligible = [pref] + [k for k in SECONDARY_POSITIONS[pref] if k != pref]
        p.E.append([POSITIONS.index(k) for k in eligible])
    p.R = [[i for i in range(p.P) if k in p.E[i]] for k in range(K)]


# --- Per-90 normalization: z_{i,m} = 90 y_{i,m} / mu_i, rating used as-is --
def compute_per90_stats(p: Problem) -> Dict[str, np.ndarray]:
    mu = np.array(p.mui, dtype=float)
    z = {
        "pass": 90.0 * np.array(p.pi, dtype=float) / mu,
        "tack": 90.0 * np.array(p.ti, dtype=float) / mu,
        "goal": 90.0 * np.array(p.gi, dtype=float) / mu,
        "xg": 90.0 * np.array(p.xg, dtype=float) / mu,
        "save": 90.0 * np.array(p.si, dtype=float) / mu,
        "rating": np.array(p.ri, dtype=float),
    }
    return z


# --- Percentile score (mid-rank convention) within reference group R_k ----
def compute_percentiles(p: Problem, z: Dict[str, np.ndarray]) -> Dict[Tuple[int, str], np.ndarray]:
    """phi[(k, m)][i] = 100 * (#{z_j < z_i} + 0.5 #{z_j == z_i}) / |R_k| for i in R_k."""
    phi: Dict[Tuple[int, str], np.ndarray] = {}
    for k in range(K):
        group = p.R[k]
        if not group:
            continue
        n_group = len(group)
        for m in POSITION_STATS[POSITIONS[k]]:
            values = z[m][group]
            scores = np.zeros(p.P, dtype=float)
            for idx, i in enumerate(group):
                vi = values[idx]
                lower = int(np.count_nonzero(values < vi))
                equal = int(np.count_nonzero(values == vi))
                scores[i] = 100.0 * (lower + 0.5 * equal) / n_group
            phi[(k, m)] = scores
    return phi


# --- q_perf, q_pot and the objective coefficients Q_{i,k} -----------------
def compute_quality(p: Problem, phi: Dict[Tuple[int, str], np.ndarray]) -> None:
    # Potential score: piecewise-linear decay in age
    p.q_pot = [
        100.0 * min(max((p.A_lo - float(a)) / (p.A_lo - p.A_hi), 0.0), 1.0) for a in p.ai
    ]
    # Current performance: unweighted mean of the position's percentile scores
    p.q_perf = [dict() for _ in range(p.P)]
    p.Q = [dict() for _ in range(p.P)]
    for i in range(p.P):
        for k in p.E[i]:
            stats = POSITION_STATS[POSITIONS[k]]
            vals = [float(phi[(k, m)][i]) for m in stats]
            perf = sum(vals) / len(vals)
            p.q_perf[i][k] = perf
            # Final quality Q_{i,k} = alpha q_perf + (1-alpha) q_pot
            p.Q[i][k] = p.alpha * perf + (1.0 - p.alpha) * p.q_pot[i]


# --- Decision variables: one binary per eligible (i,k) pair ---------------
def build_variables(p: Problem) -> Tuple[Dict[Tuple[int, int], str], List[Tuple[int, int]]]:
    """Instantiates x_{i,k} only where e_{i,k}=1 (C1's second line removes the rest)."""
    var_of: Dict[Tuple[int, int], str] = {}
    pairs: List[Tuple[int, int]] = []
    for i in range(p.P):
        for k in p.E[i]:
            var_of[(i, k)] = f"x_{i}_{k}"
            pairs.append((i, k))
    return var_of, pairs


# --- Binary slack (log) encoding for inequality -> equality conversion ----
def binary_slack_terms(prefix: str, upper: float, unit: float = 1.0) -> List[Tuple[str, float]]:
    """Returns [(variable, coefficient)] encoding a slack in [0, upper] (step `unit`)."""
    if upper <= 0:
        return []
    steps = int(math.floor(upper / unit))
    if steps <= 0:
        return []
    nbits = max(1, int(math.ceil(math.log2(steps + 1))))
    terms: List[Tuple[str, float]] = []
    remaining = steps
    for b in range(nbits):
        coeff = min(2 ** b, remaining)
        if coeff <= 0:
            break
        remaining -= coeff
        terms.append((f"{prefix}_{b}", float(coeff) * unit))
    return terms


def _add_squared_penalty(
    bqm: dimod.BinaryQuadraticModel,
    terms: List[Tuple[str, float]],
    offset: float,
    weight: float,
) -> None:
    """Adds weight * (sum_j coeff_j * b_j + offset)^2 to the BQM."""
    # Linear part of the expansion: coeff^2 + 2*offset*coeff (b^2 = b for binaries)
    for var, coeff in terms:
        bqm.add_linear(var, weight * (coeff * coeff + 2.0 * offset * coeff))
    # Quadratic cross terms
    for a in range(len(terms)):
        va, ca = terms[a]
        for b in range(a + 1, len(terms)):
            vb, cb = terms[b]
            bqm.add_quadratic(va, vb, weight * 2.0 * ca * cb)
    bqm.offset += weight * offset * offset


# --- QUBO construction: -F(x) + penalty blocks for C1..C9 -----------------
def build_qubo(
    p: Problem,
    var_of: Dict[Tuple[int, int], str],
    pairs: List[Tuple[int, int]],
    penalty_scale: float,
) -> Tuple[dimod.BinaryQuadraticModel, Dict[str, float], Dict[str, int]]:
    """Builds the QUBO (BQM) of the penalty-reformulated model.

    Returns the BQM, the penalty weights used, and the slack-bit counts.
    """
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)

    # Objective (minimization form): -sum_i sum_k Q_{i,k} x_{i,k}
    q_max = max((max(d.values()) for d in p.Q if d), default=100.0)
    for (i, k) in pairs:
        bqm.add_linear(var_of[(i, k)], -float(p.Q[i][k]))

    # Penalty base: lambda > max Q * N guarantees no violation pays off
    base = penalty_scale * max(1.0, q_max) * max(1, p.N)
    lam = {
        "C1_exclusive": base,
        "C3_size": base,
        "C2_salary": base,
        "C6_budget": base,
        "C4_C5_position": base,
        "C7_priority": base,
        "C8_signings": base,
        "C9_sales": base,
    }
    slack_bits: Dict[str, int] = {}

    # C1 — single assignment: lambda1 * (sum_k x_ik)(sum_k x_ik - 1)
    for i in range(p.P):
        if len(p.E[i]) < 2:
            continue  # a single eligible position can never violate C1
        vars_i = [var_of[(i, k)] for k in p.E[i]]
        for a in range(len(vars_i)):
            for b in range(a + 1, len(vars_i)):
                bqm.add_quadratic(vars_i[a], vars_i[b], 2.0 * lam["C1_exclusive"])

    # C3 — exact squad size: lambda2 * (sum x - N)^2
    size_terms = [(var_of[pr], 1.0) for pr in pairs]
    _add_squared_penalty(bqm, size_terms, -float(p.N), lam["C3_size"])

    # C2 — salary limit: lambda3 * (sum w_i x_ik + sigma_W - W_max)^2, scaled to EUR millions
    money_unit = 1.0e6
    salary_terms = [(var_of[(i, k)], p.wi[i] / money_unit) for (i, k) in pairs]
    slack_W = binary_slack_terms("sW", p.W_max / money_unit, unit=1.0)
    slack_bits["salary"] = len(slack_W)
    _add_squared_penalty(
        bqm, salary_terms + slack_W, -p.W_max / money_unit, lam["C2_salary"]
    )

    # C6 — transfer budget: lambda4 * (fees + sale-values-retained + sigma_B - B_avail)^2
    B_avail_max = p.B0 + sum(p.vi)
    budget_terms = []
    for (i, k) in pairs:
        coeff = (p.fee_of(i) if i >= p.NC else p.value_of(i)) / money_unit
        if coeff != 0.0:
            budget_terms.append((var_of[(i, k)], coeff))
    slack_B = binary_slack_terms("sB", B_avail_max / money_unit, unit=1.0)
    slack_bits["budget"] = len(slack_B)
    _add_squared_penalty(
        bqm, budget_terms + slack_B, -B_avail_max / money_unit, lam["C6_budget"]
    )

    # C4/C5 — position bounds: lambda5 * (sum_{R_k} x_ik - L_k - sigma_k)^2, sigma_k <= U_k - L_k
    for k in range(K):
        pos_terms = [(var_of[(i, k)], 1.0) for i in p.R[k]]
        if not pos_terms:
            continue
        slack_k = [(v, -c) for v, c in binary_slack_terms(f"sPos{k}", p.U_k[k] - p.L_k[k])]
        slack_bits[f"position_{POSITIONS[k]}"] = len(slack_k)
        _add_squared_penalty(
            bqm, pos_terms + slack_k, -float(p.L_k[k]), lam["C4_C5_position"]
        )

    # C7 — priority positions: lambda6 * (sum_{T∩R_k} x_ik - rho_k - sigma)^2
    for k in range(K):
        if p.rho_k[k] <= 0:
            continue
        prio_terms = [(var_of[(i, k)], 1.0) for i in p.R[k] if i >= p.NC]
        if not prio_terms:
            raise ValueError(
                f"Priority position {POSITIONS[k]} requires {p.rho_k[k]} signing(s) "
                "but no transfer candidate is eligible for it."
            )
        slack_rho = [
            (v, -c)
            for v, c in binary_slack_terms(f"sRho{k}", max(0, len(prio_terms) - p.rho_k[k]))
        ]
        slack_bits[f"priority_{POSITIONS[k]}"] = len(slack_rho)
        _add_squared_penalty(
            bqm, prio_terms + slack_rho, -float(p.rho_k[k]), lam["C7_priority"]
        )

    # C8 — maximum signings: lambda7 * (sum_{T} x_ik + sigma - Sigma_max)^2
    sign_terms = [(var_of[(i, k)], 1.0) for (i, k) in pairs if i >= p.NC]
    slack_sign = binary_slack_terms("sSig", p.Sigma_max)
    slack_bits["signings"] = len(slack_sign)
    _add_squared_penalty(
        bqm, sign_terms + slack_sign, -float(p.Sigma_max), lam["C8_signings"]
    )

    # C9 — maximum sales: lambda8 * (sum_{C} x_ik - sigma - (NC - Delta_max))^2
    keep_terms = [(var_of[(i, k)], 1.0) for (i, k) in pairs if i < p.NC]
    keep_lower = max(0, p.NC - p.Delta_max)
    slack_keep = [(v, -c) for v, c in binary_slack_terms("sDel", max(0, p.NC - keep_lower))]
    slack_bits["sales"] = len(slack_keep)
    _add_squared_penalty(bqm, keep_terms + slack_keep, -float(keep_lower), lam["C9_sales"])

    logger.info(
        f"QUBO built: {len(bqm.variables)} binary variables "
        f"({len(pairs)} assignment + {len(bqm.variables) - len(pairs)} slack), "
        f"{len(bqm.quadratic)} quadratic terms, penalty base={base:.1f}"
    )
    return bqm, lam, slack_bits


def qubo_matrix_from_bqm(
    bqm: dimod.BinaryQuadraticModel, pairs: List[Tuple[int, int]], var_of: Dict[Tuple[int, int], str]
) -> np.ndarray:
    """Dense matrix of the assignment-variable block of the QUBO (for the heatmap)."""
    n = len(pairs)
    index = {var_of[pr]: idx for idx, pr in enumerate(pairs)}
    M = np.zeros((n, n), dtype=float)
    for v, bias in bqm.linear.items():
        if v in index:
            M[index[v], index[v]] = float(bias)
    for (u, v), bias in bqm.quadratic.items():
        if u in index and v in index:
            a, b = index[u], index[v]
            M[min(a, b), max(a, b)] += float(bias)
    return M


# --- Solver execution: local quantum-inspired samplers --------------------
def solve_with_simulated_annealing(
    bqm: dimod.BinaryQuadraticModel, num_reads: int, num_sweeps: int, seed: Optional[int]
) -> dimod.SampleSet:
    sampler = neal.SimulatedAnnealingSampler()
    return sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)


def solve_with_tabu(
    bqm: dimod.BinaryQuadraticModel,
    num_restarts: int,
    tenure: Optional[int],
    timeout: int,
    seed: Optional[int],
) -> dimod.SampleSet:
    sampler = tabu.TabuSampler()
    kwargs = dict(num_restarts=num_restarts, timeout=timeout, seed=seed)
    if tenure is not None:
        kwargs["tenure"] = int(tenure)
    return sampler.sample(bqm, **kwargs)


# --- Decoding and independent verification (C1–C10) ----------------------
def decode_sample(
    sample: Dict[str, int], p: Problem, var_of: Dict[Tuple[int, int], str]
) -> List[Optional[int]]:
    """Returns k*_i per player (None if not in the squad). C1 violations are flagged later."""
    assign: List[Optional[int]] = [None] * p.P
    multi: List[int] = []
    for i in range(p.P):
        chosen = [k for k in p.E[i] if int(sample.get(var_of[(i, k)], 0)) == 1]
        if len(chosen) == 1:
            assign[i] = chosen[0]
        elif len(chosen) > 1:
            assign[i] = chosen[0]
            multi.append(i)
    if multi:
        # Mark the sample as C1-infeasible via a sentinel the verifier reads back
        assign.append(-1)  # type: ignore[arg-type]
        assign.pop()
        return assign if not multi else _mark_infeasible(assign)
    return assign


def _mark_infeasible(assign: List[Optional[int]]) -> List[Optional[int]]:
    """Wraps an assignment so verify_constraints() reports C1 as failed."""
    marked = list(assign)
    marked.append("C1_VIOLATED")  # type: ignore[arg-type]
    return marked


def split_marker(assign: List) -> Tuple[List[Optional[int]], bool]:
    """Separates the optional C1 marker from a decoded assignment."""
    if assign and assign[-1] == "C1_VIOLATED":
        return assign[:-1], False
    return assign, True


def verify_constraints(assign_raw: List, p: Problem) -> Tuple[Dict[str, str], bool]:
    """Recomputes C1–C10 independently from x* (Output Data field V)."""
    assign, c1_ok = split_marker(assign_raw)
    squad = [i for i in range(p.P) if assign[i] is not None]
    n_k = [sum(1 for i in squad if assign[i] == k) for k in range(K)]
    n_new_k = [sum(1 for i in squad if assign[i] == k and i >= p.NC) for k in range(K)]
    retained = [i for i in squad if i < p.NC]
    signed = [i for i in squad if i >= p.NC]

    W_tot = sum(p.wi[i] for i in squad)
    R_sale = sum(p.vi[i] for i in range(p.NC) if assign[i] is None)
    C_spend = sum(p.fee_of(i) for i in signed)

    eligible_ok = all(assign[i] in p.E[i] for i in squad)
    report = {
        "C1_single_assignment_and_eligibility": "pass" if (c1_ok and eligible_ok) else "fail",
        "C2_salary_limit": "pass" if W_tot <= p.W_max + 1e-6 else "fail",
        "C3_exact_squad_size": "pass" if len(squad) == p.N else "fail",
        "C4_min_players_per_position": "pass" if all(n_k[k] >= p.L_k[k] for k in range(K)) else "fail",
        "C5_max_players_per_position": "pass" if all(n_k[k] <= p.U_k[k] for k in range(K)) else "fail",
        "C6_transfer_budget": "pass" if C_spend <= p.B0 + R_sale + 1e-6 else "fail",
        "C7_priority_positions": "pass" if all(n_new_k[k] >= p.rho_k[k] for k in range(K)) else "fail",
        "C8_max_signings": "pass" if len(signed) <= p.Sigma_max else "fail",
        "C9_max_sales": "pass" if (p.NC - len(retained)) <= p.Delta_max else "fail",
        "C10_integrality": "pass",  # binary by construction of the decoder
    }
    feasible = all(v == "pass" for v in report.values())
    return report, feasible


# --- Greedy repair: deterministic fallback when no sample is feasible ----
def greedy_repair(p: Problem, warm_start: List[Optional[int]]) -> Optional[List[Optional[int]]]:
    """Constructs a feasible squad honouring C1–C9, biased towards the warm start.

    Strategy: fill the per-position minima L_k with the highest-quality affordable
    players, then top up to exactly N, always respecting U_k, W_max, budget,
    signing/sale limits and the priority requirements rho_k.
    """
    assign: List[Optional[int]] = [None] * p.P
    n_k = [0] * K
    n_new_k = [0] * K
    squad: List[int] = []
    W_tot = 0.0

    def retained_count() -> int:
        return sum(1 for i in squad if i < p.NC)

    def signed_count() -> int:
        return sum(1 for i in squad if i >= p.NC)

    def budget_ok(candidate: int) -> bool:
        # Sale revenue counts only the current players that end up NOT in the squad;
        # compute the optimistic bound: all currently-unassigned current players sellable
        spend = sum(p.fee_of(i) for i in squad if i >= p.NC) + p.fee_of(candidate)
        sellable = sum(
            p.vi[i] for i in range(p.NC) if i not in squad and i != candidate
        )
        return spend <= p.B0 + sellable + 1e-6

    def can_add(i: int, k: int) -> bool:
        if assign[i] is not None or k not in p.E[i]:
            return False
        if n_k[k] + 1 > p.U_k[k]:
            return False
        if len(squad) + 1 > p.N:
            return False
        if W_tot + p.wi[i] > p.W_max + 1e-6:
            return False
        if i >= p.NC:
            if signed_count() + 1 > p.Sigma_max:
                return False
            if not budget_ok(i):
                return False
        return True

    def add(i: int, k: int) -> None:
        nonlocal W_tot
        assign[i] = k
        squad.append(i)
        n_k[k] += 1
        if i >= p.NC:
            n_new_k[k] += 1
        W_tot += p.wi[i]

    # Step 1 — satisfy the priority signings rho_k first (hardest constraint)
    for k in range(K):
        needed = p.rho_k[k]
        if needed <= 0:
            continue
        candidates = sorted(
            (i for i in p.R[k] if i >= p.NC),
            key=lambda i: (-p.Q[i][k], p.fee_of(i), p.wi[i]),
        )
        for i in candidates:
            if n_new_k[k] >= needed:
                break
            if can_add(i, k):
                add(i, k)

    # Step 2 — honour the sales cap by keeping the best current players first
    min_keep = max(0, p.NC - p.Delta_max)
    current_sorted = sorted(
        range(p.NC),
        key=lambda i: -max(p.Q[i].values()),
    )
    for i in current_sorted:
        if retained_count() >= min_keep:
            break
        if assign[i] is not None:
            continue
        best_k = None
        best_q = -1.0
        for k in sorted(p.E[i], key=lambda kk: -p.Q[i][kk]):
            if can_add(i, k) and p.Q[i][k] > best_q:
                best_k, best_q = k, p.Q[i][k]
        if best_k is not None:
            add(i, best_k)

    # Step 3 — satisfy the per-position minima L_k with the best available players
    for k in range(K):
        while n_k[k] < p.L_k[k]:
            best_i = None
            best_q = -1.0
            for i in p.R[k]:
                if can_add(i, k) and p.Q[i][k] > best_q:
                    best_i, best_q = i, p.Q[i][k]
            if best_i is None:
                return None  # cannot reach L_k -> repair failed
            add(best_i, k)

    # Step 4 — top up to exactly N with the highest-quality remaining options,
    # preferring the warm-start's picks on ties
    while len(squad) < p.N:
        best = None
        best_key = None
        for i in range(p.P):
            if assign[i] is not None:
                continue
            for k in p.E[i]:
                if not can_add(i, k):
                    continue
                warm_bonus = 1.0 if (i < len(warm_start) and warm_start[i] == k) else 0.0
                key = (p.Q[i][k] + warm_bonus, -p.wi[i])
                if best_key is None or key > best_key:
                    best, best_key = (i, k), key
        if best is None:
            return None  # cannot reach exactly N -> repair failed
        add(best[0], best[1])

    report, feasible = verify_constraints(assign, p)
    if not feasible:
        logger.info(f"Greedy repair produced an infeasible squad: {report}")
        return None
    return assign


# --- Output assembly ------------------------------------------------------
def compute_outputs(assign: List[Optional[int]], p: Problem) -> dict:
    """Builds every Output Data field of the formulation from x*."""
    squad = [i for i in range(p.P) if assign[i] is not None]
    retained = [i for i in squad if i < p.NC]
    sold = [i for i in range(p.NC) if assign[i] is None]
    signed = [i for i in squad if i >= p.NC]

    k_i: List[Optional[str]] = [
        POSITIONS[assign[i]] if assign[i] is not None else None for i in range(p.P)
    ]
    q_perf_list = [float(p.q_perf[i][assign[i]]) for i in squad]
    q_pot_list = [float(p.q_pot[i]) for i in squad]
    Q_list = [float(p.Q[i][assign[i]]) for i in squad]

    n_k = [sum(1 for i in squad if assign[i] == k) for k in range(K)]
    n_new_k = [sum(1 for i in signed if assign[i] == k) for k in range(K)]

    R_sale = float(sum(p.vi[i] for i in sold))
    B_avail = float(p.B0 + R_sale)
    C_spend = float(sum(p.fee_of(i) for i in signed))
    B_rem = float(B_avail - C_spend)
    W_tot = float(sum(p.wi[i] for i in squad))
    W_rem = float(p.W_max - W_tot)

    F_x = float(sum(Q_list))
    Q_bar = float(F_x / p.N) if p.N > 0 else 0.0
    Q_bar_k: List[float] = []
    for k in range(K):
        members = [i for i in squad if assign[i] == k]
        Q_bar_k.append(
            float(sum(p.Q[i][k] for i in members) / len(members)) if members else 0.0
        )
    q_perf_bar = float(sum(q_perf_list) / p.N) if p.N > 0 else 0.0
    q_pot_bar = float(sum(q_pot_list) / p.N) if p.N > 0 else 0.0

    return {
        "F": [int(i) for i in squad],
        "A": [int(i) for i in retained],
        "D": [int(i) for i in sold],
        "N_set": [int(i) for i in signed],
        "k_i": k_i,
        "q_perf": q_perf_list,
        "q_pot": q_pot_list,
        "Q": Q_list,
        "n_k": [int(v) for v in n_k],
        "n_new_k": [int(v) for v in n_new_k],
        "n_sign": int(len(signed)),
        "n_sale": int(len(sold)),
        "R_sale": R_sale,
        "B_avail": B_avail,
        "C_spend": C_spend,
        "B_rem": B_rem,
        "W_tot": W_tot,
        "W_rem": W_rem,
        "F_x": F_x,
        "Q_bar": Q_bar,
        "Q_bar_k": Q_bar_k,
        "q_perf_bar": q_perf_bar,
        "q_pot_bar": q_pot_bar,
    }


def compute_baseline(p: Problem) -> Tuple[float, List[float], List[str]]:
    """Baseline Q0_bar / Q0_bar_k and the rules violated by the initial squad (B0_set)."""
    prefs = [POSITIONS.index(p.p_pref[i]) for i in range(p.NC)]
    q_values = [p.Q[i][prefs[i]] for i in range(p.NC)]
    Q0_bar = float(sum(q_values) / p.NC) if p.NC > 0 else 0.0
    Q0_bar_k: List[float] = []
    n0_k = [0] * K
    for k in range(K):
        members = [i for i in range(p.NC) if prefs[i] == k]
        n0_k[k] = len(members)
        Q0_bar_k.append(
            float(sum(p.Q[i][k] for i in members) / len(members)) if members else 0.0
        )

    violations: List[str] = []
    W0 = sum(p.wi[i] for i in range(p.NC))
    if W0 > p.W_max:
        violations.append(
            f"C2_salary_limit: initial wage bill {W0:.0f} EUR exceeds W_max {p.W_max:.0f} EUR"
        )
    if p.NC != p.N:
        violations.append(
            f"C3_exact_squad_size: initial squad size {p.NC} differs from required N={p.N}"
        )
    for k in range(K):
        if n0_k[k] < p.L_k[k]:
            violations.append(
                f"C4_min_players_per_position: only {n0_k[k]} {POSITIONS[k]} vs L_k={p.L_k[k]}"
            )
        if n0_k[k] > p.U_k[k]:
            violations.append(
                f"C5_max_players_per_position: {n0_k[k]} {POSITIONS[k]} vs U_k={p.U_k[k]}"
            )
        if p.rho_k[k] > 0:
            violations.append(
                f"C7_priority_positions: no new signing at priority position {POSITIONS[k]} "
                f"(rho_k={p.rho_k[k]})"
            )
    return Q0_bar, Q0_bar_k, violations


# --- JSON helpers ---------------------------------------------------------
def save_metadata_json(out_dir: str, metadata: dict) -> str:
    path = os.path.join(out_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path


# Written unconditionally (not gated by generate_plots) so every job leaves a
# record of exactly what input_data/solver_params produced its output.
# Deliberately excludes extra_arguments — it may carry platform-injected
# credential context, and additional_output/ assets are visible to end users.
def save_input_snapshot(out_dir: str, input_data: dict, solver_params: dict) -> str:
    path = os.path.join(out_dir, "input_used.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"data": input_data, "solver_params": solver_params}, f, indent=2)
    return path


# --- Visualization helpers -----------------------------------------------
# The platform runs headless: plots are written as PNG assets under
# additional_output/ and surfaced in the job's Assets tab. Plot choices follow
# this problem's structure: a composition/cardinality problem -> grouped bars;
# a per-player quality decomposition -> stacked bars; multi-sample samplers ->
# an energy distribution; a QUBO matrix -> a heatmap; budget constraints ->
# gauge-style usage bars.

def save_squad_composition_plot(out_dir: str, p: Problem, out: dict) -> str:
    """Direct visualization of the Output Data fields n_k / n_new_k vs L_k and U_k."""
    idx = np.arange(K)
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(idx - 1.5 * width, out["n_k"], width, label="$n_k$ assigned", color="royalblue",
           edgecolor="black")
    ax.bar(idx - 0.5 * width, out["n_new_k"], width, label="$n^{new}_k$ signings",
           color="seagreen", edgecolor="black")
    ax.bar(idx + 0.5 * width, p.L_k, width, label="$L_k$ min", color="lightgray",
           edgecolor="black")
    ax.bar(idx + 1.5 * width, p.U_k, width, label="$U_k$ max", color="darkgray",
           edgecolor="black")
    ax.set_xticks(idx)
    ax.set_xticklabels(POSITIONS)
    ax.set_ylabel("Players")
    ax.set_title(
        f"Squad composition — F(x*) = {out['F_x']:.2f}, "
        f"Q_bar = {out['Q_bar']:.2f}, Gamma% = {out['Gamma_pct']:.2f}%",
        fontsize=13, fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(out_dir, "squad_composition.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_quality_breakdown_plot(out_dir: str, out: dict) -> str:
    """Per-player decomposition of Q = alpha*q_perf + (1-alpha)*q_pot for the final squad."""
    squad = out["F"]
    order = np.argsort(-np.array(out["Q"]))
    labels = [f"{squad[i]} ({out['k_i'][squad[i]]})" for i in order]
    q_perf = np.array(out["q_perf"], dtype=float)[order]
    q_pot = np.array(out["q_pot"], dtype=float)[order]
    Q = np.array(out["Q"], dtype=float)[order]

    fig, ax = plt.subplots(figsize=(11, 6))
    pos = np.arange(len(labels))
    ax.bar(pos - 0.2, q_perf, 0.4, label="$q^{perf}$", color="steelblue", edgecolor="black")
    ax.bar(pos + 0.2, q_pot, 0.4, label="$q^{pot}$", color="orange", edgecolor="black")
    ax.plot(pos, Q, "k-o", markersize=4, label="$Q_{i,k^*}$ (final quality)")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Quality points (0–100)")
    ax.set_xlabel("Player index (assigned position)")
    ax.set_title(
        f"Final squad quality breakdown — Q_bar = {out['Q_bar']:.2f} "
        f"(baseline {out['Q0_bar']:.2f}, Gamma = {out['Gamma']:.2f})",
        fontsize=13, fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(out_dir, "quality_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_energy_distribution_plot(out_dir: str, energies_by_sampler: Dict[str, np.ndarray]) -> str:
    """Both samplers return many candidate solutions — compare their energy spreads."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"simulated_annealing": "slateblue", "tabu": "indianred"}
    for name, energies in energies_by_sampler.items():
        if energies.size == 0:
            continue
        ax.hist(
            energies, bins=min(40, max(5, energies.size)), alpha=0.65,
            label=f"{name} (best {energies.min():.1f})",
            color=colors.get(name, "gray"), edgecolor="black",
        )
    ax.set_xlabel("QUBO energy (objective + penalties)")
    ax.set_ylabel("Frequency")
    ax.set_title("Sampler energy distributions", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "energy_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_qubo_matrix_plot(out_dir: str, M: np.ndarray) -> str:
    """Heatmap of the assignment-variable block of Q, for sanity-checking the formulation."""
    fig, ax = plt.subplots(figsize=(8, 7))
    scale = np.max(np.abs(M)) if M.size else 1.0
    im = ax.imshow(M, cmap="coolwarm", vmin=-scale, vmax=scale, interpolation="nearest")
    ax.set_title("QUBO matrix (assignment variables $x_{i,k}$)", fontsize=13, fontweight="bold")
    ax.set_xlabel("variable index")
    ax.set_ylabel("variable index")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("QUBO coefficient")
    path = os.path.join(out_dir, "qubo_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_budget_usage_plot(out_dir: str, p: Problem, out: dict) -> str:
    """Resource usage against the hard caps W_max, B_avail, Sigma_max, Delta_max, N."""
    labels = ["Salary\n(EUR m)", "Transfer spend\n(EUR m)", "Signings", "Sales", "Squad size"]
    used = [
        out["W_tot"] / 1e6, out["C_spend"] / 1e6, out["n_sign"], out["n_sale"], len(out["F"]),
    ]
    caps = [
        p.W_max / 1e6, out["B_avail"] / 1e6, p.Sigma_max, p.Delta_max, p.N,
    ]
    pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(pos, caps, 0.6, color="lightgray", edgecolor="black", label="limit")
    ax.bar(pos, used, 0.35, color="crimson", edgecolor="black", label="used")
    for i, (u, c) in enumerate(zip(used, caps)):
        ax.text(i, max(u, c) * 1.02 + 0.05, f"{u:.2f}/{c:.2f}", ha="center", fontsize=9)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_title("Resource usage vs. constraint limits", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(out_dir, "budget_usage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Entry point ---------------------------------------------------------
def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    logger.info("Starting Football Team Creator QUBO solver (tabu + simulated annealing)...")
    start_time = time.time()

    # --- Guards: reject unknown keys before reading any value ---
    validate_input_data_keys(input_data)
    validate_solver_params(solver_params)

    # --- Solver parameters (knobs of the actual sampler calls) ---
    sampler_choice = str(
        get_with_default(solver_params, "sampler", "both", "solver_params")
    ).lower()
    if sampler_choice not in {"sa", "tabu", "both"}:
        raise ValueError(
            f"Invalid 'sampler' value {sampler_choice!r}; expected one of 'sa', 'tabu', 'both'."
        )
    num_reads_sa = int(get_with_default(solver_params, "num_reads_sa", 400, "solver_params"))
    num_sweeps_sa = int(get_with_default(solver_params, "num_sweeps_sa", 3000, "solver_params"))
    num_restarts_tabu = int(
        get_with_default(solver_params, "num_restarts_tabu", 50, "solver_params")
    )
    tenure_raw = get_with_default(solver_params, "tenure", None, "solver_params")
    tenure = int(tenure_raw) if tenure_raw is not None else None
    timeout_tabu = int(get_with_default(solver_params, "timeout_tabu", 5000, "solver_params"))
    seed_raw = get_with_default(solver_params, "seed", 42, "solver_params")
    seed = int(seed_raw) if seed_raw is not None else None
    penalty_scale = float(get_with_default(solver_params, "penalty_scale", 2.0, "solver_params"))
    if penalty_scale <= 0:
        raise ValueError("'penalty_scale' must be strictly positive.")
    repair_solution = bool(
        get_with_default(solver_params, "repair_solution", True, "solver_params")
    )
    generate_plots = bool(get_with_default(solver_params, "generate_plots", True, "solver_params"))

    # --- Traceability snapshot (always written) ---
    os.makedirs(ADDITIONAL_OUTPUT_DIR, exist_ok=True)
    save_input_snapshot(ADDITIONAL_OUTPUT_DIR, input_data, solver_params)

    # --- Load instance, build sets E_i / R_k, run data cross-checks ---
    p = load_problem_data(input_data)
    build_eligibility(p)
    input_warnings = check_input_feasibility(p)
    logger.info(
        f"Instance loaded: |C|={p.NC}, |T|={p.NT}, N={p.N}, "
        f"W_max={p.W_max:.0f}, B0={p.B0:.0f}, positions={POSITIONS}"
    )

    # --- Pre-processing: per-90 stats, percentiles, q_perf, q_pot, Q_{i,k} ---
    z = compute_per90_stats(p)
    phi = compute_percentiles(p, z)
    compute_quality(p, phi)

    # --- Variables and QUBO ---
    var_of, pairs = build_variables(p)
    bqm, lam, slack_bits = build_qubo(p, var_of, pairs, penalty_scale)

    # --- Sampling: local quantum-inspired samplers ---
    samplesets: List[Tuple[str, dimod.SampleSet]] = []
    energies_by_sampler: Dict[str, np.ndarray] = {}

    if sampler_choice in {"sa", "both"}:
        logger.info(
            f"Running neal.SimulatedAnnealingSampler (num_reads={num_reads_sa}, "
            f"num_sweeps={num_sweeps_sa}, seed={seed})..."
        )
        t0 = time.time()
        ss_sa = solve_with_simulated_annealing(bqm, num_reads_sa, num_sweeps_sa, seed)
        logger.info(f"Simulated annealing finished in {time.time() - t0:.3f}s")
        samplesets.append(("simulated_annealing", ss_sa))
        energies_by_sampler["simulated_annealing"] = np.array(ss_sa.record.energy, dtype=float)

    if sampler_choice in {"tabu", "both"}:
        logger.info(
            f"Running tabu.TabuSampler (num_restarts={num_restarts_tabu}, tenure={tenure}, "
            f"timeout={timeout_tabu}ms, seed={seed})..."
        )
        t0 = time.time()
        ss_tabu = solve_with_tabu(bqm, num_restarts_tabu, tenure, timeout_tabu, seed)
        logger.info(f"Tabu search finished in {time.time() - t0:.3f}s")
        samplesets.append(("tabu", ss_tabu))
        energies_by_sampler["tabu"] = np.array(ss_tabu.record.energy, dtype=float)

    if not samplesets or all(len(ss) == 0 for _, ss in samplesets):
        raise RuntimeError(
            "No samples were returned by the configured sampler(s). Increase 'num_reads_sa' / "
            "'num_restarts_tabu' or check the 'sampler' solver_param."
        )

    # --- Collect all candidate samples, ascending energy ---
    candidates: List[Tuple[float, str, Dict[str, int]]] = []
    for name, ss in samplesets:
        for record in ss.data(fields=["sample", "energy"], sorted_by="energy"):
            candidates.append((float(record.energy), name, dict(record.sample)))
    candidates.sort(key=lambda t: t[0])

    # --- Decode and verify independently (C1–C10) ---
    best_assign: Optional[List[Optional[int]]] = None
    best_report: Optional[Dict[str, str]] = None
    best_energy: Optional[float] = None
    best_sampler: Optional[str] = None
    repaired = False

    for energy, name, sample in candidates:
        decoded = decode_sample(sample, p, var_of)
        report, feasible = verify_constraints(decoded, p)
        if feasible:
            assign, _ = split_marker(decoded)
            best_assign, best_report, best_energy, best_sampler = assign, report, energy, name
            break

    if best_assign is None:
        logger.info(
            "No sampled configuration satisfied all constraints C1–C10 "
            f"({len(candidates)} candidates inspected)."
        )
        if repair_solution:
            logger.info("Applying deterministic greedy repair to the best-energy sample...")
            warm, _ = split_marker(decode_sample(candidates[0][2], p, var_of))
            repaired_assign = greedy_repair(p, warm)
            if repaired_assign is not None:
                report, feasible = verify_constraints(repaired_assign, p)
                if feasible:
                    best_assign = repaired_assign
                    best_report = report
                    best_energy = float(candidates[0][0])
                    best_sampler = "greedy_repair"
                    repaired = True

    if best_assign is None or best_report is None:
        raise RuntimeError(
            "no valid squad exists with the current data and parameters: neither the sampled "
            "QUBO configurations nor the deterministic repair heuristic satisfied constraints "
            "C1–C10. Try increasing 'num_reads_sa'/'num_sweeps_sa'/'num_restarts_tabu', raising "
            "'penalty_scale', or relaxing W_max / B0 / L_k / U_k / rho_k / Sigma_max / Delta_max. "
            f"Parameters used: N={p.N}, W_max={p.W_max}, B0={p.B0}, L_k={p.L_k}, U_k={p.U_k}, "
            f"rho_k={p.rho_k}, Sigma_max={p.Sigma_max}, Delta_max={p.Delta_max}."
        )

    # --- Output assembly and benchmark metrics ---
    out = compute_outputs(best_assign, p)
    Q0_bar, Q0_bar_k, B0_set = compute_baseline(p)
    out["Q0_bar"] = Q0_bar
    out["Q0_bar_k"] = Q0_bar_k
    out["Gamma"] = float(out["Q_bar"] - Q0_bar)
    out["Gamma_pct"] = float(100.0 * (out["Q_bar"] - Q0_bar) / Q0_bar) if Q0_bar > 0 else 0.0
    out["B0_set"] = B0_set
    out["V"] = best_report
    out["Status"] = "optimal squad found"

    elapsed = time.time() - start_time
    logger.info(
        f"Best feasible solution from '{best_sampler}' (energy={best_energy:.3f}"
        f"{', repaired' if repaired else ''}): F_x={out['F_x']:.3f}, Q_bar={out['Q_bar']:.3f}, "
        f"Gamma_pct={out['Gamma_pct']:.3f}%"
    )
    logger.info(
        f"Squad: {len(out['F'])} players | signings={out['n_sign']} sales={out['n_sale']} | "
        f"W_tot={out['W_tot']:.0f} (rem {out['W_rem']:.0f}) | "
        f"C_spend={out['C_spend']:.0f} (rem {out['B_rem']:.0f})"
    )
    logger.info(f"Solver finished in {elapsed:.3f}s")

    # --- Diagnostics not part of the registered output schema ---
    metadata = {
        "benchmark_metrics": {
            "F_x": out["F_x"],
            "Q_bar": out["Q_bar"],
            "Gamma_pct": out["Gamma_pct"],
        },
        "solver_metadata": {
            "sampler_selection": sampler_choice,
            "winning_source": best_sampler,
            "repaired": repaired,
            "best_energy": best_energy,
            "num_candidates_inspected": len(candidates),
            "num_reads_sa": num_reads_sa,
            "num_sweeps_sa": num_sweeps_sa,
            "num_restarts_tabu": num_restarts_tabu,
            "tenure": tenure,
            "timeout_tabu": timeout_tabu,
            "seed": seed,
            "penalty_scale": penalty_scale,
            "elapsed_seconds": round(elapsed, 3),
        },
        "model_size": {
            "num_binary_variables": int(len(bqm.variables)),
            "num_assignment_variables": int(len(pairs)),
            "num_quadratic_terms": int(len(bqm.quadratic)),
            "slack_bits": slack_bits,
            "penalty_weights": {k: float(v) for k, v in lam.items()},
        },
        "assignment_x_star": {
            f"{i}": (POSITIONS[best_assign[i]] if best_assign[i] is not None else None)
            for i in range(p.P)
        },
        "reference_group_sizes": {POSITIONS[k]: int(len(p.R[k])) for k in range(K)},
        "input_feasibility_warnings": input_warnings,
        "verification_report": best_report,
    }

    saved_files: List[str] = []
    if generate_plots:
        saved_files.append(save_squad_composition_plot(ADDITIONAL_OUTPUT_DIR, p, out))
        saved_files.append(save_quality_breakdown_plot(ADDITIONAL_OUTPUT_DIR, out))
        saved_files.append(save_energy_distribution_plot(ADDITIONAL_OUTPUT_DIR, energies_by_sampler))
        saved_files.append(
            save_qubo_matrix_plot(ADDITIONAL_OUTPUT_DIR, qubo_matrix_from_bqm(bqm, pairs, var_of))
        )
        saved_files.append(save_budget_usage_plot(ADDITIONAL_OUTPUT_DIR, p, out))
        saved_files.append(save_metadata_json(ADDITIONAL_OUTPUT_DIR, metadata))
        logger.info(f"Saved {len(saved_files)} file(s) to '{ADDITIONAL_OUTPUT_DIR}/'")
    else:
        saved_files.append(save_metadata_json(ADDITIONAL_OUTPUT_DIR, metadata))

    # --- Return exactly the registered output schema ---
    result = {
        "F": out["F"],
        "A": out["A"],
        "D": out["D"],
        "N_set": out["N_set"],
        "k_i": out["k_i"],
        "q_perf": out["q_perf"],
        "q_pot": out["q_pot"],
        "Q": out["Q"],
        "n_k": out["n_k"],
        "n_new_k": out["n_new_k"],
        "n_sign": out["n_sign"],
        "n_sale": out["n_sale"],
        "R_sale": out["R_sale"],
        "B_avail": out["B_avail"],
        "C_spend": out["C_spend"],
        "B_rem": out["B_rem"],
        "W_tot": out["W_tot"],
        "W_rem": out["W_rem"],
        "F_x": out["F_x"],
        "Q_bar": out["Q_bar"],
        "Q_bar_k": out["Q_bar_k"],
        "q_perf_bar": out["q_perf_bar"],
        "q_pot_bar": out["q_pot_bar"],
        "Q0_bar": out["Q0_bar"],
        "Q0_bar_k": out["Q0_bar_k"],
        "Gamma": out["Gamma"],
        "Gamma_pct": out["Gamma_pct"],
        "B0_set": out["B0_set"],
        "V": out["V"],
        "Status": out["Status"],
    }
    logger.info("Ending Solver...")
    return result
