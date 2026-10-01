# Football Team Creator — QUBO Squad Search (Tabu + Simulated Annealing)

Quantum-inspired solver for the **Football Team Creator** problem: decide which current
players to retain, which to sell, which transfer candidates to sign, and which single
eligible position each squad member occupies, so that **total squad quality is maximized**
under salary, transfer-budget, squad-size, position-composition, priority-reinforcement and
transaction-count rules.

The Binary Integer Linear Program is reformulated as a **QUBO** (penalty terms + binary
slacks) and sampled locally with D-Wave Ocean's `neal` (simulated annealing) and
`dwave-tabu` (tabu search) — the same binary quadratic model that could later be submitted
to an annealing QPU, but fully local and inexpensive.

## Problem

Maximize

```
F(x) = Σ_i Σ_{k∈E_i} Q_{i,k} · x_{i,k},    Q_{i,k} = α·q_perf_{i,k} + (1-α)·q_pot_i
```

subject to

| ID | Constraint |
|---|---|
| C1 | `Σ_{k∈E_i} x_{i,k} ≤ 1` and `x_{i,k}=0` for `k ∉ E_i` (single eligible assignment) |
| C2 | `Σ w_i u_i ≤ W_max` (salary cap) |
| C3 | `Σ x_{i,k} = N` (exact squad size) |
| C4/C5 | `L_k ≤ Σ_{i∈R_k} x_{i,k} ≤ U_k` (position composition) |
| C6 | `Σ_{i∈T} c_i u_i ≤ B0 + Σ_{i∈C} v_i (1-u_i)` (transfer budget incl. sale revenue) |
| C7 | `Σ_{i∈T∩R_k} x_{i,k} ≥ ρ_k` for priority positions |
| C8 | `Σ_{i∈T} u_i ≤ Σ_max` (max signings) |
| C9 | `Σ_{i∈C} (1-u_i) ≤ Δ_max` (max sales) |
| C10 | `x_{i,k} ∈ {0,1}` |

Pre-processing (done once, outside the optimization): per-90 normalization
`z_{i,m} = 90·y_{i,m}/μ_i`, mid-rank percentile scores `φ_{i,k,m}` inside each reference
group `R_k`, the unweighted position average `q_perf_{i,k}`, and the age-based potential
`q_pot_i = 100·clip((A_lo − a_i)/(A_lo − A_hi), 0, 1)`.

Eligibility `E_i` is built from the preferred position plus a fixed secondary-position
policy: `GK → {GK}`, `DF → {DF, MF}`, `MF → {MF, DF, FW}`, `FW → {FW, MF}` (no outfielder
is ever eligible for GK).

## Files

| File | Purpose |
|---|---|
| `qcentroid.py` | Mandatory solver entry point (`run()`), deployed to the platform |
| `app.py` | Local test harness — static, never edited; loads `input.json` and calls `run()` |
| `input.json` | Local fixture: `{"data": ..., "solver_params": ...}` — edit this to try scenarios |
| `requirements.txt` | Non-stdlib dependencies, pinned |
| `README.md` | This document |
| `pseudocode.txt` / `mermaid_flow_chart` | Walkthrough and flowchart of `run()` |
| `summary` | Short summary of the generated bundle |

## Input format (`input_data`)

```json
{
  "NC": 22, "NT": 50, "N": 23,
  "W_max": 60000000, "B0": 20000000,
  "L_k": [2, 5, 5, 3], "U_k": [3, 9, 9, 6], "rho_k": [0, 1, 0, 0],
  "Sigma_max": 6, "Delta_max": 6,
  "mu_min": 900, "alpha": 0.7, "A_hi": 20, "A_lo": 30,
  "ai": [...], "wi": [...], "vi": [...], "ci": [...],
  "mui": [...], "ri": [...], "pi": [...], "ti": [...], "gi": [...],
  "xi_xg": [...], "si": [...], "p_pref": ["GK", "DF", ...]
}
```

- Position-indexed arrays (`L_k`, `U_k`, `rho_k`) follow the fixed order **`[GK, DF, MF, FW]`**.
- Player-indexed arrays (`ai`, `wi`, `mui`, `ri`, `pi`, `ti`, `gi`, `xi_xg`, `si`, `p_pref`)
  have length `NC + NT`; indices `0 … NC-1` are current players, `NC … NC+NT-1` are candidates.
- `vi` has length `NC` (sale values) and `ci` has length `NT` (transfer fees).
- Unrecognized keys raise a `ValueError` instead of being silently ignored; missing required
  keys and length mismatches also raise.

## Solver parameters (`solver_params`)

| Key | Default | Description |
|---|---|---|
| `sampler` | `"both"` | Which local sampler(s) to run: `"sa"`, `"tabu"` or `"both"` (best feasible sample across all runs wins) |
| `num_reads_sa` | `400` | `neal` independent annealing reads — more reads, better chance of a feasible/high-quality squad |
| `num_sweeps_sa` | `3000` | `neal` sweeps per read (annealing length) |
| `num_restarts_tabu` | `50` | `TabuSampler` restarts (`num_restarts`) |
| `tenure` | `null` | Tabu tenure; `null` lets `dwave-tabu` pick its own default |
| `timeout_tabu` | `5000` | Tabu total wall-clock budget in milliseconds |
| `seed` | `42` | RNG seed shared by both samplers (`null` for a non-reproducible run) |
| `penalty_scale` | `2.0` | Multiplier on the penalty base `max(Q)·N`; raise it if the samplers return constraint-violating configurations |
| `repair_solution` | `true` | If no sampled configuration is feasible, run the deterministic greedy repair heuristic before giving up |
| `generate_plots` | `true` | Write visualizations to `additional_output/` |

Unrecognized keys raise a `ValueError`. An omitted-but-known key still defaults, and that
fallback is logged to `qcentroid-user-log` — check Execution logs (or stdout locally) to see
which defaults a job actually used.

## Output (`run()` return value)

Matches the registered output schema exactly:

```json
{
  "F": [0, 1, 3, ...], "A": [...], "D": [...], "N_set": [...],
  "k_i": ["GK", "DF", null, ...],
  "q_perf": [...], "q_pot": [...], "Q": [...],
  "n_k": [2, 7, 8, 6], "n_new_k": [0, 1, 0, 0],
  "n_sign": 3, "n_sale": 2,
  "R_sale": 0.0, "B_avail": 0.0, "C_spend": 0.0, "B_rem": 0.0,
  "W_tot": 0.0, "W_rem": 0.0,
  "F_x": 0.0, "Q_bar": 0.0, "Q_bar_k": [0, 0, 0, 0],
  "q_perf_bar": 0.0, "q_pot_bar": 0.0,
  "Q0_bar": 0.0, "Q0_bar_k": [0, 0, 0, 0],
  "Gamma": 0.0, "Gamma_pct": 0.0,
  "B0_set": ["..."],
  "V": {"C1_single_assignment_and_eligibility": "pass", "...": "pass"},
  "Status": "optimal squad found"
}
```

`k_i` has one entry per player in `P` (`null` for players outside the final squad), while
`q_perf`, `q_pot` and `Q` are listed in the order of `F`.

### Benchmarking metrics

| Metric | Goal | Where it appears |
|---|---|---|
| **`F_x`** | higher is better | Squad quality `F(x*) = Σ_{i∈F} Q_{i,k*_i}` — primary quality indicator |
| **`Q_bar`** | higher is better | `F_x / N`, comparable across squad sizes |
| **`Gamma_pct`** | higher is better | `100·(Q_bar − Q0_bar)/Q0_bar`, the business-facing improvement over the initial squad |

All three are returned under those exact keys, logged at the end of the run, written to
`additional_output/metadata.json` under `benchmark_metrics`, and annotated on the
`squad_composition.png` / `quality_breakdown.png` plots.

## Visualizations (`additional_output/`)

- `squad_composition.png` — `n_k` and `n_new_k` against `L_k`/`U_k`, titled with the three
  benchmark metrics (the direct view of the Output Data fields).
- `quality_breakdown.png` — per-player `q_perf` / `q_pot` / `Q` for the final squad.
- `energy_distribution.png` — energy histograms of the simulated-annealing and tabu samples.
- `qubo_matrix.png` — heatmap of the assignment-variable block of the QUBO.
- `budget_usage.png` — salary, transfer spend, signings, sales and squad size vs. their caps.
- `metadata.json` — benchmark metrics, solver metadata, model size, penalty weights, the full
  `x*` assignment, reference-group sizes and the verification report.
- `input_used.json` — always written (never gated), the resolved `data` + `solver_params`.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Edit `input.json` to try a different market or tuning — `app.py` never needs touching.

## Deploying to QCentroid

1. Register the use case with the input/output specs described above.
2. Push this repository to Git (do not rename `qcentroid.py` or change `run()`'s signature).
3. Connect the repo via SSH deploy key on the platform, then pull and build.
4. Launch jobs; inspect Execution logs and the Assets tab.

## Caveats

- Simulated annealing and tabu search are **heuristics**: `Status = "optimal squad found"`
  means a fully verified feasible squad was returned, not a proven optimum.
- Penalty-based QUBOs with large coefficient ranges (EUR salaries/fees) are numerically
  hard; money terms are rescaled to EUR millions and slacks are log-encoded, but if the
  samplers keep returning infeasible configurations, raise `penalty_scale`, raise the read
  budget, or rely on `repair_solution` (enabled by default).
- Every returned solution is independently re-verified against C1–C10 (`V` in the output);
  if nothing feasible can be produced, `run()` raises a `RuntimeError` describing the
  parameter set used.
