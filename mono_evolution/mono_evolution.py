from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, cast

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Problem parameters  (N = 30 products, fixed seed for reproducibility)
# ---------------------------------------------------------------------------

N = 30
_RNG_SEED = 1810
np.random.seed(_RNG_SEED)

FloatArray = NDArray[np.float64]

Y_MIN: FloatArray = np.random.randint(5, 25, N).astype(float)
ALPHA: FloatArray = np.random.uniform(.5, 2.5, N).astype(float)
D: FloatArray = Y_MIN * ALPHA * np.random.uniform(.95, 1.10, N).astype(float)
PROFIT: FloatArray = np.random.uniform(2, 10, N).astype(float)
TARGET_PROFIT: float = float(np.sum(Y_MIN * PROFIT) * 1.15)

A_RES: FloatArray = np.random.uniform(.5, 4.0, (3, N)).astype(float)
B_RES: FloatArray = (np.sum(A_RES * Y_MIN, axis=1) * 1.15).astype(float)

C_PENALTY: FloatArray = np.random.uniform(.5, 8., N).astype(float)
D_PENALTY: FloatArray = np.random.uniform(.1, 1., N).astype(float)


# ---------------------------------------------------------------------------
# Pre-built constraint matrices (built once, reused everywhere)
# ---------------------------------------------------------------------------

def _build_constraints() -> tuple[FloatArray, FloatArray]:
    """Build the full (A_ub, b_ub) pair for the surrogate LP.

    Returns:
        A tuple (A_ub, b_ub) ready to pass to scipy.optimize.linprog.
    """
    rows: list[FloatArray] = []
    rhs: list[float] = []

    for i in range(A_RES.shape[0]):
        row = np.zeros(2 * N, dtype=float)
        row[:N] = A_RES[i]
        rows.append(row)
        rhs.append(float(B_RES[i]))

    for i in range(N):
        row = np.zeros(2 * N, dtype=float)
        row[i] = ALPHA[i]
        row[N + i] = -1.
        rows.append(row)
        rhs.append(float(D[i]))

    for i in range(N):
        row = np.zeros(2 * N, dtype=float)
        row[i] = -1.
        rows.append(row)
        rhs.append(float(-Y_MIN[i]))

    row = np.zeros(2 * N, dtype=float)
    row[:N] = -PROFIT
    rows.append(row)
    rhs.append(-TARGET_PROFIT)

    return np.array(rows, dtype=float), np.array(rhs, dtype=float)


_A_UB, _B_UB = _build_constraints()
_BOUNDS: list[tuple[float, Optional[float]]] = [(0., None)] * (2 * N)

# Pre-built SLSQP constraints and bounds (reused in lagrange_refine)
_SLSQP_CONSTRAINTS: list[LinearConstraint] = [
    LinearConstraint(A_RES, -np.inf, B_RES),
    LinearConstraint(PROFIT.reshape(1, -1), np.array([TARGET_PROFIT], dtype=float), np.array([np.inf], dtype=float)),
]
_SLSQP_BOUNDS = Bounds(lb=Y_MIN, ub=np.full(N, np.inf))


# ---------------------------------------------------------------------------
# Result data-classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LpSolution:
    """Optimal solution of the surrogate LP for a given weight vector.

    Attributes:
        y: Production volumes (length N).
        z: Delay variables (length N).
        f_vals: Per-product nonlinear penalties (length N).
        total_penalty: Sum of f_vals — the true nonlinear objective value.
    """
    y: FloatArray
    z: FloatArray
    f_vals: FloatArray
    total_penalty: float


@dataclass(frozen=True)
class Individual:
    """One individual in the evolutionary population.

    Attributes:
        omega: Weight vector (length N).
        fitness: Value of the nonlinear objective F(omega).
    """
    omega: FloatArray
    fitness: float


@dataclass(frozen=True)
class EvolutionResult:
    """Full result returned by evolutionary_programming.

    Attributes:
        best_omega: Best weight vector found.
        best_solution: LP solution evaluated at best_omega.
        history_best: Best fitness value per generation.
        history_mean: Mean fitness value per generation.
        history_omega: Best omega vector per generation (shape: generations x N).
        history_lagrange_wins: How many times SLSQP beat mono-evo per generation.
    """
    best_omega: FloatArray
    best_solution: LpSolution
    history_best: list[float]
    history_mean: list[float]
    history_omega: np.ndarray
    history_lagrange_wins: list[int]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def nonlinear_penalty(z: FloatArray) -> FloatArray:
    """Compute the convex nonlinear penalty f_i(z_i) = C_i*z_i^2*ln(z_i+1) + D_i*z_i.

    Args:
        z: Delay vector of length N.

    Returns:
        Per-product penalty values, shape (N, ).
    """
    return C_PENALTY * (z ** 2) * np.log1p(z) + D_PENALTY * z


def solve_surrogate_lp(omega: FloatArray) -> Optional[LpSolution]:
    """Solve the linear surrogate LP for fixed weight vector omega.

    Minimises sum(omega_i * z_i) subject to the pre-built constraints.
    After solving, evaluates the *true* nonlinear penalty on the result.

    Args:
        omega: Positive weight vector of length N.

    Returns:
        An LpSolution if the LP is feasible/optimal, otherwise None.
    """
    c = np.concatenate([np.zeros(N, dtype=float), omega])
    # noinspection PyDeprecation
    res = linprog(c, A_ub=_A_UB, b_ub=_B_UB, bounds=_BOUNDS, method="highs")  # type: ignore[deprecated]
    if res.status != 0 or res.x is None:
        return None
    y = res.x[:N]
    z = res.x[N:]
    f_vals = nonlinear_penalty(cast(FloatArray, z))
    return LpSolution(
        y=cast(FloatArray, y),
        z=cast(FloatArray, z),
        f_vals=f_vals,
        total_penalty=float(np.sum(f_vals)),
    )


# ---------------------------------------------------------------------------
# Section 6 — mono-evolutionary operation
# ---------------------------------------------------------------------------

def mono_evolutionary_operation(
        omega_start: FloatArray,
        beta1: float = .6,
        beta2: float = .15,
        n_iters: int = 10,
) -> Individual:
    """Artificial mono-evolutionary operation (Section 6, formulae 24–31).

    Iteratively adjusts omega so that products whose nonlinear penalty exceeds
    the average receive higher weights (formula 29), while products below the
    average receive lower weights (formula 31).  The final step replaces omega
    with the derivative vector (last operation described on page 9).

    Args:
        omega_start: Initial weight vector of length N. All entries > 0.
        beta1: Amplification coefficient, beta1 in [.5, 1].
        beta2: Damping coefficient, beta2 in [0, .5].
        n_iters: Number of iterative refinement steps (parameter N in the paper).

    Returns:
        The best Individual found across all iterations.
    """
    best_omega = cast(FloatArray, omega_start.copy())
    sol = solve_surrogate_lp(omega_start)
    if sol is None:
        return Individual(omega=best_omega, fitness=float("inf"))

    best_fitness = sol.total_penalty
    best_z = sol.z.copy()
    current_omega = cast(FloatArray, omega_start.copy())

    for _ in range(n_iters):
        f_vals = sol.f_vals
        avg = float(np.mean(f_vals))
        max_f = float(np.max(f_vals))
        if max_f < 1e-9:
            break

        next_omega = np.empty(N, dtype=float)
        for i in range(N):
            if f_vals[i] > avg:
                gamma_i = f_vals[i] / max_f
                next_omega[i] = (1. + beta1 * gamma_i) * current_omega[i]
            else:
                factor = (1. - beta2 * (1. - f_vals[i] / avg)) if avg > 0 else 1.
                next_omega[i] = factor * current_omega[i]

        current_omega = cast(FloatArray, np.clip(next_omega, 1e-4, 1e4))
        sol = solve_surrogate_lp(current_omega)
        if sol is None:
            break
        if sol.total_penalty < best_fitness:
            best_fitness = sol.total_penalty
            best_omega = cast(FloatArray, current_omega.copy())
            best_z = sol.z.copy()

    # Final derivative step (page 9): omega_N = df_i/dz_i at z*
    deriv_term = 2. * best_z * np.log1p(best_z) + (best_z ** 2) / (best_z + 1.)
    deriv_omega = cast(FloatArray, np.clip(C_PENALTY * deriv_term + D_PENALTY, 1e-4, 1e4))
    deriv_sol = solve_surrogate_lp(deriv_omega)
    if deriv_sol is not None and deriv_sol.total_penalty < best_fitness:
        best_fitness = deriv_sol.total_penalty
        best_omega = cast(FloatArray, np.asarray(deriv_omega, dtype=float).copy())

    return Individual(omega=best_omega, fitness=best_fitness)


# ---------------------------------------------------------------------------
# SLSQP local search — exact NLP solve
# ---------------------------------------------------------------------------

def lagrange_refine(ind: Individual) -> Individual:
    """Refine an individual by solving the exact nonlinear problem via SLSQP local search.

    The mono-evolutionary operation works in *omega-space* (surrogate LP).
    This function takes the LP solution of *ind* as a warm start and solves
    the true nonlinear programme directly over the production volumes y,
    using the KKT / Lagrange-multiplier optimality conditions enforced by
    SLSQP.

    The nonlinear objective is:
        F(y) = Σ_i C_i · z_i^2 · ln(z_i + 1) + D_i · z_i,
    where z_i = max (0, α_i · y_i − D_i).

    Because F is convex and the constraints are linear, every local minimum
    found by SLSQP is a global minimum in its neighbourhood.

    After finding the optimal y*, the clone’s omega is set to the gradient
    ∂F/∂z_i |_{z*}, which is the theoretically optimal surrogate weight
    (same "derivative trick" used at the end of mono_evolutionary_operation).

    Args:
        ind: Individual whose LP solution is used as the warm-start point.

    Returns:
        A new Individual that is the better of the SLSQP clone and *ind*.
    """
    # Warm-start: use the LP solution of the incoming individual
    sol = solve_surrogate_lp(ind.omega)
    if sol is None:
        return ind

    y0 = sol.y.copy()

    # --- objective and analytic gradient ---

    def _obj(y: FloatArray) -> float:
        z = np.maximum(0., ALPHA * y - D)
        return float(np.sum(C_PENALTY * z ** 2 * np.log1p(z) + D_PENALTY * z))

    def _grad(y: FloatArray) -> FloatArray:
        z = np.maximum(0., ALPHA * y - D)
        active = (ALPHA * y > D).astype(float)  # sub-gradient at 0
        df_dz = C_PENALTY * (2. * z * np.log1p(z) + z ** 2 / (z + 1.)) + D_PENALTY
        return active * df_dz * ALPHA

    # --- solve ---
    minimize_fn = cast(Any, minimize)
    res = minimize_fn(
        _obj,
        y0,
        jac=_grad,
        method="SLSQP",
        bounds=_SLSQP_BOUNDS,
        constraints=_SLSQP_CONSTRAINTS,
        options={"ftol": 1e-10, "maxiter": 500},
    )

    # Guard: reject if SLSQP failed or found something worse
    if not res.success or res.fun is None or not np.isfinite(res.fun) or res.fun >= ind.fitness:
        return ind

    # Convert optimal y* → omega via ∂F/∂z_i |_{z*}
    best_y = cast(FloatArray, res.x)
    best_z = np.maximum(0., ALPHA * best_y - D)
    deriv_term = 2. * best_z * np.log1p(best_z) + best_z ** 2 / (best_z + 1.)
    omega_clone = np.clip(C_PENALTY * deriv_term + D_PENALTY, 1e-4, 1e4)

    return Individual(omega=omega_clone, fitness=float(res.fun))


# ---------------------------------------------------------------------------
# Section 7 — modified evolutionary programming algorithm
# ---------------------------------------------------------------------------

def evolutionary_programming(
        k1: int = 25,
        k2: int = 5,
        k4: int = 5,
        n_generations: int = 40,
) -> EvolutionResult:
    """Modified evolutionary programming algorithm (Section 7).

    Population size is fixed at k1.  Each generation:
      1. Select k2 elite individuals (Section 7, point 3).
      2. Produce k3 = k1 - k2 offspring by crossover (arithmetic mean).
      3. For k4 of those offspring (≈30 %):
           1) Apply mono-evolutionary operation (no mutation).
           2) Also run SLSQP local search.
           3) Keep whichever individual has the lower fitness.
      4. Mutate the remaining k3 - k4 offspring (±alpha * omega), then apply
          a mono-evolutionary operation to each (Section 7, point 4).
         No SLSQP step here — mutation already diversifies sufficiently.
      5. New population = elite and offspring.

    Args:
        k1: Total population size.
        k2: Number of elite individuals carried forward unchanged.
        k4: Number of crossover offspring sent to the dual mono-evo / SLSQP path.
            Must satisfy k4 < k1 - k2.
        n_generations: Number of generations to run.

    Returns:
        An EvolutionResult with the best individual and full convergence history.
    """
    k3 = k1 - k2
    logger.info("Initialising population (%d individuals)...", k1)

    # Section 7, point 2: generate initial population
    population: list[Individual] = []
    for _ in range(k1):
        w_init = np.random.uniform(.8, 1.2, N).astype(float)
        ind_init = mono_evolutionary_operation(cast(FloatArray, w_init))
        population.append(ind_init)

    history_best: list[float] = []
    history_mean: list[float] = []
    history_omega: list[np.ndarray] = []
    history_lagrange_wins: list[int] = []

    for gen in range(n_generations):
        # Section 7, point 3: selection
        population.sort(key=lambda ind: ind.fitness)
        elite = population[:k2]

        history_best.append(elite[0].fitness)
        history_mean.append(float(np.mean([ind.fitness for ind in population])))
        history_omega.append(elite[0].omega.copy())

        # Crossover: arithmetic mean of two randomly chosen parents
        crossed_omegas: list[FloatArray] = []
        for _ in range(k3):
            idx1, idx2 = np.random.choice(k1, size=2, replace=False)
            w_crossed = (population[idx1].omega + population[idx2].omega) / 2.
            crossed_omegas.append(cast(FloatArray, w_crossed))

        # Split offspring into k4 (mono-evo + SLSQP local search) and k3-k4 (mutate + mono-evo)
        chosen = np.atleast_1d(np.random.choice(k3, size=k4, replace=False))
        indices_k4 = set(cast(np.ndarray, chosen).tolist())
        offspring: list[Individual] = []
        lagrange_wins_this_gen = 0

        for idx, w in enumerate(crossed_omegas):
            if idx in indices_k4:
                # ── dual path ────────────────────────────────────────────
                # Step A: standard mono-evolutionary operation
                ind_mono = mono_evolutionary_operation(cast(FloatArray, w))

                # Step B: SLSQP local search — exact NLP solve
                ind_lagrange = lagrange_refine(ind_mono)

                # Step C: keep the better individual
                if ind_lagrange.fitness < ind_mono.fitness:
                    offspring_ind = ind_lagrange
                    lagrange_wins_this_gen += 1
                else:
                    offspring_ind = ind_mono
                # ─────────────────────────────────────────────────────────
            else:
                # Section 7, point 4: mutation ±alpha*omega, then mono-evo only
                signs = np.random.choice([-1., 1.], N)
                alpha = np.random.uniform(.05, .95)  # random step per offspring
                w_mutated = cast(FloatArray, np.maximum(w + signs * alpha * w, 1e-3))
                offspring_ind = mono_evolutionary_operation(cast(FloatArray, w_mutated))

            offspring.append(offspring_ind)

        history_lagrange_wins.append(lagrange_wins_this_gen)
        population = list(elite) + offspring

        logger.info(
            "Generation %2d | best=%.2f | mean=%.2f | "
            "omega in [%.4f, %.4f] | SLSQP wins=%d/%d",
            gen + 1,
            history_best[-1],
            history_mean[-1],
            float(np.min(elite[0].omega)),
            float(np.max(elite[0].omega)),
            lagrange_wins_this_gen,
            k4,
        )

    # Final generation bookkeeping
    population.sort(key=lambda ind: ind.fitness)
    history_best.append(population[0].fitness)
    history_mean.append(float(np.mean([ind.fitness for ind in population])))
    history_omega.append(population[0].omega.copy())
    history_lagrange_wins.append(0)  # no SLSQP in final bookkeeping step

    best_ind = population[0]
    best_sol = solve_surrogate_lp(best_ind.omega)
    if best_sol is None:
        raise RuntimeError("Best individual produced an infeasible LP solution.")

    return EvolutionResult(
        best_omega=best_ind.omega,
        best_solution=best_sol,
        history_best=history_best,
        history_mean=history_mean,
        history_omega=np.array(history_omega),
        history_lagrange_wins=history_lagrange_wins,
    )


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_inputs() -> None:
    """Log all problem inputs when the application starts."""
    logger.info("N=%d  seed=%d  TARGET_PROFIT=%.4f", N, _RNG_SEED, TARGET_PROFIT)
    for name, vec in [
        ("Y_MIN", Y_MIN), ("ALPHA", ALPHA), ("D", D), ("PROFIT", PROFIT),
        ("B_RES", B_RES), ("C_PENALTY", C_PENALTY), ("D_PENALTY", D_PENALTY),
    ]:
        logger.info("%s = %s", name, np.array2string(vec, precision=3, separator=", "))
    logger.info("A_RES =\n%s", np.array2string(A_RES, precision=3, separator=", "))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_dashboard(
        base_sol: LpSolution,
        result: EvolutionResult,
) -> None:
    """Render a five-panel result dashboard and save it to disk.

    Args:
        base_sol: Baseline LP solution with uniform weights.
        result: Full evolution results from evolutionary_programming.
    """
    opt_sol = result.best_solution
    hist_f = result.history_best
    hist_mean = result.history_mean
    hist_w = result.history_omega

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "Modified Evolutionary Algorithm + SLSQP Local Search — N=30",
        fontsize=15, fontweight="bold", y=.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=.38, wspace=.32)

    # Panel 1: convergence
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(hist_f, color="#e31a1c", linewidth=2.5, marker="o", markersize=4,
             label="Best fitness")
    ax1.plot(hist_mean, color="#1f78b4", linewidth=2., marker="s", markersize=3,
             label="Mean fitness")
    ax1.axhline(base_sol.total_penalty, color="black", linestyle="--", alpha=.5,
                label="Baseline (ω=1)")
    ax1.set_title("1. Convergence", fontweight="bold")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Sum of nonlinear penalties")
    ax1.grid(True, linestyle=":", alpha=.7)
    ax1.legend(fontsize=8)

    # Panel 2: weight heatmap
    ax2 = fig.add_subplot(gs[0, 1:3])
    w_norm = hist_w / hist_w.max(axis=1, keepdims=True)
    cax = ax2.imshow(w_norm.T, aspect="auto", cmap="viridis", origin="lower")
    ax2.set_title("2. Weight vector evolution (normalised heatmap)", fontweight="bold")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Product index")
    fig.colorbar(cax, ax=ax2, label="Relative weight (lighter = higher)")

    # Panel 3: delay comparison
    ax3 = fig.add_subplot(gs[1, 0:2])
    x = np.arange(N)
    ax3.bar(x - .2, base_sol.z, .4, label="Baseline", color="gray", alpha=.5)
    ax3.bar(x + .2, opt_sol.z, .4, label="After EA+SLSQP", color="#1f78b4")
    ax3.set_title("3. Production delays z_i per product", fontweight="bold")
    ax3.set_xlabel("Product i")
    ax3.set_ylabel("Delay (hours)")
    ax3.set_xticks(x[::2])
    ax3.grid(True, axis="y", alpha=.3)
    ax3.legend(fontsize=8)

    # Panel 4: penalty coefficient vs delay scatter
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.scatter(C_PENALTY, opt_sol.z, color="#33a02c", s=70, alpha=.8,
                edgecolor="black")
    ax4.set_title("4. EA+SLSQP strategy: penalty vs delay", fontweight="bold")
    ax4.set_xlabel("Nonlinear penalty coefficient C_i")
    ax4.set_ylabel("Delay z_i at optimum")
    ax4.grid(True, linestyle=":", alpha=.7)

    plt.subplots_adjust(top=.90, bottom=.08)
    plt.savefig("ea_lagrange_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full pipeline: baseline → EA+SLSQP → visualisation."""
    logger.info("=" * 65)
    logger.info(" Aggregate Volume-Time Model — First Nonlinear AVTM")
    logger.info(" (with SLSQP local search)")
    logger.info("=" * 65)
    _log_inputs()

    base_sol = solve_surrogate_lp(np.ones(N, dtype=float))
    if base_sol is None:
        raise RuntimeError("Baseline LP is infeasible.")
    logger.info("Baseline penalty (omega=1): %.2f", base_sol.total_penalty)

    # --- quick sanity check: pure SLSQP lower bound ---
    lb_ind = lagrange_refine(Individual(omega=np.ones(N, dtype=float), fitness=base_sol.total_penalty))
    logger.info("Pure SLSQP local search lower bound (single run): %.2f", lb_ind.fitness)

    result = evolutionary_programming(k1=15, k2=5, k4=5, n_generations=50)
    opt_sol = result.best_solution

    # Use the best recorded fitness for reporting to avoid any LP/fitness mismatch.
    best_fitness = float(result.history_best[-1])
    if not np.isclose(opt_sol.total_penalty, best_fitness, rtol=1e-6, atol=1e-8):
        logger.warning(
            "Best fitness (%.2f) differs from best LP penalty (%.2f); "
            "reporting the best fitness for improvement.",
            best_fitness,
            opt_sol.total_penalty,
        )

    logger.info("Final optimised penalty: %.2f", best_fitness)
    improvement = (
            (base_sol.total_penalty - best_fitness) / base_sol.total_penalty * 100
    )
    logger.info("Improvement over baseline: %.1f%%", improvement)
    logger.info(
        "Summary: omega in [%.4f, %.4f], z in [%.4f, %.4f]",
        float(np.min(result.best_omega)),
        float(np.max(result.best_omega)),
        float(np.min(opt_sol.z)),
        float(np.max(opt_sol.z)),
    )
    total_slsqp_wins = sum(result.history_lagrange_wins)
    total_slsqp_calls = len(result.history_lagrange_wins) * 5  # k4=5
    logger.info(
        "SLSQP local search wins over mono-evo: %d / %d (%.1f%%)",
        total_slsqp_wins, total_slsqp_calls,
        100. * total_slsqp_wins / max(total_slsqp_calls, 1),
    )

    plot_dashboard(base_sol, result)


if __name__ == "__main__":
    main()
