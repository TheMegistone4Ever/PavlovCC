from typing import Any, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]

N = 30
np.random.seed(1810)

Y_MIN = np.asarray(np.random.randint(5, 25, N), dtype=float)
ALPHA = np.asarray(np.random.uniform(.5, 2.5, N), dtype=float)
D = np.asarray(Y_MIN * ALPHA * np.random.uniform(.95, 1.10, N), dtype=float)
PROFIT = np.asarray(np.random.uniform(2, 10, N), dtype=float)
TARGET_PROFIT = float(np.sum(Y_MIN * PROFIT) * 1.15)

A_RES = np.asarray(np.random.uniform(.5, 4., (3, N)), dtype=float)
B_RES = np.asarray(np.sum(A_RES * Y_MIN, axis=1) * 1.15, dtype=float)

C_PENALTY = np.asarray(np.random.uniform(.5, 8., N), dtype=float)
D_PENALTY = np.asarray(np.random.uniform(.1, 1., N), dtype=float)


def objective(x: FloatArray) -> float:
    x = np.asarray(x, dtype=float)
    z = x[N:]
    return float(np.sum(C_PENALTY * (z ** 2) * np.log1p(z) + D_PENALTY * z))


def jacobian(x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=float)
    z = x[N:]
    jac = np.zeros(2 * N, dtype=float)
    # Похідна для C_i * z^2 * ln(z+1) + D_i * z
    jac[N:] = C_PENALTY * (2. * z * np.log1p(z) + (z ** 2) / (z + 1.)) + D_PENALTY
    return jac


# Обмеження та межі (такі самі, бо вони лінійні)
bounds = list(zip(Y_MIN.tolist(), [None] * N)) + [(0., None)] * N


def constr_res(x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=float)
    return B_RES - (A_RES @ x[:N])


def constr_jac_res(_x: FloatArray) -> FloatArray:
    jac = np.zeros((3, 2 * N), dtype=float)
    jac[:, :N] = -A_RES
    return jac


def constr_prof(x: FloatArray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(PROFIT * x[:N]) - TARGET_PROFIT)


def constr_jac_prof(_x: FloatArray) -> FloatArray:
    jac = np.zeros(2 * N, dtype=float)
    jac[:N] = PROFIT
    return jac


def constr_delay(x: FloatArray) -> FloatArray:
    x = np.asarray(x, dtype=float)
    return D - ALPHA * x[:N] + x[N:]


def constr_jac_delay(_x: FloatArray) -> FloatArray:
    jac = np.zeros((N, 2 * N), dtype=float)
    np.fill_diagonal(jac[:, :N], -ALPHA)
    np.fill_diagonal(jac[:, N:], 1.)
    return jac


constraints: Sequence[dict[str, Any]] = [
    {"type": "ineq", "fun": constr_res, "jac": constr_jac_res},
    {"type": "ineq", "fun": constr_prof, "jac": constr_jac_prof},
    {"type": "ineq", "fun": constr_delay, "jac": constr_jac_delay},
]

y0 = Y_MIN * 1.15
z0 = np.maximum(0, ALPHA * y0 - D)
x0 = np.concatenate([y0, z0])

if __name__ == "__main__":
    minimize_fn = cast(Any, minimize)
    res = minimize_fn(
        objective, x0, method="SLSQP", jac=jacobian,
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 3000, "disp": False}
    )

    print("-" * 50)
    print(f"Справжній глобальний мінімум (для z^2): {res.fun:.4f}")
    print("-" * 50)
