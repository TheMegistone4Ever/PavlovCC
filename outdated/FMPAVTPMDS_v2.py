import matplotlib.pyplot as plt
import numpy as np
from ortools.linear_solver import pywraplp

# Constants for data generation
NUM_RESOURCES = 4
NUM_PRODUCTS = 5
NUM_AGGREGATED_PRODUCTS = 3
NUM_LINEAR_FUNCTIONS = 5

# Seed for reproducibility
np.random.seed(1810)


def generate_data():
    """Generates random data for the optimization problem."""
    A = np.random.rand(NUM_RESOURCES, NUM_PRODUCTS)
    b = np.random.rand(NUM_RESOURCES) * 10000
    C = np.random.rand(NUM_LINEAR_FUNCTIONS, NUM_PRODUCTS) * 10
    f = np.random.rand(NUM_AGGREGATED_PRODUCTS)
    priorities = np.ones(NUM_PRODUCTS)  # Not used in the current code
    D = np.random.rand(NUM_AGGREGATED_PRODUCTS) * 100
    t_0 = np.arange(NUM_AGGREGATED_PRODUCTS)
    alpha = np.random.rand(NUM_AGGREGATED_PRODUCTS) * 2
    omega = np.random.rand(NUM_LINEAR_FUNCTIONS) * 0.1
    return A, b, C, f, D, t_0, alpha, omega


def print_data(A, b, C, f, D, t_0, alpha, omega):
    """Prints the generated data."""
    print("Production matrix (A):\n", A)
    print("\nResource limits (b):\n", b)
    print("\nCost coefficients (C):\n", C)
    print("\nAggregation coefficients (f):\n", f)
    print("\nDirective terms (D):\n", D)
    print("\nInitial completion times (t_0):\n", t_0)
    print("\nAlpha coefficients (alpha):\n", alpha)
    print("\nOmega coefficients (omega):\n", omega)


def solve_problem(A, b, C, f, D, t_0, alpha, omega):
    """Solves the linear optimization problem."""
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Variables
    y = [solver.NumVar(0, solver.infinity(), f'y_{i}') for i in range(NUM_PRODUCTS)]
    z = [solver.NumVar(0, solver.infinity(), f'z_{i}') for i in range(NUM_AGGREGATED_PRODUCTS)]

    # Constraints
    for i in range(NUM_RESOURCES):
        solver.Add(solver.Sum([A[i][j] * y[j] for j in range(NUM_PRODUCTS)]) <= b[i])

    for i in range(NUM_AGGREGATED_PRODUCTS):
        T_i = t_0[i] + alpha[i] * y[i]
        solver.Add(T_i - z[i] <= D[i])

    # Objective
    objective = solver.Objective()
    for l in range(NUM_LINEAR_FUNCTIONS):
        f_opt_l = find_f_opt(A, b, C[l], f, D, t_0, alpha)
        objective.SetCoefficient(y[l], omega[l] * C[l][l])
        for i in range(NUM_AGGREGATED_PRODUCTS):
            objective.SetCoefficient(z[i], -omega[l] * f[i])
        # Dummy variable (not affecting the solution)
        # dummy_variable = solver.NumVar(0, 0, f'dummy_{l}')
        # objective.SetCoefficient(dummy_variable, -omega[l] * f_opt_l)

    objective.SetMaximization()

    # Solve
    status = solver.Solve()
    return solver, y, z, status


def find_f_opt(A, b, c_l, f, D, t_0, alpha):
    """Calculates the optimal value of f for a given linear function."""
    solver_f_opt = pywraplp.Solver.CreateSolver('GLOP')

    # Variables
    y_f_opt = [solver_f_opt.NumVar(0, solver_f_opt.infinity(), f'y_{i}') for i in range(NUM_PRODUCTS)]
    z_f_opt = [solver_f_opt.NumVar(0, solver_f_opt.infinity(), f'z_{i}') for i in range(NUM_AGGREGATED_PRODUCTS)]

    # Constraints
    for i in range(NUM_RESOURCES):
        solver_f_opt.Add(solver_f_opt.Sum([A[i][j] * y_f_opt[j] for j in range(NUM_PRODUCTS)]) <= b[i])

    for i in range(NUM_AGGREGATED_PRODUCTS):
        T_i = t_0[i] + alpha[i] * y_f_opt[i]
        solver_f_opt.Add(T_i - z_f_opt[i] <= D[i])

    # Objective
    objective_f_opt = solver_f_opt.Objective()
    for i in range(NUM_PRODUCTS):
        objective_f_opt.SetCoefficient(y_f_opt[i], c_l[i])
    for i in range(NUM_AGGREGATED_PRODUCTS):
        objective_f_opt.SetCoefficient(z_f_opt[i], -f[i])

    objective_f_opt.SetMaximization()

    # Solve
    solver_f_opt.Solve()

    return objective_f_opt.Value()


def print_solution(solver, y, z, D, t_0, alpha, C, f):
    """Prints the solution of the optimization problem."""
    if solver.Solve() == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        print("\nProduction quantities (y):")
        for i in range(NUM_PRODUCTS):
            print(f'y_{i} = {y[i].solution_value()}')

        print("\nAggregated product completion times (z):")
        for i in range(NUM_AGGREGATED_PRODUCTS):
            print(f'z_{i} = {z[i].solution_value()}')

        print("\nPolicy deadlines (D):\n", D)
        print("\nCompletion dates (T):")
        for i in range(NUM_AGGREGATED_PRODUCTS):
            print(f'T_{i} = {t_0[i] + alpha[i] * y[i].solution_value()}')

        print("\nDifferences (D - T):")
        for i in range(NUM_AGGREGATED_PRODUCTS):
            print(f'D_{i} - T_{i} = {D[i] - (t_0[i] + alpha[i] * y[i].solution_value())}')

        print("\nComparison of f_optimum and f_solution:")
        diffs = []
        for l in range(NUM_LINEAR_FUNCTIONS):
            f_opt_l = find_f_opt(A, b, C[l], f, D, t_0, alpha)
            f_solution_l = C[l] @ np.array([y[i].solution_value() for i in range(NUM_PRODUCTS)]) - f @ np.array(
                [z[i].solution_value() for i in range(NUM_AGGREGATED_PRODUCTS)])
            diff = f_opt_l - f_solution_l
            diffs.append(diff)
            print(f'l={l}, f_optimum={f_opt_l:.2f}, f_solution={f_solution_l:.2f}, difference={diff:.2f}')

        # Plotting the results (optional)
        plot_results(C, f, y, z, diffs)
    else:
        print('The problem does not have an optimal solution.')


def plot_results(C, f, y, z, diffs):
    """Plots the comparison of f_optimum and f_solution."""
    fig, ax = plt.subplots()
    f_optimum_values = [find_f_opt(A, b, C[l], f, D, t_0, alpha) for l in range(NUM_LINEAR_FUNCTIONS)]
    f_solution_values = [C[l] @ np.array([y[i].solution_value() for i in range(NUM_PRODUCTS)]) - f @ np.array(
        [z[i].solution_value() for i in range(NUM_AGGREGATED_PRODUCTS)]) for l in range(NUM_LINEAR_FUNCTIONS)]
    ax.plot(range(NUM_LINEAR_FUNCTIONS), f_optimum_values, label='f_optimum')
    ax.plot(range(NUM_LINEAR_FUNCTIONS), f_solution_values, label='f_solution')
    ax.legend()
    plt.title("Comparison of f_optimum and f_solution")
    plt.show()

    fig, ax = plt.subplots()
    ax.bar(range(NUM_LINEAR_FUNCTIONS), diffs, color=['blue' if diff >= 0 else 'red' for diff in diffs])
    plt.title("Differences between f_optimum and f_solution")
    plt.show()


A, b, C, f, D, t_0, alpha, omega = generate_data()


def main():
    print_data(A, b, C, f, D, t_0, alpha, omega)
    solver, y, z, status = solve_problem(A, b, C, f, D, t_0, alpha, omega)
    if status == pywraplp.Solver.OPTIMAL:
        print_solution(solver, y, z, D, t_0, alpha, C, f)


def solve_and_return_solution(A, b, C, f, D, t_0, alpha, omega):
    # Solve the problem
    solver, y, z, status = solve_problem(A, b, C, f, D, t_0, alpha, omega)

    return solver, y, z, status


if __name__ == '__main__':
    main()
