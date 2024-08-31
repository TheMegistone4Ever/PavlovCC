import numpy as np
from ortools.linear_solver import pywraplp

np.random.seed(1810)

import matplotlib.pyplot as plt


def main():
    # Data Generation
    m = 20  # Number of resources
    n = 30  # Number of products
    n_1 = 9  # Number of aggregated products
    L = 5  # Number of linear functions

    A = np.random.rand(m, n)
    b = np.random.rand(m) * 10000
    C = np.random.rand(L, n) * 10
    f = np.random.rand(n_1)
    priorities = np.ones(n)
    D = np.random.rand(n_1) * 100
    t_0 = np.arange(n)
    alpha = np.random.rand(n_1) * 2
    omega = np.random.rand(L) * 0.1

    # Print generated data
    print("Production matrix:")
    print(A)
    print("====================================================================================================")
    print("B:")
    print(b)
    print("====================================================================================================")
    print("C:")
    print(C)
    print("====================================================================================================")
    print("F:")
    print(f)
    print("====================================================================================================")
    print("Priorities:")
    print(priorities)
    print("====================================================================================================")
    print("Directive terms:")
    print(D)
    print("====================================================================================================")
    print("T_0:")
    print(t_0)
    print("====================================================================================================")
    print("Alpha:")
    print(alpha)
    print("====================================================================================================")
    print("Omega:")
    print(omega)
    print("====================================================================================================")

    # Create the linear solver using the GLOP backend.
    solver = pywraplp.Solver.CreateSolver('GLOP')

    # Variables
    y = [solver.NumVar(0, solver.infinity(), f'y_{i}') for i in range(n)]
    z = [solver.NumVar(0, solver.infinity(), f'z_{i}') for i in range(n_1)]

    # Constraints
    for i in range(m):
        solver.Add(solver.Sum([A[i][j] * y[j] for j in range(n)]) <= b[i])

    for i in range(n_1):
        T_i = t_0[i] + alpha[i] * y[i]
        solver.Add(T_i - z[i] <= D[i])

    # Objective
    objective = solver.Objective()
    for l in range(L):
        f_opt_l = find_f_opt(A, b, C[l], f, D, t_0, alpha, n, n_1, solver)
        objective.SetCoefficient(y[l], omega[l] * C[l][l])
        for i in range(n_1):
            objective.SetCoefficient(z[i], -omega[l] * f[i])
        dummy_variable = solver.NumVar(0, 0, f'dummy_{l}')
        objective.SetCoefficient(dummy_variable, -omega[l] * f_opt_l)

    objective.SetMaximization()

    # Solve
    status = solver.Solve()

    # Print solution
    if status == pywraplp.Solver.OPTIMAL:
        print('Detailed results:')
        print('Objective:')
        print(objective.Value())
        print("====================================================================================================")
        print("Y_solution:")
        print([y[i].solution_value() for i in range(n)])
        print("====================================================================================================")
        print("Z_solution:")
        print([z[i].solution_value() for i in range(n_1)])
        print("====================================================================================================")
        print("Policy deadlines:")
        print([D[i] for i in range(n_1)])
        print("====================================================================================================")
        print("Completion dates:")
        print([t_0[i] + alpha[i] * y[i].solution_value() for i in range(n_1)])
        print("====================================================================================================")
        print("Differences:")
        print([D[i] - (t_0[i] + alpha[i] * y[i].solution_value()) for i in range(n_1)])
        print("====================================================================================================")
        print("Differences between f_optimum and f_solution:")
        diffs = []
        for l in range(L):
            f_opt_l = find_f_opt(A, b, C[l], f, D, t_0, alpha, n, n_1, solver)
            f_solution_l = C[l] @ np.array([y[i].solution_value() for i in range(n)]) - f @ np.array(
                [z[i].solution_value() for i in range(n_1)])
            diff = f_opt_l - f_solution_l
            diffs.append(diff)
            print(f'l={l},\tf_optimum={f_opt_l:.2f},\tf_solution={f_solution_l:.2f},\t{diff=:.2f}')

        fig, ax = plt.subplots()
        ax.plot([l for l in range(L)], [find_f_opt(A, b, C[l], f, D, t_0, alpha, n, n_1, solver) for l in range(L)],
                label='f_optimum')
        ax.plot([l for l in range(L)], [C[l] @ np.array([y[i].solution_value() for i in range(n)]) - f @ np.array(
            [z[i].solution_value() for i in range(n_1)]) for l in range(L)], label='f_solution')
        ax.legend()
        plt.show()

        fig, ax = plt.subplots()
        ax.bar([l for l in range(L)], diffs, color=['blue' if diff >= 0 else 'red' for diff in diffs])
        plt.show()


    else:
        print('The problem does not have an optimal solution.')


def find_f_opt(A, b, c_l, f, D, t_0, alpha, n, n_1, solver):
    # Create a new solver for each f_opt calculation
    solver_f_opt = pywraplp.Solver.CreateSolver('GLOP')

    # Variables
    y_f_opt = [solver_f_opt.NumVar(0, solver_f_opt.infinity(), f'y_{i}') for i in range(n)]
    z_f_opt = [solver_f_opt.NumVar(0, solver_f_opt.infinity(), f'z_{i}') for i in range(n_1)]

    # Constraints
    for i in range(len(A)):
        solver_f_opt.Add(solver_f_opt.Sum([A[i][j] * y_f_opt[j] for j in range(n)]) <= b[i])

    for i in range(n_1):
        T_i = t_0[i] + alpha[i] * y_f_opt[i]
        solver_f_opt.Add(T_i - z_f_opt[i] <= D[i])

    # Objective
    objective_f_opt = solver_f_opt.Objective()
    for i in range(n):
        objective_f_opt.SetCoefficient(y_f_opt[i], c_l[i])
    for i in range(n_1):
        objective_f_opt.SetCoefficient(z_f_opt[i], -f[i])

    objective_f_opt.SetMaximization()

    # Solve
    solver_f_opt.Solve()

    return objective_f_opt.Value()


if __name__ == '__main__':
    main()
