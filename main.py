from ortools.linear_solver import pywraplp


def criteria1_example1_solve_lp(criteria1_example1_c, criteria1_example1_A, criteria1_example1_b,
                                criteria1_example1_alpha, criteria1_example1_t_0, criteria1_example1_D,
                                criteria1_example1_y_com_z_com, criteria1_example1_y_priorities):
    solver = pywraplp.Solver.CreateSolver('GLOP')

    n = len(criteria1_example1_c)
    y = [solver.NumVar(0, solver.infinity(), f'y_{i}') for i in range(n)]
    z = [solver.NumVar(0, solver.infinity(), f'z_{i}') for i in range(len(criteria1_example1_D))]

    objective = solver.Objective()
    for l, omega_l in enumerate(criteria1_example1_y_priorities):
        objective.SetCoefficient(y[l], omega_l)
    objective.SetMaximization()

    # Ay <= b
    for i in range(len(criteria1_example1_b)):
        constraint = solver.Constraint(-solver.infinity(), criteria1_example1_b[i])
        for j in range(n):
            constraint.SetCoefficient(y[j], criteria1_example1_A[i][j])

    # (t_0 + alpha * y) - z <= D
    for i in range(len(criteria1_example1_D)):
        constraint = solver.Constraint(-solver.infinity(), criteria1_example1_D[i])
        constraint.SetCoefficient(z[i], -1)
        for j in range(n):
            constraint.SetCoefficient(y[j], criteria1_example1_alpha[i][j])
        solver.Add(criteria1_example1_t_0[i] + sum(criteria1_example1_alpha[i][j] * y[j] for j in range(n)) - z[i] <=
                   criteria1_example1_D[i])

    penalties = []
    for i in range(len(criteria1_example1_D)):
        penalties.append(solver.NumVar(0, solver.infinity(), f'penalty_{i}'))

    # y >= y_p
    for i in range(len(criteria1_example1_y_priorities)):
        solver.Add(y[i] >= criteria1_example1_y_priorities[i])

    solver.Solve()

    detailed_results = {
        "Objective": objective.Value(),
        "Y_solution": [y[j].solution_value() for j in range(n)],
        "Z_solution": [z[i].solution_value() for i in range(len(criteria1_example1_D))],
        "Penalties": [penalties[i].solution_value() for i in range(len(criteria1_example1_D))]
    }

    # f_opt^l - (c_l^T * y_com - sum(f_i * z_com))
    criteria1_example1_results = []
    for l, omega_l in enumerate(criteria1_example1_y_priorities):
        f_opt_l = objective.Value()
        c_l_y_com = sum(criteria1_example1_c[j] * criteria1_example1_y_com_z_com[j][l] if len(
            criteria1_example1_y_com_z_com[j]) > l else 0 for j in range(n))
        sum_fi_z_com = sum(penalties[i].solution_value() for i in range(len(criteria1_example1_D)))
        result = f_opt_l - (c_l_y_com - sum_fi_z_com)
        criteria1_example1_results.append(result)

    return criteria1_example1_results, detailed_results


def criteria1a_example1_solve_lp(criteria1a_example1_c, criteria1a_example1_A, criteria1a_example1_b,
                                 criteria1a_example1_alpha, criteria1a_example1_D, criteria1a_example1_P,
                                 criteria1a_example1_y_priorities):
    solver = pywraplp.Solver.CreateSolver('GLOP')

    n = len(criteria1a_example1_c[0][0])
    M = len(criteria1a_example1_P[0])
    y = [solver.NumVar(0, solver.infinity(), f'y_{i}') for i in range(n)]
    z = [solver.NumVar(0, solver.infinity(), f'z_{i}') for i in range(len(criteria1a_example1_D))]

    f = [[solver.NumVar(0, solver.infinity(), f'penalty_{l}_{m}_{i}') for i in range(n)] for m in range(M) for l in
         range(len(criteria1a_example1_P))]

    f_times_z = []
    for l in range(len(criteria1a_example1_P)):
        f_times_z_l = []
        for m in range(M):
            f_times_z_l.append(solver.NumVar(0, solver.infinity(), f'f_times_z_{l}_{m}'))
        f_times_z.append(f_times_z_l)

    objective_terms = []
    for l, omega_l in enumerate(criteria1a_example1_y_priorities):
        for m in range(M):
            term = solver.NumVar(0, solver.infinity(), f'term_{l}_{m}')
            inner_term = solver.NumVar(0, solver.infinity(), f'inner_term_{l}_{m}')
            for j in range(n):
                solver.Add(term - inner_term >= criteria1a_example1_c[l][m][j] * y[j] - f[l * M + m][j] * z[j])
            objective_terms.append(term - inner_term)
            solver.Add(objective_terms[-1] == omega_l * criteria1a_example1_P[l][m])

    solver.Maximize(sum(objective_terms))

    for i in range(len(criteria1a_example1_b)):
        constraint = solver.Constraint(-solver.infinity(), criteria1a_example1_b[i])
        for j in range(n):
            constraint.SetCoefficient(y[j], criteria1a_example1_A[i][j])
    for i in range(len(criteria1a_example1_D)):
        constraint = solver.Constraint(-solver.infinity(), criteria1a_example1_D[i])
        constraint.SetCoefficient(z[i], -1)
        for j in range(n):
            constraint.SetCoefficient(y[j], criteria1a_example1_alpha[i][j])

    solver.Solve()

    criteria1a_example1_detailed_results = {}
    for l, omega_l in enumerate(criteria1a_example1_y_priorities):
        for m in range(M):
            f_opt_l_m = sum(criteria1a_example1_c[l][m][j] * y[j].solution_value() for j in range(n))
            f_opt_l_m -= sum(f_i.solution_value() * z[j].solution_value() for j, f_i in enumerate(f[l * M + m]))
            criteria1a_example1_detailed_results[f'f_opt_{l}_{m}'] = f_opt_l_m

    return criteria1a_example1_detailed_results, [y[j].solution_value() for j in range(n)]


if __name__ == "__main__":
    # ТЕСТ1
    criteria1_example1_c = [1, 2, 3]
    criteria1_example1_A = [[1, 2, 3], [4, 5, 6]]
    criteria1_example1_b = [10, 20]
    criteria1_example1_alpha = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    criteria1_example1_t_0 = [1, 2]
    criteria1_example1_D = [5, 10]
    criteria1_example1_y_com_z_com = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    criteria1_example1_y_priorities = [0, 0, 0]

    criteria1_example1_results, criteria1_example1_detailed_results = criteria1_example1_solve_lp(criteria1_example1_c,
                                                                                                  criteria1_example1_A,
                                                                                                  criteria1_example1_b,
                                                                                                  criteria1_example1_alpha,
                                                                                                  criteria1_example1_t_0,
                                                                                                  criteria1_example1_D,
                                                                                                  criteria1_example1_y_com_z_com,
                                                                                                  criteria1_example1_y_priorities)
    print("", criteria1_example1_results)
    print("", criteria1_example1_detailed_results)

    # ТЕСТ2
    criteria1a_example1_c = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    criteria1a_example1_A = [[1, 2, 3], [4, 5, 6]]
    criteria1a_example1_b = [10, 20]
    criteria1a_example1_alpha = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    criteria1a_example1_D = [5, 10]
    criteria1a_example1_P = [[1, 2], [3, 4]]
    criteria1a_example1_y_priorities = [0, 0]

    criteria1a_example1_detailed_results, criteria1a_example1_y_solution = criteria1a_example1_solve_lp(
        criteria1a_example1_c, criteria1a_example1_A, criteria1a_example1_b, criteria1a_example1_alpha,
        criteria1a_example1_D, criteria1a_example1_P, criteria1a_example1_y_priorities)
    print("", criteria1a_example1_detailed_results)
    print("", criteria1a_example1_y_solution)
