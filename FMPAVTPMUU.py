import random

import numpy as np
from ortools.linear_solver import pywraplp

np.set_printoptions(linewidth=1000)

random.seed(1810)
np.random.seed(1810)

# Constants
num_aggregated_products = 20  # m
num_production_factors = 10  # n
num_assigned_products = 9  # n1
L = 5
M_L = 15


def generate_production_data():
    """Generates production data including matrix, assigned quantities, resource limits, etc."""
    production_matrix = np.random.uniform(0.1, 1, (num_aggregated_products, num_production_factors))
    y_assigned = [random.uniform(1, 100) for _ in range(num_assigned_products)]
    b = [random.uniform(y_assigned[i] if i < num_assigned_products else 1000, 10000) for i in
         range(num_aggregated_products)]

    C_L = [[[random.uniform(1, 10) for _ in range(num_production_factors)] for _ in range(L)] for _ in range(M_L)]

    P_L = [random.uniform(0, 1) for _ in range(M_L)]
    P_L = [np.exp(P_m_i) / sum(np.exp(P_L)) for P_m_i in P_L]

    f = [random.uniform(0.1, 1) for _ in range(num_assigned_products)]
    priorities = np.ones(num_production_factors)
    directive_terms = sorted([random.uniform(10, 100) for _ in range(num_production_factors)])
    t_0 = [float(i) for i in range(num_production_factors)]
    alpha = [random.uniform(1.0, 2) for _ in range(num_production_factors)]
    omega = [random.uniform(0, 1) for _ in range(L)]
    omega = [np.exp(omega_i) / sum(np.exp(omega)) for omega_i in omega]  # Softmax normalization
    test_production_data = production_matrix, y_assigned, b, C_L, f, priorities, directive_terms, t_0, alpha, omega

    F_L_M_optimums = list()
    for m in range(M_L):
        inner_optimums = list()
        for l in range(L):
            temp_test_production_data = production_matrix, y_assigned, b, C_L[m][
                l], f, priorities, directive_terms, t_0, alpha
            _, _, objective_value = find_temp_optimal_solution(temp_test_production_data)
            inner_optimums.append(objective_value)
        F_L_M_optimums.append(inner_optimums)

    return *test_production_data, F_L_M_optimums, P_L


def print_data(data, def_names=(
        "Production matrix", "Y assigned", "B", "C_L", "F", "Priorities", "Directive terms", "T_0", "Alpha", "Omega",
        "F_L_M_optimums", "P_L")):
    """Prints the generated production data in a formatted way."""
    for name, value in zip(def_names, data):
        print(f"{name}:\n{np.round(value, 2)}")
        print("=" * 100)


def find_temp_optimal_solution(temp_production_data):
    """Defines and solves the linear programming problem for production optimization."""
    production_matrix, y_assigned, b, c_l_m, f, priorities, directive_terms, t_0, alpha = temp_production_data

    lp_solver = pywraplp.Solver.CreateSolver("GLOP")

    # Define Variables
    y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(num_production_factors)]
    z = [lp_solver.NumVar(0, lp_solver.infinity(), f"z_{i}") for i in range(num_assigned_products)]

    # Define Constraints
    for i in range(num_aggregated_products):
        lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(num_production_factors)) <= b[i])
    for i in range(num_assigned_products):
        lp_solver.Add(z[i] >= 0)
        lp_solver.Add((t_0[i] + alpha[i] * y[i]) - z[i] <= directive_terms[i])
    for i in range(num_assigned_products):
        lp_solver.Add(y[i] >= y_assigned[i])

    # Define Objective Function
    objective = lp_solver.Objective()
    for i in range(num_production_factors):
        objective.SetCoefficient(y[i], c_l_m[i] * priorities[i])
    for i in range(num_assigned_products):
        objective.SetCoefficient(z[i], -f[i])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], objective.Value()


# Criteria: 1a
def solve_production_problem(production_data):
    """Defines and solves the linear programming problem for production optimization."""
    production_matrix, y_assigned, b, C_L, f, priorities, directive_terms, t_0, alpha, omega, _, P_L = production_data

    lp_solver = pywraplp.Solver.CreateSolver("GLOP")

    # Define Variables
    y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(num_production_factors)]
    z = [lp_solver.NumVar(0, lp_solver.infinity(), f"z_{i}") for i in range(num_assigned_products)]

    # Define Constraints
    for i in range(num_aggregated_products):
        lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(num_production_factors)) <= b[i])
    for i in range(num_assigned_products):
        lp_solver.Add(z[i] >= 0)
        lp_solver.Add((t_0[i] + alpha[i] * y[i]) - z[i] <= directive_terms[i])
    for i in range(num_assigned_products):
        lp_solver.Add(y[i] >= y_assigned[i])

    # Define Objective Function
    objective = lp_solver.Objective()
    for l in range(L):
        for m in range(M_L):
            for i in range(num_production_factors):
                objective.SetCoefficient(y[i], C_L[m][l][i] * priorities[i] * omega[l] * P_L[m])
            for i in range(num_assigned_products):
                objective.SetCoefficient(z[i], -f[i] * P_L[m])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], objective.Value()


if __name__ == "__main__":
    test_production_data = generate_production_data()
    print_data(test_production_data)
    y_solution, z_solution, objective_value = solve_production_problem(test_production_data)
    t_0 = test_production_data[7]
    alpha = test_production_data[8]
    policy_deadlines = [t_0[i] + alpha[i] * y_solution[i] for i in range(num_assigned_products)]
    completion_dates = [z_solution[i] for i in range(num_assigned_products)]
    differences = [policy_deadlines[i] - completion_dates[i] for i in range(num_assigned_products)]
    print("Detailed results:")
    names = ["Objective", "Y_solution", "Z_solution", "Policy deadlines", "Completion dates", "Differences"]
    print_data([objective_value, y_solution, z_solution, policy_deadlines, completion_dates, differences], names)
    print("Differences between f_optimum and f_solution:")
    for l in range(L):
        F_L_M_optimums = test_production_data[-2]
        mean_difference = 0
        P_L = test_production_data[-1]
        weighted_optimum_values = 0
        weighted_inner_differences = 0
        for m in range(M_L):
            inner_difference = 0  # C_M_l ^ T * Y_solution - F^T * Z_solution
            c_m_l = test_production_data[3][m][l]
            f = test_production_data[4]
            for i in range(num_assigned_products):
                inner_difference += c_m_l[i] * y_solution[i] - f[i] * z_solution[i]
            optimum_value = F_L_M_optimums[m][l]
            weighted_optimum_values += P_L[m] * optimum_value
            weighted_inner_differences += P_L[m] * inner_difference
        mean_difference = weighted_optimum_values - weighted_inner_differences
        print(f"{l = },\tomega_l = {test_production_data[-3][l]:,.2f},\t{weighted_optimum_values = :,.2f}"
              f",\t{weighted_inner_differences = :,.2f},\t{mean_difference = :,.2f}")
