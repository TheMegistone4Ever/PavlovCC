import numpy as np
import random
from ortools.linear_solver import pywraplp
from pprint import pprint

random.seed(1810)
np.random.seed(1810)

# Constants
num_aggregated_products = 20  # m
num_production_factors = 10  # n
num_assigned_products = 9  # n1
L = 5


def generate_production_data():
    """Generates production data including matrix, assigned quantities, resource limits, etc."""
    production_matrix = np.random.uniform(0.1, 1, (num_aggregated_products, num_production_factors))
    y_assigned = [random.uniform(1, 100) for _ in range(num_assigned_products)]
    b = [random.uniform(y_assigned[i] if i < num_assigned_products else 1000, 10000) for i in
         range(num_aggregated_products)]
    c = [[random.uniform(1, 10) for _ in range(num_production_factors)] for _ in range(L)]
    f = [random.uniform(0.1, 1) for _ in range(num_assigned_products)]
    priorities = np.ones(num_production_factors)
    directive_terms = sorted([random.uniform(10, 100) for _ in range(num_production_factors)])
    t_0 = [float(i) for i in range(num_production_factors)]
    alpha = [random.uniform(1.0, 2) for _ in range(num_production_factors)]
    omegas = [np.exp([random.uniform(0, 1) for _ in range(num_production_factors)]) for _ in range(L)]
    omegas = [[np.exp(omega_i) / sum(np.exp(omega)) for omega_i in omega] for omega in omegas]  # Softmax normalization
    return production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha, omegas


def print_data(data):
    """Prints the generated production data in a formatted way."""
    names = ["Production matrix", "Y assigned", "B", "C", "F", "Priorities", "Directive terms", "T_0", "Alpha",
             "Omegas"]
    for name, value in zip(names, data):
        print(f"{name}:")
        pprint(value)
        print("=" * 100)


def find_temp_optimal_solution(production_data, l):
    """Finds the temporary optimal solution for the given production data."""
    production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha, _ = production_data

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

    # Define Goal
    objective = lp_solver.Objective()
    for i in range(num_production_factors):
        objective.SetCoefficient(y[i], c[l][i] * priorities[i])
    for i in range(num_assigned_products):
        objective.SetCoefficient(z[i], -f[i])
    objective.SetMaximization()

    lp_solver.Solve()
    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], objective.Value()


def solve_production_problem(production_data):
    """Defines and solves the linear programming problem for production optimization."""
    production_matrix, y_assigned, b, c, f, priorities, directive_terms, t_0, alpha, omegas = production_data

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

    objectives = list()
    for i in range(L):
        objective = lp_solver.Objective()
        for l in range(num_production_factors):
            objective.SetCoefficient(y[l], c[i][l] * priorities[l] * omegas[i][l])
        for l in range(num_assigned_products):
            objective.SetCoefficient(z[l], -f[l])
        objectives.append(objective)

    for i in range(L):
        objectives[i].SetMaximization()
        lp_solver.Solve()
        print(f"Objective {i}: {objectives[i].Value()}")
        print(f"Y: {[y[i].solution_value() for i in range(num_production_factors)]}")
        print(f"Z: {[z[i].solution_value() for i in range(num_assigned_products)]}")
        print("=" * 100)

    return [y[i].solution_value() for i in range(num_production_factors)], [z[i].solution_value() for i in range(
        num_assigned_products)], [objective.Value() for objective in objectives]


if __name__ == "__main__":
    test_production_data = generate_production_data()
    F_optimums = [find_temp_optimal_solution(test_production_data, l)[2] for l in range(L)]
    print_data(test_production_data)
    y_solution, z_solution, objective_value = solve_production_problem(test_production_data)
    print("Detailed results:")
    pprint({"Objectives": objective_value, "Y_solution": y_solution, "Z_solution": z_solution})
    print("Differences between f_optimum and f_solution:")
    for l in range(L):
        # C_L^T * y_solution - F^T * z_solution
        c_l = test_production_data[3][l]
        f = test_production_data[4]
        f_solution = sum([c_l[i] * y_solution[i] for i in range(num_production_factors)]) - \
                     sum([f[i] * z_solution[i] for i in range(num_assigned_products)])
        print(f"F_optimum_{l} - F_solution_{l}: {F_optimums[l] - f_solution}")
