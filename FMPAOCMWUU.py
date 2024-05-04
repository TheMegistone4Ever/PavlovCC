import numpy as np
import random
from ortools.linear_solver import pywraplp
from pprint import pprint

random.seed(1810)
np.random.seed(1810)

aggregated_products = 20  # m
production_factors = 10  # n
upper_bound = 10.0

# Ay <= b, y[i] >= 0
# production_matrix[i][j] (A) - value of a j-th production factor for i-th product

production_matrix = np.random.uniform(0.0, upper_bound, (aggregated_products, production_factors))  # Uniform floats

# y[i] - how many first n1 products we want to produce
n1 = 9
y_assigned = [random.uniform(1.0, upper_bound) for _ in range(n1)]  # Uniform floats

# b[i] - recourse variables, the first n1 products are produced with the first production factors
# and must be greater than y_assigned[i]
b = [random.uniform(y_assigned[i] if i < n1 else 0.0, upper_bound) for i in range(aggregated_products)]

# c - cost of production factors
c = [random.uniform(0, 1) for _ in range(production_factors)]  # Uniform floats

# f - penalty for not producing enough of the first n1 products, must be less than c
f = [random.uniform(0.0, c[i] / 2) for i in range(n1)]

# priorities - order of production factors
priorities = np.zeros(production_factors)

# directive_terms - directive_terms - times we can use the directive term
# directive_terms = np.argsort([random.uniform(10.0, upper_bound) for _ in range(production_factors)])  # Uniform floats
directive_terms = [random.uniform(10.0, upper_bound) for _ in range(production_factors)]  # Uniform floats
directive_terms.sort()

# t_0 - starting time for product
t_0 = [float(i) for i in range(production_factors)]

# alpha - time for a production factor
alpha = [random.uniform(1.0, upper_bound) for _ in range(production_factors)]  # Uniform floats

# ======================================================================================================================

lp_solver = pywraplp.Solver.CreateSolver("GLOP")

y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(production_factors)]

# T_i - z_i <= directive_terms[i], z_i >= 0

z = [lp_solver.NumVar(0, lp_solver.infinity(), f"z_{i}") for i in range(n1)]

f_times_z = [lp_solver.NumVar(0, lp_solver.infinity(), f"f_times_z_{i}") for i in range(aggregated_products)]

# Ay <= b, y[i] >= 0;
# T_i - z_i <= directive_terms[i], z_i >= 0, T_i = t_0[i] + alpha[i] * y[i], i <= n1;
# y[i] >= y_assigned[i], i < n1;
# n1 < production_factors;
# Objective: max sum(c[i] * y[i]) - sum(f[i] * z[i])

# First: Ay <= b, y[i] >= 0
for i in range(aggregated_products):
    # constraint = lp_solver.Constraint(0, b[i])
    lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(production_factors)) <= b[i])
    # for j in range(production_factors):
    #     constraint.SetCoefficient(y[j], production_matrix[i][j])

# Second: T_i - z_i <= directive_terms[i], z_i >= 0, T_i = t_0[i] + alpha[i] * y[i], i <= n1
for i in range(n1):
    lp_solver.Add(z[i] >= 0)
    lp_solver.Add(t_0[i] + alpha[i] * y[i] - z[i] <= directive_terms[i])
    # constraint = lp_solver.Constraint(-lp_solver.infinity(), directive_terms[i])  # <= directive_terms[i]
    # constraint.SetCoefficient(z[i], -1)  # - z_i
    # constraint.SetCoefficient(y[i], alpha[i])  # + alpha[i] * y[i]
    # constraint.SetBounds(-lp_solver.infinity(), t_0[i])  # - z_i + alpha[i] * y[i] <= directive_terms[i]

# Third: y[i] >= y_assigned[i], i < n1
for i in range(n1):
    lp_solver.Add(y[i] >= y_assigned[i])

# Objective: max sum(c[i] * y[i]) - sum(f[i] * z[i])
objective = lp_solver.Objective()
for i in range(production_factors):
    objective.SetCoefficient(y[i], c[i])

for i in range(n1):
    objective.SetCoefficient(z[i], -f[i])

objective.SetMaximization()

lp_solver.Solve()

# Results

detailed_results = {
    "Objective": objective.Value(),
    "Y_solution": [y[i].solution_value() for i in range(production_factors)],
    "Z_solution": [z[i].solution_value() for i in range(n1)],
    # "F_times_Z": [f_times_z[i].solution_value() for i in range(aggregated_products)]
}

# f_opt^l - (c_l^T * y_com - sum(f_i * z_com))
results = []
f_opt = objective.Value()
for i in range(n1):
    c_y_com = sum(c[j] * production_matrix[j][i] for j in range(production_factors))
    sum_fi_z_com = sum(f_times_z[j].solution_value() for j in range(aggregated_products))
    result = f_opt - (c_y_com - sum_fi_z_com)
    results.append(result)

pprint(results)
pprint(detailed_results)
