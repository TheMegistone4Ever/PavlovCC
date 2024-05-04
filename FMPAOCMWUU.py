import numpy as np
import random
from ortools.linear_solver import pywraplp
from pprint import pprint

random.seed(1810)
np.random.seed(1810)

aggregated_products = 20  # m
production_factors = 10  # n
# upper_bound = 1000.0

# Ay <= b, y[i] >= 0
# production_matrix[i][j] (A) - value of a j-th production factor for i-th product
production_matrix = np.random.uniform(0.1, 1, (aggregated_products, production_factors))

# y[i] - how many first n1 products we want to produce
n1 = 9
y_assigned = [random.uniform(1, 100) for _ in range(n1)]

# b[i] - recourse variables, the first n1 products are produced with the first production factors
# and must be greater than y_assigned[i]
b = [random.uniform(y_assigned[i] if i < n1 else 1000, 10000) for i in range(aggregated_products)]

# c - profit for producing one unit of the i-th production factor
c = [random.uniform(1, 10) for _ in range(production_factors)]

# f - penalty for not producing enough of the first n1 products, must be less than c
f = [random.uniform(0.1, 1) for i in range(n1)]

# priorities - order of production factors
priorities = np.ones(production_factors)

# directive_terms - directive_terms - times we can use the directive term
directive_terms = [random.uniform(10, 100) for _ in range(production_factors)]
directive_terms.sort()

# t_0 - starting time for product
t_0 = [float(i) for i in range(production_factors)]

# alpha - time for a production factor
alpha = [random.uniform(1.0, 2) for _ in range(production_factors)]  # Uniform floats

solve_criteria_1 = True

# For Criterion 1:
omega = [random.uniform(0, 1) for _ in range(production_factors)]
omega = np.exp(omega) / np.sum(np.exp(omega))  # Softmax

# Print all the data

print("Production matrix:")
pprint(production_matrix)
print("=" * 100)

print("Y assigned:")
pprint(y_assigned)
print("=" * 100)

print("B:")
pprint(b)
print("=" * 100)

print("C:")
pprint(c)
print("=" * 100)

print("F:")
pprint(f)
print("=" * 100)

print("Priorities:")
pprint(priorities)
print("=" * 100)

print("Directive terms:")
pprint(directive_terms)
print("=" * 100)

print("T_0:")
pprint(t_0)
print("=" * 100)

print("Alpha:")
pprint(alpha)
print("=" * 100)

print("Omega:")
pprint(omega)
print("=" * 100)

# ======================================================================================================================

lp_solver = pywraplp.Solver.CreateSolver("GLOP")

y = [lp_solver.NumVar(0, lp_solver.infinity(), f"y_{i}") for i in range(production_factors)]

# T_i - z_i <= directive_terms[i], z_i >= 0

z = [lp_solver.NumVar(0, lp_solver.infinity(), f"z_{i}") for i in range(n1)]

# Ay <= b, y[i] >= 0;
# T_i - z_i <= directive_terms[i], z_i >= 0, T_i = t_0[i] + alpha[i] * y[i], i <= n1;
# y[i] >= y_assigned[i], i < n1;
# n1 < production_factors;
# Objective: max sum(c[i] * y[i]) - sum(f[i] * z[i])

# First: Ay <= b, y[i] >= 0
for i in range(aggregated_products):
    lp_solver.Add(sum(production_matrix[i][j] * y[j] for j in range(production_factors)) <= b[i])

# Second: T_i - z_i <= directive_terms[i], z_i >= 0, T_i = t_0[i] + alpha[i] * y[i], i <= n1
for i in range(n1):
    lp_solver.Add(z[i] >= 0)
    lp_solver.Add((t_0[i] + alpha[i] * y[i]) - z[i] <= directive_terms[i])

# Third: y[i] >= y_assigned[i], i < n1
for i in range(n1):
    lp_solver.Add(y[i] >= y_assigned[i])

# Objective: max sum(c[i] * y[i]) - sum([i] * z[i])
objective = lp_solver.Objective()
for l in range(production_factors):
    if solve_criteria_1:
        objective.SetCoefficient(y[l], c[l] * priorities[l] * omega[l])
    else:  # Criterion 1a
        for m in range(aggregated_products):
            objective.SetCoefficient(y[l], c[l] * production_matrix[m][l] * omega[l])
for i in range(n1):
    objective.SetCoefficient(z[i], -f[i])
objective.SetMaximization()

lp_solver.Solve()

# Results

detailed_results = {
    "Objective": objective.Value(),
    "Y_solution": [y[i].solution_value() for i in range(production_factors)],
    "Z_solution": [z[i].solution_value() for i in range(n1)]
}

# f_opt^l - (c_l^T * y_com - sum(f_i * z_com))
results = []
for i in range(n1):
    f_opt = objective.Value()
    c_y_com = sum(c[j] * production_matrix[i][j] for j in range(production_factors))
    sum_fi_z_com = sum(z[i].solution_value() for i in range(n1))
    result = f_opt - (c_y_com - sum_fi_z_com)
    results.append(result)

# pprint("Results:")
# pprint(results)

print("Detailed results:")
pprint(detailed_results)
