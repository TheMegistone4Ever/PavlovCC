import logging
import random
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Assuming these are the same import statements as in the original code
from outdated.FLAVTPM import solve_and_return_solution as solve_model1
from outdated.FMPAVTPMDS_v2 import solve_and_return_solution as solve_model2
from outdated.MPLAVTPM_2 import solve_and_return_solution as solve_model3


class DataGenerator:
    @staticmethod
    def generate_controlled_random_data(
            dimension: int,
            num_elements: int,
            seed: Optional[int] = None,
            distribution: str = 'uniform'
    ) -> Dict[str, Any]:
        """
        Generate controlled random data with multiple distribution options.

        Args:
            dimension (int): Dimension of the data
            num_elements (int): Number of elements in the system
            seed (int, optional): Random seed for reproducibility
            distribution (str): Type of distribution ('uniform', 'normal', 'exponential')

        Returns:
            Dict containing generated data parameters
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        generators = {
            'uniform': np.random.uniform,
            'normal': np.random.normal,
            'exponential': np.random.exponential
        }

        gen_func = generators.get(distribution, np.random.uniform)

        # New matrices for expert coefficients and element-level plans
        y_L_K = gen_func(1, 100, (num_elements, dimension))
        D_L_K = gen_func(0.1, 1, (num_elements, dimension))

        # Model 1 specific data generation
        production_matrix_1 = gen_func(0.1, 1, (4, dimension))
        y_assigned_1 = [float(x) for x in gen_func(1, 100, dimension)]
        b_1 = [random.uniform(y_assigned_1[i] if i < 3 else 1000, 10000) for i in range(4)]
        c_1 = [float(x) for x in gen_func(1, 10, dimension)]
        f_1 = [float(x) for x in gen_func(0.1, 1, dimension)]
        priorities_1 = np.ones(dimension)
        directive_terms_1 = sorted([float(x) for x in gen_func(10, 100, dimension)])
        t_0_1 = [float(i) for i in range(dimension)]
        alpha_1 = [float(x) for x in gen_func(1.0, 2, dimension)]

        # Model 2 specific data generation
        A_2 = gen_func(0.1, 1, (4, dimension))
        b_2 = gen_func(1, 10000, 4)
        C_2 = gen_func(1, 10, (5, dimension))
        f_2 = gen_func(0.1, 1, 9)
        D_2 = gen_func(10, 100, 9)
        t_0_2 = np.arange(9)
        alpha_2 = gen_func(1.0, 2, 9)
        omega_2 = gen_func(0, 1, 5)

        # Model 3 specific data generation
        production_matrix_3 = gen_func(0.1, 1, (4, dimension))
        y_assigned_3 = [float(x) for x in gen_func(1, 100, dimension)]
        b_3 = [random.uniform(y_assigned_3[i] if i < 3 else 1000, 10000) for i in range(4)]
        c_3 = [[float(random.uniform(1, 10)) for _ in range(dimension)] for _ in range(5)]
        priorities_3 = np.ones(dimension)
        directive_terms_3 = sorted([float(x) for x in gen_func(10, 100, dimension)])
        t_0_3 = [float(i) for i in range(dimension)]
        alpha_3 = [float(x) for x in gen_func(1.0, 2, dimension)]
        omega_3 = [random.uniform(0, 1) for _ in range(5)]
        omega_3 = [np.exp(w) / sum(np.exp(omega_3)) for w in omega_3]
        a_plus_3 = [random.uniform(0, 1) for _ in range(dimension)]
        a_plus_3 = [np.exp(a) / sum(np.exp(a_plus_3)) for a in a_plus_3]
        a_minus_3 = [random.uniform(0, 1) for _ in range(dimension)]
        a_minus_3 = [np.exp(a) / sum(np.exp(a_minus_3)) for a in a_minus_3]

        return {
            'model1': (
                production_matrix_1, y_assigned_1, b_1, c_1, f_1, priorities_1, directive_terms_1, t_0_1, alpha_1),
            'model2': (A_2, b_2, C_2, f_2, D_2, t_0_2, alpha_2, omega_2),
            'model3': (production_matrix_3, y_assigned_3, b_3, c_3, priorities_3, directive_terms_3,
                       t_0_3, alpha_3, omega_3, a_plus_3, a_minus_3),
            'expert_matrices': (y_L_K, D_L_K)
        }


class TwoLevelPlanningSystem:
    def __init__(
            self,
            delta: int = 100,
            dimension: int = 5,
            num_elements: int = 3,
            seed: Optional[int] = 1810,
            distribution: str = 'uniform'
    ):
        """
        Initialize the two-level planning system with enhanced configuration.

        Args:
            dimension (int): Dimension of the optimization problem (max 5)
            num_elements (int): Number of elements in the system
            seed (int, optional): Random seed for reproducibility
            distribution (str): Random data distribution type
        """
        if dimension > 5:
            raise ValueError("Dimension must be 5 or less")

        self.delta = delta
        self.dimension = dimension
        self.num_elements = num_elements
        self.seed = seed
        self.distribution = distribution

        logger.info(f"Initialized Planning System: Dimension={dimension}, Elements={num_elements}")

    def _format_value(self, val):
        """
        Convert various array types to a formatted list of values

        Args:
            val (numpy.ndarray or list): Input value to be formatted

        Returns:
            Formatted numpy array or list with detailed numpy printing
        """
        # Set numpy print options to display full arrays
        np.set_printoptions(
            threshold=np.inf,  # Show all elements
            linewidth=np.inf,  # Prevent line wrapping
            precision=4,  # 4 decimal places
            suppress=True  # Suppress scientific notation
        )

        # Handle numpy arrays
        if isinstance(val, np.ndarray):
            return val

        # Handle nested lists (like for model 3's C matrix)
        if isinstance(val, list) and isinstance(val[0], list):
            return np.array(val)

        # Handle regular lists and simple values
        return np.array(val)

    def generate_element_data(self) -> List[Dict[str, Any]]:
        """
        Generate enhanced random data for each element's model.

        Returns:
            List of dictionaries containing element-specific data
        """
        data = DataGenerator.generate_controlled_random_data(
            self.dimension,
            self.num_elements,
            seed=self.seed,
            distribution=self.distribution
        )

        # Extract expert matrices
        y_L_K, D_L_K = data['expert_matrices']

        element_data = [
            {
                'model': 'Model 1 (FLAVTPM)',
                'data_details': {
                    key: self._format_value(vals)
                    for key, vals in zip(
                        ['Production Matrix', 'Y Assigned', 'B', 'C', 'F',
                         'Priorities', 'Directive Terms', 'T0', 'Alpha'],
                        data['model1']
                    )
                },
                'data': data['model1'],
                'solver_func': solve_model1
            },
            {
                'model': 'Model 2 (FMPAVTPMDS_v2)',
                'data_details': {
                    key: self._format_value(vals)
                    for key, vals in zip(
                        ['A Matrix', 'B', 'C Matrix', 'F', 'D',
                         'T0', 'Alpha', 'Omega'],
                        data['model2']
                    )
                },
                'data': data['model2'],
                'solver_func': solve_model2
            },
            {
                'model': 'Model 3 (MPLAVTPM_2)',
                'data_details': {
                    key: self._format_value(vals)
                    for key, vals in zip(
                        ['Production Matrix', 'Y Assigned', 'B', 'C Matrix',
                         'Priorities', 'Directive Terms', 'T0', 'Alpha',
                         'Omega', 'A Plus', 'A Minus'],
                        data['model3']
                    )
                },
                'data': data['model3'],
                'solver_func': solve_model3
            },
            {
                'model': 'Expert Coefficient Matrices',
                'data_details': {
                    'y_L_K': self._format_value(y_L_K),
                    'D_L_K': self._format_value(D_L_K)
                },
                'data': (y_L_K, D_L_K)
            }
        ]

        return element_data

    def create_detailed_results_table(self, results):
        """
        Create a comprehensive table of results and input data with tabulation and rounding

        Args:
            results (list): List of solution results

        Returns:
            str: Formatted tables of results and input data
        """
        full_output = []

        # Input Data Tables
        full_output.append("\n=== Input Data Details ===")
        for result in results:
            full_output.append(f"\n{result['model']} - Input Data Details")
            full_output.append(f"{'Attribute':<20} | {'Value':<50}")
            full_output.append("-" * 75)
            for key, value in result['data_details'].items():
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        value_str = np.array2string(value, formatter={'float_kind': lambda x: f"{x:.4f}"},
                                                    separator=', ')
                    elif value.ndim == 2:
                        value_str = '\n' + np.array2string(value, formatter={'float_kind': lambda x: f"{x:.4f}"},
                                                           separator=', ')
                    else:
                        value_str = str(value)
                elif isinstance(value, list):
                    value_str = ", ".join([f"{x:.4f}" if isinstance(x, (float, int)) else str(x) for x in value])
                else:
                    value_str = f"{value:.4f}" if isinstance(value, (float, int)) else str(value)
                full_output.append(f"{key:<20} | {value_str:<50}")

        # Results Table
        full_output.append("\n=== Solution Results ===")
        full_output.append(f"{'Model':<25} | {'Objective Value':<18} | {'Solutions':<50}")
        full_output.append("-" * 95)
        for result in results:
            model_name = result['model']

            # Skip the Expert Coefficient Matrices result
            if model_name == 'Expert Coefficient Matrices':
                continue

            solution = result['solution']

            if model_name == 'Model 1 (FLAVTPM)':
                y_solution, z_solution, objective_value = solution
                y_str = np.array2string(np.array(y_solution), formatter={'float_kind': lambda x: f"{x:.4f}"},
                                        separator=', ')
                full_output.append(f"{model_name:<25} | {objective_value:<18.4f} | {y_str:<50}")

            elif model_name == 'Model 2 (FMPAVTPMDS_v2)':
                solver, y, z, _ = solution
                y_sol = [y[i].solution_value() for i in range(len(y))]
                y_str = np.array2string(np.array(y_sol), formatter={'float_kind': lambda x: f"{x:.4f}"}, separator=', ')
                full_output.append(
                    f"{model_name:<25} | {solver.Objective().Value():<18.4f} | {y_str:<50}")

            elif model_name == 'Model 3 (MPLAVTPM_2)':
                y_solution, u_plus_solution, u_minus_solution, objective_value = solution
                y_str = np.array2string(np.array(y_solution), formatter={'float_kind': lambda x: f"{x:.4f}"},
                                        separator=', ')
                full_output.append(f"{model_name:<25} | {objective_value:<18.4f} | {y_str:<50}")

        # Expert Matrix Analysis Table
        full_output.append("\n=== Expert Matrix Analysis ===")
        full_output.append(f"{'Element':<10} | {'D_L^T x y_L':<20} | {'Sum(f_j_l * z_j_l)':<25} | {'Difference':<15}")
        full_output.append("-" * 75)

        # Find expert matrices
        expert_matrices = next((result for result in results if result.get('model') == 'Expert Coefficient Matrices'),
                               None)
        if expert_matrices:
            y_L_K, D_L_K = expert_matrices['data']

            # Compute differences for each element
            for i in range(self.num_elements):
                # In a real implementation, you'd use actual solver results for z_j_l
                # Here, we'll use a placeholder computation
                D_L_i = D_L_K[i]
                y_L_i = y_L_K[i]

                # Placeholder computations - replace with actual solver logic
                D_L_T_x_y_L = np.dot(D_L_i, y_L_i)
                sum_f_j_l_z_j_l = np.sum(np.random.uniform(0, 1, self.dimension))  # Placeholder

                difference = D_L_T_x_y_L - sum_f_j_l_z_j_l

                full_output.append(
                    f"{f'Element {i + 1}':<10} | {D_L_T_x_y_L:<20.4f} | {sum_f_j_l_z_j_l:<25.4f} | {difference:<15.4f}"
                )

        full_output.append(f"\n=== Criteria №2[d={self.delta}] ===")
        full_output.append(f"{'Element':<10} | {'D_L^T x y_L':<20} | {'Sum(f_j_l * z_j_l)':<25} | {'Difference':<15}")
        full_output.append("-" * 75)
        if expert_matrices:
            differences = []
            for i in range(self.num_elements):
                D_L_i = D_L_K[i]
                y_L_i = y_L_K[i]

                D_L_T_x_y_L = np.dot(D_L_i, y_L_i)
                sum_f_j_l_z_j_l = np.sum(np.random.uniform(0, 1, self.dimension))  # Placeholder

                difference = float(D_L_T_x_y_L - sum_f_j_l_z_j_l)
                differences.append(difference)

            min_difference = min(differences)
            for i, difference in enumerate(differences):
                if difference <= min_difference + self.delta:
                    D_L_T_x_y_L = np.dot(D_L_K[i], y_L_K[i])
                    sum_f_j_l_z_j_l = np.sum(np.random.uniform(0, 1, self.dimension))  # Placeholder
                    full_output.append(
                        f"{f'Element {i + 1}':<10} | {D_L_T_x_y_L:<20.4f} | {sum_f_j_l_z_j_l:<25.4f} | {difference:<15.4f}"
                    )

        return "\n".join(full_output)

    def run_planning_system(self):
        """
        Main method to run the two-level planning system

        Returns:
            dict: Comprehensive results of the planning system
        """
        # Generate data for each element
        element_data = self.generate_element_data()

        # Solve problems and evaluate results
        results = self.solve_and_evaluate_elements(element_data)

        # Create detailed results table
        detailed_results = self.create_detailed_results_table(results)
        print("\n--- Two-Level Planning System Results ---")
        print(detailed_results)

        return {
            'element_data': element_data,
            'results': results
        }

    def solve_and_evaluate_elements(self, element_data):
        """
        Solve problems for each element and evaluate results

        Args:
            element_data (list): List of element data dictionaries

        Returns:
            list: Solutions and evaluation results for each element
        """
        results = []

        for element in element_data:
            try:
                # Skip expert matrices in solving
                if element['model'] == 'Expert Coefficient Matrices':
                    results.append(element)
                    continue

                # Call the specific solver function for each model
                solution = element['solver_func'](*element['data'])

                results.append({
                    'model': element['model'],
                    'solution': solution,
                    'data_details': element['data_details']
                })
            except Exception as e:
                logger.error(f"Error solving {element['model']}: {e}")

        return results

    def visualize_results(self, results):
        """
        Visualize results from different models

        Args:
            results (list): List of solution results
        """
        plt.figure(figsize=(12, 6))

        # Extract objective values
        objective_values = []
        model_names = []

        for result in results:
            model_name = result['model']

            # Skip expert matrices visualization
            if model_name == 'Expert Coefficient Matrices':
                continue

            solution = result['solution']

            if model_name == 'Model 1 (FLAVTPM)':
                objective_values.append(solution[2])
            elif model_name == 'Model 2 (FMPAVTPMDS_v2)':
                objective_values.append(solution[0].Objective().Value())
            elif model_name == 'Model 3 (MPLAVTPM_2)':
                objective_values.append(solution[3])

            model_names.append(model_name)

        plt.bar(model_names, objective_values)
        plt.title('Objective Values Comparison')
        plt.xlabel('Models')
        plt.ylabel('Objective Value')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


def main():
    planning_system = TwoLevelPlanningSystem(
        dimension=5,
        num_elements=3,
        seed=1810,
        distribution='uniform'
    )
    results = planning_system.run_planning_system()
    planning_system.visualize_results(results['results'])


if __name__ == '__main__':
    main()
