# from typing import NamedTuple, Callable
#
# import numpy as np
#
# rand = np.random.RandomState(1810)
#
#
# def softmax(x: np.ndarray) -> np.ndarray:
#     """
#     Computes the softmax function for a given input array.
#
#     :param x: The input array.
#     :type x: np.ndarray
#     :return: The softmax function output.
#     :rtype: np.ndarray
#     """
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()
#
#
# class ProductionData(NamedTuple):
#     """
#     Represents the base data for an Aggregated Volume-Time Production Model (AVTM).
#
#     This data structure encapsulates the parameters required for modeling and optimizing
#     production schedules in a discrete production system, considering both volume and time constraints.
#     It is based on the principles outlined in the research paper:
#
#     "Aggregated volume-time scheduling models for one class of discrete production systems"
#
#     The model aims to minimize production costs while adhering to resource constraints and
#     directive terms (deadlines). Penalties are incurred for violating these deadlines.
#
#     Attributes:
#         production_matrix (np.ndarray): A matrix representing the resource consumption per unit
#                                         of each aggregated product (matrix A in the article).
#                                         Dimensions: (number of aggregated products, number of production factors).
#         assigned_products (np.ndarray): The quantities of each assigned product to be produced (y_j in the article).
#                                          Dimensions: (number of assigned products,).
#         resource_limits (np.ndarray): The limits on the availability of each production factor (b in the article).
#                                       Dimensions: (number of aggregated products,).
#         linear_coefficients (np.ndarray): Coefficients of the linear objective function (cost function) (c in the article).
#                                           Dimensions: (number of production factors,).
#         penalty_coefficients (np.ndarray): Coefficients for penalties incurred due to deadline violations (f_j in the article).
#                                            Dimensions: (number of assigned products,).
#         priorities (np.ndarray): Priority levels for each aggregated product.
#                                   Dimensions: (number of production factors,).
#         directive_terms (np.ndarray): Directive terms (deadlines) for the completion of each aggregated product (D_j in the article).
#                                        Dimensions: (number of production factors,).
#         start_times (np.ndarray): Scheduled start times for the production of each aggregated product (t_(0,i) in the article).
#                                    Dimensions: (number of production factors,).
#         expert_coefficients (np.ndarray): Expert-defined coefficients influencing the production time
#                                            of each aggregated product (α_i in the article).
#                                            Dimensions: (number of production factors,).
#     """
#
#     # Production Data part
#     production_matrix: np.ndarray
#     assigned_products: np.ndarray
#     resource_limits: np.ndarray
#     linear_coefficients: np.ndarray
#     penalty_coefficients: np.ndarray
#     priorities: np.ndarray
#     directive_terms: np.ndarray
#     start_times: np.ndarray
#     expert_coefficients: np.ndarray
#
#     @classmethod
#     def generate_prod(cls, num_aggregated_products: int, num_assigned_products: int, num_production_factors: int):
#         """
#         Generates a random instance of BaseProductionData for testing and simulation purposes.
#
#         :param num_aggregated_products: The number of aggregated products.
#         :type num_aggregated_products: int
#         :param num_assigned_products: The number of assigned products.
#         :type num_assigned_products: int
#         :param num_production_factors: The number of production factors (resources).
#         :type num_production_factors: int
#         :return: A randomly generated instance of BaseProductionData.
#         :rtype: BaseProductionData
#         """
#         production_matrix = rand.uniform(0.1, 1, (num_aggregated_products, num_production_factors))
#         assigned_products = rand.uniform(1, 100, num_assigned_products)
#         resource_limits = np.where(np.arange(num_aggregated_products) < num_assigned_products, rand.uniform(1, 100),
#                                    rand.uniform(1000, 10000))
#         linear_coefficients = rand.uniform(1, 10, num_production_factors)
#         penalty_coefficients = rand.uniform(0.1, 1, num_assigned_products)
#         priorities = np.ones(num_production_factors)
#         directive_terms = np.sort(rand.uniform(10, 100, num_production_factors))
#         start_times = np.arange(num_production_factors).astype(np.float64)
#         expert_coefficients = rand.uniform(1.0, 2, num_production_factors)
#
#         return cls(production_matrix, assigned_products, resource_limits, linear_coefficients, penalty_coefficients,
#                    priorities, directive_terms, start_times, expert_coefficients)
#
#
# class ProductionDataWithOmega(NamedTuple):
#     production_matrix: np.ndarray
#     assigned_products: np.ndarray
#     resource_limits: np.ndarray
#     linear_coefficients: np.ndarray
#     penalty_coefficients: np.ndarray
#     priorities: np.ndarray
#     directive_terms: np.ndarray
#     start_times: np.ndarray
#     expert_coefficients: np.ndarray
#     omega: np.ndarray
#
#     @classmethod
#     def generate_with_omega(cls, num_aggregated_products, num_assigned_products, num_production_factors, L):
#         """
#         Generates a random instance of ProductionDataWithOmega for testing and simulation purposes.
#
#         :param num_aggregated_products: The number of aggregated products.
#         :type num_aggregated_products: int
#         :param num_assigned_products: The number of assigned products.
#         :type num_assigned_products: int
#         :param num_production_factors: The number of production factors (resources).
#         :type num_production_factors: int
#         :param L: The number of goals.
#         :type L: int
#         :return: A randomly generated instance of ProductionDataWithOmega.
#         :rtype: ProductionDataWithOmega
#         """
#
#         base_data = ProductionData.generate_prod(num_aggregated_products, num_assigned_products, num_production_factors)
#         omega = softmax(rand.uniform(.1, 1, L))
#
#         return cls(*base_data, omega)
#
#
# class ProductionDataWithCL(NamedTuple):
#     # Production Data part
#     production_matrix: np.ndarray
#     assigned_products: np.ndarray
#     resource_limits: np.ndarray
#     linear_coefficients: np.ndarray
#     penalty_coefficients: np.ndarray
#     priorities: np.ndarray
#     directive_terms: np.ndarray
#     start_times: np.ndarray
#     expert_coefficients: np.ndarray
#
#     # CL part
#     p_l: np.ndarray
#     f_l_m_optimums: np.ndarray
#
#     @classmethod
#     def generate_with_c_l(cls, num_aggregated_products, num_assigned_products, num_production_factors, L, M_L,
#                           find_optimal_solution: Callable):
#         """
#         Generates a random instance of ProductionDataWithCL for testing and simulation purposes.
#
#         :param num_aggregated_products: The number of aggregated products.
#         :type num_aggregated_products: int
#         :param num_assigned_products: The number of assigned products.
#         :type num_assigned_products: int
#         :param num_production_factors: The number of production factors (resources).
#         :type num_production_factors: int
#         :param L: The number of goals.
#         :type L: int
#         :param M_L: The number of production factors.
#         :type M_L: int
#         :param find_optimal_solution: A function to find the optimal solution for given production data.
#         :type find_optimal_solution: Callable
#         :return: A randomly generated instance of ProductionDataWithCL.
#         :rtype: ProductionDataWithCL
#         """
#
#         production_data = ProductionData.generate_prod(num_aggregated_products, num_assigned_products,
#                                                        num_production_factors)
#         production_data.linear_coefficients = rand.uniform(1, 10, (M_L, L, num_production_factors))
#         p_l = softmax(rand.uniform(.1, 1, L))
#         f_l_m_optimums = np.fromfunction(lambda m, l: find_optimal_solution(production_data, l, m), (M_L, L),
#                                          dtype=np.float64)
#
#         return cls(*production_data, p_l, f_l_m_optimums)
#
#
# class ProductionDataWithPm(NamedTuple):
#     # Production Data part
#     production_matrix: np.ndarray
#     assigned_products: np.ndarray
#     resource_limits: np.ndarray
#     linear_coefficients: np.ndarray
#     penalty_coefficients: np.ndarray
#     priorities: np.ndarray
#     directive_terms: np.ndarray
#     start_times: np.ndarray
#     expert_coefficients: np.ndarray
#
#     # PM part
#     p_m: np.ndarray
#
#     @classmethod
#     def generate_with_p_m(cls, num_aggregated_products, num_assigned_products, num_production_factors, M):
#         """
#         Generates a random instance of ProductionDataWithPm for testing and simulation purposes.
#
#         :param num_aggregated_products: The number of aggregated products.
#         :type num_aggregated_products: int
#         :param num_assigned_products: The number of assigned products.
#         :type num_assigned_products: int
#         :param num_production_factors: The number of production factors (resources).
#         :type num_production_factors: int
#         :param M: The number of production factors.
#         :type M: int
#         :return: A randomly generated instance of ProductionDataWithPm.
#         :rtype: ProductionDataWithPm
#         """
#
#         production_data = ProductionData.generate_prod(num_aggregated_products, num_assigned_products, num_production_factors)
#         production_data.linear_coefficients = rand.uniform(1, 10, (M, num_production_factors))
#         p_m = softmax(rand.uniform(.1, 1, M))
#
#         return cls(*production_data, p_m)
#
#
# class ProductionDataWithAPlusAMinus(NamedTuple):
#     # Production Data part
#     production_matrix: np.ndarray
#     assigned_products: np.ndarray
#     resource_limits: np.ndarray
#     linear_coefficients: np.ndarray
#     penalty_coefficients: np.ndarray
#     priorities: np.ndarray
#     directive_terms: np.ndarray
#     start_times: np.ndarray
#     expert_coefficients: np.ndarray
#
#     # A+ A- part
#     a_plus: np.ndarray
#     a_minus: np.ndarray
#
#     @classmethod
#     def generate_with_apm(cls, num_aggregated_products, num_assigned_products, num_production_factors, L):
#         """
#         Generates a random instance of ProductionDataWithAPlusAMinus for testing and simulation purposes.
#
#         :param num_aggregated_products: The number of aggregated products.
#         :type num_aggregated_products: int
#         :param num_assigned_products: The number of assigned products.
#         :type num_assigned_products: int
#         :param num_production_factors: The number of production factors (resources).
#         :type num_production_factors: int
#         :param L: The number of goals.
#         :type L: int
#         :return: A randomly generated instance of ProductionDataWithAPlusAMinus.
#         :rtype: ProductionDataWithAPlusAMinus
#         """
#
#         base_data = ProductionData.generate_prod(num_aggregated_products, num_assigned_products, num_production_factors)
#         a_plus = softmax(rand.uniform(.1, 1, num_assigned_products))
#         a_minus = softmax(rand.uniform(.1, 1, num_assigned_products))
#
#         return cls(*base_data, a_plus, a_minus)
#
#
# # TODO - CHECK ALL TYPES AND DIMENSIONS - WARNING - C
#
#
# if __name__ == "__main__":
#     data = ProductionData.generate_prod(20, 9, 30)
#     print(f"{data=}")
#     data_with_omega = ProductionDataWithOmega.generate_with_omega(20, 9, 30, 20)
#     print(f"{data_with_omega=}")
#     data_with_cl = ProductionDataWithCL.generate_with_c_l(20, 9, 30, 5, 15, lambda x, y, z: 0)
#     print(f"{data_with_cl=}")
#     data_with_pm = ProductionDataWithPm.generate_with_p_m(20, 9, 30, 15)
#     print(f"{data_with_pm=}")
#     data_with_apm = ProductionDataWithAPlusAMinus.generate_with_apm(20, 9, 30, 5)
#     print(f"{data_with_apm=}")


from dataclasses import dataclass
from typing import Callable

import numpy as np

rand = np.random.RandomState(1810)


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


@dataclass
class ProductionData:
    production_matrix: np.ndarray
    assigned_products: np.ndarray
    resource_limits: np.ndarray
    linear_coefficients: np.ndarray
    penalty_coefficients: np.ndarray
    priorities: np.ndarray
    directive_terms: np.ndarray
    start_times: np.ndarray
    expert_coefficients: np.ndarray

    @classmethod
    def generate_prod(cls, num_aggregated_products: int, num_assigned_products: int, num_production_factors: int):
        production_matrix = rand.uniform(0.1, 1, (num_aggregated_products, num_production_factors))
        assigned_products = rand.uniform(1, 100, num_assigned_products)
        resource_limits = np.where(np.arange(num_aggregated_products) < num_assigned_products,
                                   rand.uniform(1, 100),
                                   rand.uniform(1000, 10000))
        linear_coefficients = rand.uniform(1, 10, num_production_factors)
        penalty_coefficients = rand.uniform(0.1, 1, num_assigned_products)
        priorities = np.ones(num_production_factors)
        directive_terms = np.sort(rand.uniform(10, 100, num_production_factors))
        start_times = np.arange(num_production_factors).astype(np.float64)
        expert_coefficients = rand.uniform(1.0, 2, num_production_factors)

        return cls(production_matrix, assigned_products, resource_limits, linear_coefficients,
                   penalty_coefficients, priorities, directive_terms, start_times, expert_coefficients)


@dataclass
class ProductionDataWithOmega(ProductionData):
    omega: np.ndarray

    @classmethod
    def generate_with_omega(cls, num_aggregated_products, num_assigned_products, num_production_factors, L):
        production_data = ProductionData.generate_prod(num_aggregated_products, num_assigned_products,
                                                       num_production_factors)
        production_data.linear_coefficients = rand.uniform(1, 10, (L, num_production_factors))
        omega = softmax(rand.uniform(.1, 1, L))

        return cls(**production_data.__dict__, omega=omega)


@dataclass
class ProductionDataWithCL(ProductionDataWithOmega):
    p_l: np.ndarray
    f_l_m_optimums: np.ndarray

    @classmethod
    def generate_with_c_l(cls, num_aggregated_products, num_assigned_products, num_production_factors, L, M_L,
                          find_optimal_solution: Callable):
        production_data = ProductionDataWithOmega.generate_with_omega(num_aggregated_products, num_assigned_products,
                                                                      num_production_factors, L)
        p_l = softmax(rand.uniform(.1, 1, L))
        f_l_m_optimums = np.fromfunction(lambda m, l: find_optimal_solution(production_data, l, m), (M_L, L),
                                         dtype=np.float64)

        return cls(**production_data.__dict__, p_l=p_l, f_l_m_optimums=f_l_m_optimums)


@dataclass
class ProductionDataWithPm(ProductionData):
    p_m: np.ndarray

    @classmethod
    def generate_with_p_m(cls, num_aggregated_products, num_assigned_products, num_production_factors, M):
        production_data = ProductionData.generate_prod(num_aggregated_products, num_assigned_products,
                                                       num_production_factors)
        production_data.linear_coefficients = rand.uniform(1, 10, (M, num_production_factors))
        p_m = softmax(rand.uniform(.1, 1, M))

        return cls(**production_data.__dict__, p_m=p_m)


@dataclass
class ProductionDataWithAPlusAMinus(ProductionDataWithOmega):
    a_plus: np.ndarray
    a_minus: np.ndarray

    @classmethod
    def generate_with_apm(cls, num_aggregated_products, num_assigned_products, num_production_factors, L):
        base_data = ProductionDataWithOmega.generate_with_omega(num_aggregated_products, num_assigned_products,
                                                                num_production_factors, L)
        a_plus = softmax(rand.uniform(.1, 1, num_assigned_products))
        a_minus = softmax(rand.uniform(.1, 1, num_assigned_products))

        return cls(**base_data.__dict__, a_plus=a_plus, a_minus=a_minus)


if __name__ == "__main__":
    data = ProductionData.generate_prod(20, 9, 30)
    print(f"{data=}")
    data_with_omega = ProductionDataWithOmega.generate_with_omega(20, 9, 30, 20)
    print(f"{data_with_omega=}")
    data_with_cl = ProductionDataWithCL.generate_with_c_l(20, 9, 30, 5, 15, lambda x, y, z: 0)
    print(f"{data_with_cl=}")
    data_with_pm = ProductionDataWithPm.generate_with_p_m(20, 9, 30, 15)
    print(f"{data_with_pm=}")
    data_with_apm = ProductionDataWithAPlusAMinus.generate_with_apm(20, 9, 30, 5)
    print(f"{data_with_apm=}")
