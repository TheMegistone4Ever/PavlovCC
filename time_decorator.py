import functools
import time


def time_decorator(func):
    """
    Decorator that measures the execution time of the decorated function.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: A function wrapper that returns a tuple with the result of the function
                  and the execution time in seconds.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        res = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record end time
        return res, end_time - start_time

    return wrapper


# Example usage
@time_decorator
def example_function(x, y):
    total = 0
    for _ in range(10000000):  # Simulate a time-consuming task
        total += (x + y) * (x / y) - (x - y)
    return total


if __name__ == "__main__":
    result, execution_time = example_function(3, 2)
    print("Result:", result)
    print("Execution Time (seconds):", execution_time)
