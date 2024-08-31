from numpy import set_printoptions
from numpy.random import RandomState


class PrintableDataMixin:
    """
    Mixin class for printing data in a formatted way.
    """

    def _print_data(self, def_names):
        """
        Prints the generated production data in a formatted way.

        :param def_names: Tuple of strings representing the names of the data.
        :type def_names: tuple
        """

        print(f"{'-' * 200}\n\"{self.__class__.__name__}\" Data:\n{'-' * 200}\n")
        set_printoptions(linewidth=255, precision=2)
        for name, value in zip(def_names, self.__dict__.values()):
            print(f"{name}:\n{value}\n\n{'=' * 200}\n")
        set_printoptions(linewidth=75, precision=8)


def get_random_by_seed(seed: int):
    """
    Returns a random number generator with the given seed.

    :param seed: Seed for the random number generator.
    :type seed: int
    :return: Random number generator with the given seed.
    :rtype: np.random.RandomState
    """

    return RandomState(seed)
