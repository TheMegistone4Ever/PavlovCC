from numpy import ndarray, exp, max


def softmax(x: ndarray) -> ndarray:
    """
    Computes the softmax function for the given input vector.

    :param x: The input vector.
    :type x: ndarray
    :return: The vector, where each element is greater than or equal to zero and the sum of all elements is equal to one.
    :rtype: ndarray
    """

    return (e_x := exp(x - max(x))) / e_x.sum()
