"""
data set operations
"""
import random
import math


def split_dataset(data, ratios):
    """

    Args:
        data:
        ratios:

    Returns:

    """

    random.shuffle(data)

    assert sum(ratios) == 1, "sum of ratios does not equal to 1"

    numbers = [math.floor(len(data) * x) for x in ratios]
    numbers[-1] += len(data) - sum(numbers)

    results = []
    start = 0
    for n in numbers:
        results.append(data[start: start + n])
        start += n

    return results
