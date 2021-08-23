# A simple implementation of some probability distributions using Python.

import math


# The events A and B are defined as equivalents when PA(x) = PB(x)

# Bernoulli Distribution

def bernoulli_distribution(n, x, p):
    prob = math.comb(n, x) * math.pow(p, x) * math.pow(1 - p, n - x)

    return prob


def bernoulli_distribution_min(n, minvalue, p):
    total_sum = 0
    for i in range(n - minvalue + 1):
        total_sum += bernoulli_distribution(n, minvalue + i, p)

    return total_sum
