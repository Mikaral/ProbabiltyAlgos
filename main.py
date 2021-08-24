# A simple implementation of some probability distributions using Python.

import math


# The events A and B are defined as equivalents when PA(x) = PB(x)

#  - Bernoulli Distribution -

# Implementation of a Bernoulli Distribution with "n" total trials, and "x" successful ones.
# Which "p" is the probability of success.
def bernoulli_distribution(n, x, p):
    prob = math.comb(n, x) * math.pow(p, x) * math.pow(1 - p, n - x)

    return prob


# Implementation of a Bernoulli Distribution with "n" total trials, and at least "minvalue" successful ones.
# Which "p" is the probability of success.
def bernoulli_distribution_min(n, minvalue, p):
    total_sum = 0
    for i in range(n - minvalue + 1):
        total_sum += bernoulli_distribution(n, minvalue + i, p)

    return total_sum


# Implementation of expected values on discrete variables.
def expected_value(x_values, probs_x):
    sum_probs = 0
    size = len(x_values)

    if size != len(probs_x):
        print("Incompatible sizes")
    else:
        for i in range(size):
            sum_probs += x_values[i] * probs_x[i]

    return sum_probs
