# A simple implementation of some probability distributions using Python.

import math
import sympy as sp

x, y, z = sp.symbols('x y z')


# Auxiliary function that returns the square of the parameter j
def power(j):
    return j ** 2


# Implementation of the Cumulative Distribution Function, which "interval" is an 2-size array
# of the interval used on the integral and expr is a string expressing the function.
def cumulative_distribution_continuous(interval, expr):
    return sp.integrate(expr, (x, interval[0], interval[1]))


# Implementation of expected values on discrete variables, which "x_values" are all the Values of x
# and "probs_x" are the respective probabilities of each x value.
def expected_value(x_values, probs_x):
    sum_probs = 0
    size = len(x_values)

    if size != len(probs_x):
        print("Incompatible sizes")
    else:
        for i in range(size):
            sum_probs += x_values[i] * probs_x[i]

    return sum_probs


# Implementation of expected values on continuous variables, which "interval" is an 2-size array
# of the interval used on the integral and expr is a string expressing the function.
def expected_value_continuous(interval, expr):
    expr += "*x"
    expr = sp.sympify(expr)

    return sp.integrate(expr, (x, interval[0], interval[1]))


# Implementation of Variance on discrete variables, which "x_values" are all the Values of x
# and "probs_x" are the respective probabilities of each x value.
def variance(x_values, probs_x):
    return expected_value(list(map(power, x_values)), probs_x) - math.pow(expected_value(x_values, probs_x), 2)


# Implementation of Variance on continuous variables, which "interval" is an 2-size array
# of the interval used on the integral and expr is a string expressing the function.
def variance_continuous(interval, expr):
    return expected_value_continuous(interval, expr + "*x") - math.pow(expected_value_continuous(interval, expr), 2)


# Implementation of Standard Deviation on discrete variables, which "x_values" are all the Values of x
# and "probs_x" are the respective probabilities of each x value.
def standard_deviation(x_values, probs_x):
    return math.sqrt(variance(x_values, probs_x))


# Implementation of the Standard deviation on continuous variables, which "interval" is an 2-size array
# of the interval used on the integral and expr is a string expressing the function.
def standard_deviation_continuous(interval, expr):
    return math.sqrt(variance_continuous(interval, expr))


#  - Binomial Distribution -

# Implementation of a Binomial Distribution with "n" total trials, and "k" successful ones.
# Which "p" is the probability of success.
def binomial_distribution(n, k, p):
    return math.comb(n, k) * math.pow(p, k) * math.pow(1 - p, n - k)


# Implementation of a Binomial Distribution with "n" total trials, and at least "minvalue" successful ones.
# Which "p" is the probability of success.
def binomial_distribution_min(n, minvalue, p):
    total_sum = 0
    for i in range(n - minvalue + 1):
        total_sum += binomial_distribution(n, minvalue + i, p)

    return total_sum


# Implementation of a Binomial Distribution with "n" total trials, and at most "maxvalue" successful ones.
# Which "p" is the probability of success.
def binomial_distribution_max(n, maxvalue, p):
    total_sum = 0
    for i in range(maxvalue + 1):
        total_sum += binomial_distribution(n, i, p)

    return total_sum


# Implementation of the expected value on a Binomial distribution, with "n" total trials and "p"
# is the probability of success
def expected_value_binomial(n, p):
    return n * p


# Implementation of the variance on a Binomial distribution with "n" total trials and "p"
# is the probability of success
def variance_binomial(n, p):
    return n * p * (1 - p)
