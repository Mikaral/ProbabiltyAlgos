# Este projeto objetiva explicar e implementar as diversas funções do estudo de variáveis discretas
# da cadeira de Probabilidade e Estatística.

import math


# Os eventos A e B são definidos como equivalentes quando PA(x) = PB(x)

# Ensaio de Bernoulli

def ensaio_bernoulli(n, x, p):
    prob = math.comb(n, x) * math.pow(p, x) * math.pow(1 - p, n - x)

    return prob


def ensaio_bernoulli_min(n, minvalue, p):
    soma = 0
    for i in range(n - minvalue + 1):
        soma += ensaio_bernoulli(n, minvalue + i, p)

    return soma
