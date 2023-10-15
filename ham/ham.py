import numpy as np


def random_hamiltonian(size):
    links = dict()
    for i in range (size):
        a = np.random.randint(0,size)
        w = np.unique(np.random.randint(0,size,a))
        j = np.random.standard_cauchy(a)
        links[i] = [(w[p],j[p]) for p in range (len(w)) if a > 0 ]
    return links