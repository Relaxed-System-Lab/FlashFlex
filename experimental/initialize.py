import numpy as np

def initialize(param, size):
    K = np.random.binomial(param[0], param[1], size=size) + 1
    K = K[0]

    return K


