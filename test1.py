import numpy as np
import scipy.linalg as sp
from matplotlib import pyplot as plt
import projective_transformation_lib as ptlib

n = 5
e = np.zeros((n,1), dtype=float)
e = e + 1.0
e_T = np.transpose(e)
I = np.identity(n)
factor_1 = I - 1/n*np.dot(e, e_T)
print(factor_1)