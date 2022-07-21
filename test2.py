import numpy as np
import scipy.linalg as sp
from matplotlib import pyplot as plt
import projective_transformation_lib as ptlib
n = 2
A = np.random.randn(2,2)
xk = np.array([[0.3], [0.7]])
D = np.array([[0.3, 0], [0, 0.7]])
c = np.array([[2], [3]])
answer = ptlib.project_objective_onto_transformed_null_space(A, D, c, n)
