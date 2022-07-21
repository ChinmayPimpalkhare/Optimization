import numpy as np
import scipy.linalg as sp
from matplotlib import pyplot as plt
import projective_transformation_lib as ptlib
import pandas as pd
import integral_transform_lib as intlib
import time

st = time.time()
R = np.random.RandomState(1)
#m = input("Enter the dimension m for the problem: ")
m = 5
#n = input("Enter the dimension n for the problem: ")
n = 11
alpha = input("Input step-length for the problem: ")
alpha = float(alpha)
#A = R.randn(m,n)
A = np.array([[0.3190, -0.9855, -0.6188, 0.7835, 0.00489, -1.0077, 0.00562, -1.2389, -0.9723, 0.02989, 3.6803], [0.2023, -1.615, -2.008, -0.2642, 0.00471, 0.8544, 0.00858, -0.143, -1.452, -0.0365, 4.4503], [-0.5901, -0.4841, -2.9863, 0.1008, 0.000299, -0.163, -0.006447, 0.0345, -1.205, 0.0865, 5.213], [0.1055, 0.2377, 0.0687, -0.935, -0.00263, -0.8499, 0.004, 0.662, 0.386, 0.0584, 0.264],[0.8541, 2.198, 0.07348, 1.304, -0.00109, -1.196, -0.01095, -0.847, 1.445, 0.3682, -4.187]], dtype = float)
#c = np.zeros((n,1), dtype = float)
#for i in range (0, n): 
    #c[i,0] = input("Enter the component of the objective function: ")
    #c[i,0] = float(c[i,0])
c = np.array([[0.448], [-0.0636], [0.0119], [-1.205], [-0.00342], [-0.316], [-0.00373], [-1.154], [1.149], [0.326], [25.296]], dtype = float)
# Debug Line 1:
# print(n)
number_of_iterations = input("Enter the number of iterations you would want to have: ")
number_of_iterations = int(number_of_iterations)
store_x = np.zeros((number_of_iterations, 4*n), dtype = float)
store_x_inv = np.zeros((number_of_iterations, n), dtype = float)
stietljes_of_x_and_x_inv = np.zeros((number_of_iterations, 2*n ), dtype = float)
x0 = ptlib.find_center_of_simplex(n)
#D0 = ptlib.find_diagonal_matrix_D_given_x(x0, n)
xk = x0
for i in range(0,n): 
    store_x[0,i] = x0[i,0]
#Dk = D0
for i in range (0, number_of_iterations): 
    Dk = ptlib.find_diagonal_matrix_D_given_x(xk, n)
    #print(Dk)
    Dk_inv = sp.inv(Dk, check_finite = True)
    yk = ptlib.projective_transform(Dk_inv, xk, n)
    c_subscript_p = ptlib.project_objective_onto_transformed_null_space(A, Dk, c, n)
    #print(c_subscript_p)
    x_prime = ptlib.follow_projected_direction(c_subscript_p, alpha, n)
    #print(x_prime)
    xk_plus_1 = ptlib.inverse_projective_transform(Dk, x_prime, n)
    for j in range(0,n): 
        store_x[i,j] = xk_plus_1[j,0]
        store_x[i,j+n] = 1/store_x[i,j]
        store_x_inv[i,j] = store_x[i,j+n]
        #print(*store_x, sep = ' ')
    xk = xk_plus_1
for i in range(0, number_of_iterations):
    for j in range (0,n):
        stietljes_of_x_and_x_inv[i,j] = intlib.stieltjes_transform_bounded_old(store_x, i, number_of_iterations, n, j, 0.01, 1)
        stietljes_of_x_and_x_inv[i,j+n] = intlib.stieltjes_transform_bounded_old(store_x_inv, i, number_of_iterations, n, j, 0.01, 1)
        store_x[i,j+2*n] = stietljes_of_x_and_x_inv[i,j]
        store_x[i,j+3*n] = stietljes_of_x_and_x_inv[i, j+n]
x_end = np.zeros((n,1), dtype = float)
for i in range (0,n): 
    print(store_x[number_of_iterations - 1, i])
    x_end[i,0] = store_x[number_of_iterations - 1, i]
c_T = np.transpose(c)
objective_value = np.dot(c_T,x_end)
print("The value of the objective function at the end of the program is: ", objective_value)
et = time.time()
elapsed_time = et - st
print("Execution time: ", elapsed_time, "seconds")
df = pd.DataFrame(store_x)
df.to_excel("stored_values_of_data_points.xlsx")


