import numpy as np
import scipy.linalg as sp
import math

def find_center_of_simplex(n):
    x0 = np.zeros((n,1),dtype = float)
    x0 = x0 + 1.0
    x0 = x0/n
    return x0

def find_diagonal_matrix_D_given_x(xk, n): 
    Dk = np.zeros((n,n), dtype = float)
    for i in range (0,n): 
        Dk[i,i] = xk[i,0] 
    return Dk 

def projective_transform(D_inv, x, n): 
    e = np.zeros((n,1), dtype = float)
    e = e + 1.0
    e_T = np.transpose(e)
    D_inv_times_x = np.dot(D_inv, x)
    e_T_times_D_inv_times_x = np.dot(e_T, D_inv_times_x)
    y = D_inv_times_x / e_T_times_D_inv_times_x
    return y

def inverse_projective_transform(D, y, n): 
    e = np.zeros((n,1), dtype = float)
    e = e + 1.0
    e_T = np.transpose(e)
    D_times_y = np.dot(D,y)
    e_T_times_D_times_y = np.dot(e_T, D_times_y)
    x = D_times_y / e_T_times_D_times_y
    return x

def project_objective_onto_transformed_null_space(A, D, c, n):
    e = np.zeros((n,1), dtype=float)
    e = e + 1.0
    e_T = np.transpose(e)
    I = np.identity(n)
    A_T = np.transpose(A)
    factor_1 = I - 1/n*np.dot(e, e_T)
    #print("factor 1: ",  factor_1)
    A_times_D = np.dot(A,D)
    #print("AD: ", A_times_D)
    D_times_A_T = np.dot(D,A_T)
    #print("DA^T: ", D_times_A_T)
    D_times_c = np.dot(D,c)
    #print("Dc: ", D_times_c)
    A_times_D_square_times_D_times_A_T = np.dot(A_times_D, D_times_A_T) 
    #print("AD^2A^T: ", A_times_D_square_times_D_times_A_T)
    inverse_of_A_times_D_square_times_D_times_A_T = sp.inv(A_times_D_square_times_D_times_A_T, check_finite = True)
    #print("(AD^2A^T)^-1:" , inverse_of_A_times_D_square_times_D_times_A_T)
    part_1_of_factor_2 = np.dot(D_times_A_T, inverse_of_A_times_D_square_times_D_times_A_T)
    part_2_of_factor_2 = np.dot(part_1_of_factor_2, A_times_D)
    factor_2 = I - part_2_of_factor_2
    #print("factor_2: ", factor_2)
    projection_vector = np.dot(factor_1, factor_2)
    #print(projection_vector)
    c_subscript_p = np.dot(projection_vector, D_times_c)
    #print(c_subscript_p)
    return c_subscript_p

def follow_projected_direction(c_subscript_p, alpha, n):
    e = np.zeros((n,1), dtype = float)
    e = e + 1.0 
    x_prime = e/n - alpha*c_subscript_p
    return x_prime 

def compute_projection_operator_P_e(n): 
    e = np.zeros((n,1), dtype = float)
    e = e + 1.0 
    e_T = np.transpose(e)
    I = np.identity(n)
    P_e = I - 1/n*np.dot(e,e_T)
    return P_e 

def triple_product_vector_matrix_vector(vector_1, matrix, vector_2): 
    intermediate_matrix_1 = np.dot(matrix, vector_2)
    vector_1_transpose = np.transpose(vector_1)
    triple_product = np.dot(vector_1_transpose, intermediate_matrix_1)
    return triple_product

def compute_projection_matrix_P_AD(A,D,c,x,n): 
    A_T = np.transpose(A)
    I = np.identity(n)
    matrix_1 = np.dot(A,D)
    matrix_2 = np.dot(D,A_T)
    matrix_3 = np.dot(matrix_1, matrix_2)
    matrix_3_inv = sp.inv(matrix_3)
    matrix_4 = np.dot(matrix_2, matrix_3_inv)
    matrix_5 = np.dot(matrix_4, matrix_1)
    P_AD = I - matrix_5
    return P_AD

def compute_derivative_dx_dt(A,D,c,x,n):
    matrix_1 = np.dot(D,c)
    P_AD = compute_projection_matrix_P_AD(A,D,c,x,n)
    matrix_2 = np.dot(P_AD, matrix_1)
    dx_dt = -np.dot(D,matrix_2)
    return dx_dt

def compute_V_from_x(A, D, c, x, n):
    V = np.zeros((n,n), dtype = float)
    dx_dt = compute_derivative_dx_dt(A, D, c,x, n)
    for i in range(0,n): 
        V[i,i] = dx_dt[i]/x[i]
    return V

def compute_v_from_V_and_e(A,D,c,x,n):
    V = compute_V_from_x(A,D,c,x,n) 
    e = np.zeros((n,1), dtype = float)
    e = e + 1.0 
    v = np.dot(V,e)
    return v 

def compute_w_from_V_and_e(A,D,c,x,n): 
    e = np.zeros((n,1), dtype = float)
    e = e + 1.0
    V = compute_V_from_x(A,D,c,x,n)
    v = np.dot(V,e)
    w = np.dot(V,v)
    return w

def compute_curvature(A,D,c,x,n): 
    w = compute_w_from_V_and_e(A,D,c,x,n)
    v = compute_v_from_V_and_e(A,D,c,x,n)
    P_e = compute_projection_operator_P_e(n)
    vPv = triple_product_vector_matrix_vector(v,P_e,v)
    wPw = triple_product_vector_matrix_vector(w,P_e,w)
    vPw = triple_product_vector_matrix_vector(v,P_e,w)
    curvature = math.sqrt((vPv*wPw - math.pow(vPw,2))/(math.pow(vPv, 3)))
    return curvature


