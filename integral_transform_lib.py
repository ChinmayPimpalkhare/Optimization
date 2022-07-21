from tkinter import Y
import numpy as np
import scipy.linalg as sp

def stieltjes_transform_bounded(array_in_t, array_spacing, x, dt, upper_bound_for_integration):
    x = float(x)
    y = 0
    number_of_additions = int(upper_bound_for_integration*1.0/dt)
    for i in range(1, number_of_additions): 
        y += dt*1.0*array_in_t[int(i*dt/array_spacing)]/(x + i)
    return y

def stieltjes_transform_bounded_old(store_x, x, number_of_iterations, n, component_number, dx, dt): 
    stieltjes_of_f_of_t_bounded = np.zeros((number_of_iterations, n), dtype = float)
    lower_integral_in_the_variable_t = np.zeros((n,1), dtype = float)
    upper_integral_in_the_variable_t = np.zeros((n,1), dtype = float)
    number_of_integral_iterations = int(number_of_iterations/dt)
    for t in range (0, number_of_integral_iterations - 1): 
        for i in range(0,n):
            lower_integral_in_the_variable_t[i,0] += dt*store_x[t,i]/(x + t + dx)
    return lower_integral_in_the_variable_t[component_number, 0]

#The code snippet below computes the Stieltjes transform for a given epsilon that
#is a very small number. The value of R has to be supplied as an integer
#We are splitting the Stieltjes integral into two parts. 
#The first part computes the integral between epsilon to 1
#The second part computes the integral between 1 to R
#Refinement 0: 
def stieltjes_transform_epsilon_to_R(array_t, s, epsilon, R, i):
    stieltjes_from_epsilon_to_1 = 0
    stieltjes_from_1_to_R = 0
    stietljes_from_epsilon_to_1 = (1 - epsilon)/(s + epsilon)*(array_t[0,i]*(1 - epsilon) + array_t[1,i]*epsilon)
    for j in range (1,R-1):
        stieltjes_from_1_to_R += 0.5*(array_t[j,i] + array_t[j + 1, i])/(s + j + 0.5)
    stieltjes_from_epsilon_to_R = stieltjes_from_epsilon_to_1 + stieltjes_from_1_to_R
    return stieltjes_from_epsilon_to_R


#In the next version of the program, we use adaptive meshing here.
#So we split both the intervals into parts and then solve in both of the intervals. 
#Refinement 1: 
def stieltjes_transform_epsilon_to_R_refinement_1(array_t, s, epsilon, R, i, n_1, n_2):
    stieltjes_from_epsilon_to_1 = 0.0
    stieltjes_from_1_to_R = 0.0
    dt_1 = (1 - epsilon)/n_1
    for j in range(0,n_1):
        stietljes_from_epsilon_to_1 =+ dt_1*(1 - (epsilon + j*dt_1))/(s + (epsilon + j*dt_1))*(array_t[0,i]*(1 - (epsilon + j*dt_1)) + array_t[1,i]*(epsilon + j*dt_1))
    #n_2 is the number of additions per tile of size 1
    dt_2 = 1/n_2
    for j in range (1,R-1):
        for k in range(0,n_2):
            stieltjes_from_1_to_R += dt_2*(array_t[j,i]*(1-k*dt_2) + k*dt_2*array_t[j + 1, i])/(s + j +k*dt_2 + 0.5)
    stieltjes_from_epsilon_to_R = stieltjes_from_epsilon_to_1 + stieltjes_from_1_to_R
    return stieltjes_from_epsilon_to_R

    
