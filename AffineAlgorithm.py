import numpy as np
import scipy as sp
import projective_transformation_lib as ptlib
from matplotlib import pyplot as plt
n = 4
c = np.random.randn(4,1)
x = np.zeros((4,1), dtype = float)
for i in range (0,4): 
    x[i] = 1.0/n
A = np.identity(4)
t = 0.01
D = np.zeros((4,4), dtype = float)
Solution = np.zeros((100,4), dtype = float)
for j in range(0,100):
 for i in range(0,4): 
     D[i,i] = x[i]
     Solution[j,i] = x[i]
 dxdt_1 = np.matmul(D,c)
 dxdt = -np.matmul(D,dxdt_1) 
 x = x + t*dxdt
x_grid = np.linspace(0,1,101)
plotting_x, plotting_y = np.zeros(100), np.zeros(100)
delta_x = 0.01
for i in range (0,100): 
    plotting_x[i] = Solution[i,0]
    plotting_y[i] = Solution[i,1]
plt.plot(plotting_x,plotting_y,'o')
plt.xlabel("Value of x")
plt.ylabel("Value of y")
plt.title("Approximate Solution with Affine Method")
plt.show()
 

 

 

