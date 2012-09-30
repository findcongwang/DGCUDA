"""
This just plots the jacobi polynomials and the 2d legendre polynomials
because i want some familiarity with them before i use them as basis
functions in a big project. Here we go.
"""

import numpy as np
import matplotlib.pylab as p
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import gamma

def jacobiA(n, alpha, beta):
    return 2./(2.*n + alpha + beta) * np.sqrt((n * (n + alpha + beta) * (n + alpha) * (n + beta)) / (2.*n + alpha + beta - 1) * (2*n + alpha + beta + 1))

def jacobiB(n, alpha, beta):
    return - (alpha**2 - beta**2) / ((2*n + alpha + beta) * (2*n + alpha + beta + 2))

def jacobi(x, n, alpha, beta):
    if n == 0:
        return np.sqrt(2**(-alpha-beta-1)*gamma(alpha + beta + 2)/(gamma(alpha + 1) * gamma(beta + 1))) * np.ones(x.shape)
    if n == 1:
        return 1./2*jacobi(x, 0, alpha, beta) * np.sqrt((alpha + beta + 3) / ((alpha + 1) * (beta + 1))) * ((alpha + beta + 2)* x + (alpha - beta))

    a = jacobiA(n, alpha, beta)
    b = jacobiB(n, alpha, beta)

    return (x - b) * jacobi(x, n-1, alpha, beta) - a * jacobi(x, n-2, alpha, beta)

if __name__ == "__main__":
    
    X,Y = np.meshgrid(np.linspace(-0.9,0.9), np.linspace(-0.9,0.9))

    a = 2 * (1 + X) / (1 - Y) - 1
    b = Y
    #for i in xrange(0,3):
        #for j in xrange(3,0, -1):
    i = 2
    j = 0
    Z = np.sqrt(2) * jacobi(a, i, 0, 0) * jacobi(b, j, 2*i + 1, 0) * (1 - b)**i

    fig = p.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)

    p.show()
