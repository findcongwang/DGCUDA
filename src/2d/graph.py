#!/usr/bin/python

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as p
import numpy as np
from sys import argv

f = open(str(argv[1]), 'rb')

X = []
Y = []
U = []

for line in f:
    data = [float(x) for x in line.split(',')]
    X.append(data[0])
    Y.append(data[1])
    U.append(data[2])

X = np.array(X)
Y = np.array(Y)
U = np.array(U)

fig = p.figure()
ax = fig.gca(projection='3d')

surf = ax.scatter(X, Y, U)

p.show()
