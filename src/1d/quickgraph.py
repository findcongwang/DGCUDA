from time import sleep
import matplotlib.pylab as p
from pylab import rcParams
import numpy as np

rcParams['figure.figsize'] = 20,10

f = open("data.txt", "rb")

Np = int(f.readline())
x  = [float(i) for i in f.readline().split()]
dx = x[Np+1] - x[0]

def graph(data):
    val = list()
    for i in xrange(0, len(data), Np+1):
        val.append(1./2*(sum([-1**i * c for i,c in enumerate(data[i:i+Np+1])]) + sum(data[i:i+Np+1])))
    return val

# grab data from the file
data =[float(i) for i in f.readline().split()]
d1 = graph([data[i] for i in xrange(0,len(data))])

# pull out each of these things.
p.ion()
x = [x[i] for i in xrange(0,len(x), Np+1)]
lines, = p.plot(x, d1)

for i,line in enumerate(f):
    data = [float(i) for i in line.split()]
    d1 = graph([data[i] for i in xrange(0,len(data))])
    lines.set_ydata(d1)
    p.draw()
