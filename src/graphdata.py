from time import sleep
import matplotlib.pylab as p
import numpy as np

f = open("data.txt", "rb")

Np = int(f.readline())
x  = [float(i) for i in f.readline().split()]
dx = x[Np+1] - x[0]

def lagrange(x, i):
    if i == 0: 
        return np.ones(len(x))
    elif i == 1: 
        return x;
    elif i == 2: 
        return (3.*x**2 -1.) / 2.;
    elif i == 3: 
        return (5.*x**3 - 3.*x) / 2.;
    elif 4: 
        return  (35.*x**4 - 30.*x**2 + 3.)/8.;
    elif 5: 
        return  (63.*x**5 - 70.*x**3 + 15.*x)/8.;
    elif 6: 
        return  (231.*x**6 - 315.*x**4 + 105.*x**2 -5.)/16.;
    elif 7: 
        return  (429.*x**7 - 693.*x**5 + 315.*x**3 - 35.*x)/16.;
    elif 8: 
        return  (6435.*x**8 - 12012.*x**6 + 6930.*x**4 - 1260.*x**2 + 35.)/128.;
    elif 9: 
        return  (12155.*x**9 - 25740.*x**7 + 18018*x**5 - 4620.*x**3 + 315.*x)/128.;
    elif 10: 
        return (46189.*x**10 - 109395.*x**8 + 90090.*x**6 - 30030.*x**4 + 3465.*x**2 - 63.)/256.;

def graph(data):
    val = list()
    for i in xrange(0, len(data), Np+1):
        x = np.arange(-1,1+0.5,0.5)
        interpolate = sum([c * lagrange(x,i) for i,c in enumerate(data[i:i+Np+1])])
        val.append(interpolate)
    return val

# grab data from the file
data =[float(i) for i in f.readline().split()]
d1 = graph([data[i] for i in xrange(0,len(data))])

# pull out each of these things.
p.ion()
lines = []
xstart = x[0]
for i in xrange(0, len(data), Np+1):
    xscaled = np.arange(xstart, xstart+dx, dx/5)[:5]
    xstart  = xstart + dx 
    l, = p.plot(xscaled, d1[i/(Np+1)])
    p.xlim((-1,1))
    p.ylim((-1,1))
    lines.append(l)

for i,line in enumerate(f):
    data = [float(i) for i in line.split()]
    d1 = graph([data[i] for i in xrange(0,len(data))])
    for i in xrange(0, len(data)/(Np+1)):
        lines[i].set_ydata(d1[i])
    p.draw()
