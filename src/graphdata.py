from time import sleep
import matplotlib.pylab as p

f = open("data.txt", "rb")

def graph(data):
    val = list()
    for i in xrange(0, len(data), 3):
        val.append(data[i] - data[i+1])
        #val.append(data[i] + data[i+1])
    return val

p.ion()
data =[float(i) for i in f.readline().split()]
d1 = graph([data[i] for i in xrange(0,len(data),1)])
lines, = p.plot(xrange(0,len(d1)),d1)
p.ylim((-1,1))
for i,line in enumerate(f):
    data = [float(i) for i in line.split()]
    d1 = graph([data[i] for i in xrange(0,len(data),1)])
    lines.set_ydata(d1)
    p.draw()
