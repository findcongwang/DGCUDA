from time import sleep
import matplotlib.pylab as p

f = open("data.txt", "rb")

p.ion()
data =[float(i) for i in f.readline().split()]
d1 = [data[i] for i in xrange(0,len(data),1)]
lines, = p.plot(xrange(0,len(d1)),d1)
p.ylim((-1,1))
for i,line in enumerate(f):
    data = [float(i) for i in line.split()]
    d1 = [data[i] for i in xrange(0,len(data),1)]
    lines.set_ydata(d1)
    sleep(.5)
    p.draw()

