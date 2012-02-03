import matplotlib.pylab as p

f = open("data.txt", "rb")

p.ion()
data =[float(i) for i in f.readline().split()]
d1 = [data[i] for i in xrange(0,len(data),2)]
lines, = p.plot(xrange(0,len(d1)),d1)
p.ylim((-1,1))
for line in f:
    data = [float(i) for i in line.split()]
    d1 = [data[i] for i in xrange(0,len(data),2)]
    lines.set_ydata(d1)
    p.draw()

