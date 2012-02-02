import matplotlib.pylab as p

f = open("data.txt", "rb")

p.ion()
data =[float(i) for i in f.readline().split()]
lines, = p.plot(xrange(0,len(data)),data)
p.ylim((-1,1))
for line in f:
    data = [float(i) for i in line.split()]
    lines.set_ydata(data)
    p.draw()

