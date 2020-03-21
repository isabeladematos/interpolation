import numpy
import matplotlib.pyplot as plt

def lagrange(x, y, x_interp):

    w = numpy.ones(x.shape)
    for i in range(x.size):
        for k in range(x.size):
            if k != i:
                w[i] = w[i] * (x[i] - x[k])
    for i in range(x.size):
        w[i] = y[i]/w[i]
      
    L = numpy.ones(x_interp.shape)
    for i in range(x_interp.size):
        for k in range(x.size):
            L[i] = L[i] * (x_interp[i] - x[k])

    s = numpy.zeros(x_interp.shape)
    for i in range(x_interp.size):
        for k in range(x.size):
            s[i] = s[i] + w[k] / (x_interp[i] - x[k])
    for i in range(x_interp.size):
        s[i] = s[i] * L[i]

    return s

def runge(x):
    return 1.0/(1.0 + 25.0 * x**2)


#k = 5
ek = []
for k in range(1,15):
    x_interp = numpy.linspace(-1.1,1.1,100)

    x = numpy.zeros(k+1)
    for i in range(k+1):
        x[i] = -1 + 2*i/k

    pk = lagrange(x, runge(x), x_interp)
    # plt.plot(x_interp, runge(x_interp), 'r')
    # plt.plot(x_interp, pk, 'b')
    # plt.plot(x, runge(x), 'ko')
    # plt.show()

    ek.append(numpy.amax(numpy.absolute(numpy.subtract(runge(x_interp), pk))))

plt.plot(range(1,15),ek)
plt.show()
