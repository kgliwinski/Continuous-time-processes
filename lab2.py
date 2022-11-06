import mpmath
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# from control import TransferFunction
plt.rcParams['text.usetex'] = True

k = np.array([15., -1., 2., -4., -2., 9.])
T = np.array([10., -6., 1., 4., 20., 360.])
# delt = a ** 2 - 4*b

K = [signal.TransferFunction(k[i], [T[i], 1.]) for i in range(np.size(k))]
w0 = 0.8
# print(np.size(a))
# print(T)

t = list()
y = list()
x = list()

time = np.linspace(0., 150., num=2500, endpoint=False)
u = np.sin(w0 * time)

print(K)

for i in range(len(K)):
    plt.figure(i)
    tp, yp, xp = signal.lsim(K[i], T=time, U=u)
    t.append(tp)
    y.append(yp)
    x.append(xp)

    plt.plot(t[i], y[i], 'b')
    plt.plot(t[i], u, 'r')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title("k = '%f', T = '%f'" %(k[i], T[i]))
    plt.grid()
    plt.savefig('lab2_docs/figure%d.png' % i)

for i in plt.get_fignums():
    plt.show()