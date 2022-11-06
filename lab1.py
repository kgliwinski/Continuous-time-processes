import mpmath
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# from control import TransferFunction

a = np.array([5., -1., 2., -4., -2., 9.])
b = np.array([6., -6., 1., 4., 20., 360.])
# delt = a ** 2 - 4*b
delt = np.array(np.subtract(np.power(a, 2), np.multiply(b, 4)))
roots = [np.roots([1., a[i], b[i]]) for i in range(np.size(a))]

print(delt)
print(roots)


T = [signal.TransferFunction(1, [1., a[i], b[i]]) for i in range(np.size(a))]

# print(np.size(a))
# print(T)

t = list()
y = list()

for i in range(len(T)):
    plt.figure(i)
    figure, axis = plt.subplots(2, 1)
    figure.subplots_adjust(hspace=0.4)
    figure.subplots_adjust(wspace=0.8)
    figure.set_size_inches(12, 8)
    tp, yp = signal.step(T[i], T=np.linspace(0., 15., num=2500), N=1000)
    t.append(tp)
    y.append(yp)
    
    figure.suptitle("a = '%.2f', b = '%.2f', delta = '%.2f'" % (a[i], b[i], delt[i]), fontsize='xx-large')
    axis[0].plot(t[i], y[i])
    axis[0].set_xlabel('Time [s]')
    axis[0].set_ylabel('Amplitude')
    axis[1].set_xlabel('Re')
    axis[1].set_ylabel('Im')
    axis[0].grid()

    axis[1].plot(np.real(roots[i]), np.imag(roots[i]), 'bo')
    axis[1].grid()
    figure.savefig('lab1_docs/figure%d.png' % i)

for i in plt.get_fignums():
    plt.show()
    
