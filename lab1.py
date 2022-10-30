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

figure, axis = plt.subplots((len(T) // 2), 2)
figure.subplots_adjust(hspace=1.2)
figure2, axis2 = plt.subplots((len(roots) // 2), 2)
figure2.subplots_adjust(hspace=1.2)
for i in range(len(T)):
    tp, yp = signal.step(T[i], T=np.linspace(0., 15., num=2500), N=1000)
    t.append(tp)
    y.append(yp)

    axis[i//2, i % 2].plot(t[i], y[i])
    axis[i//2, i % 2].set_xlabel('Time [s]')
    axis[i//2, i % 2].set_ylabel('Amplitude')
    axis2[i//2, i % 2].set_xlabel('Re')
    axis2[i//2, i % 2].set_ylabel('Im')
    if delt[i] < 0.:
        axis[i//2, i % 2].set_title("Delta < 0")
        axis2[i//2, i % 2].set_title("Delta < 0")
    elif delt[i] == 0.:
        axis[i//2, i % 2].set_title("Delta == 0")
        axis2[i//2, i % 2].set_title("Delta == 0")
    else:
        axis[i//2, i % 2].set_title('Delta > 0')
        axis2[i//2, i % 2].set_title("Delta > 0")
    axis[i//2, i % 2].grid()

    axis2[i//2, i % 2].plot(np.real(roots[i]), np.imag(roots[i]), 'bo')
    axis2[i//2, i % 2].grid()



plt.show()

