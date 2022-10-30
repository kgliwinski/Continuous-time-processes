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

figure, axis = plt.subplots((len(K) // 2), 2)
figure.subplots_adjust(hspace=1.2)

time = np.linspace(0., 150., num=2500, endpoint=False)
u = np.sin(w0 * time)

print(K)

for i in range(len(K)):
    tp, yp, xp = signal.lsim(K[i], T=time, U=u)
    t.append(tp)
    y.append(yp)
    x.append(xp)

    axis[i//2, i % 2].plot(t[i], y[i])
    axis[i//2, i % 2].set_xlabel('Time [s]')
    axis[i//2, i % 2].set_ylabel('Amplitude')
    axis[i//2, i % 2].set_title("k = '%f', T = '%f'" %( k[i], T[i]))
    axis[i//2, i % 2].grid()

plt.show()
