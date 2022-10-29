import mpmath
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# from control import TransferFunction

a = np.array([2., 2., 4.])
b = np.array([2., 1., 2.])
# delt = a ** 2 - 4*b
delt = np.array(np.subtract(np.power(a, 2), np.multiply(b, 4)))

print(delt)

T = [signal.TransferFunction(1, [1., a[i], b[i]]) for i in range(np.size(a))]

print(np.size(a))
print(T)

t = list()
y = list()

figure, axis = plt.subplots(3, 1)
figure.subplots_adjust(hspace=1.2)

for i in range(len(T)):
    tp, yp = signal.step(T[i])
    t.append(tp)
    y.append(yp)

    axis[i].plot(t[i], y[i])
    axis[i].set_xlabel('Time [s]')
    axis[i].set_ylabel('Amplitude')
    if delt[i] < 0.:
        axis[i].set_title("Delta < 0")
    elif delt[i] == 0.:
        axis[i].set_title("Delta == 0")
    else:
        axis[i].set_title('Delta > 0')
    axis[i].grid()

plt.show()

