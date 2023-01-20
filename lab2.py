import mpmath
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# from control import TransferFunction
plt.rcParams['text.usetex'] = True

k = np.array([15., 2., -4.])
T = np.array([10., 1., 4.])
# delt = a ** 2 - 4*b

K = [signal.TransferFunction(k[i], [T[i], 1.]) for i in range(np.size(k))]
w0 = [0.5, 0.2]
# print(np.size(a))
# print(T)

t = list()
y = list()
x = list()
A = list()
w = list()
time = np.linspace(0., 150., num=1500, endpoint=False)
u = list()
u.append(np.sin(w0[0] * time))
u.append(np.sin(w0[1] * time))

# print(K)

for i in range(len(K)):
    for s in range(len(u)):
        f = 2*i + s
        plt.figure(f)
        tp, yp, xp = signal.lsim(K[i], T=time, U=u[s])

        t.append(tp)
        y.append(yp)
        x.append(xp)
        A.append((yp[1000:].max() - yp[1000:].min()) / 2)

        # w.append(abs(z1 - z0) / np.pi)

        plt.plot(t[f], y[f], 'r', label='Char. wyjściowa')
        plt.plot(t[f], u[s], 'b', label='Pobudzenie')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title(r"k = %f, T = %f, A = %f, $\omega_0 = %f$" %
                  (k[i], T[i], A[i], w0[s]))
        plt.grid()
        plt.legend()
        plt.savefig('lab2_docs/figure%d.png' % f)

takenIndex = 2
P = list()
P.append(k[takenIndex] / (T[takenIndex] * w0[0] * 1j + 1))
P.append(k[takenIndex] / (T[takenIndex] * w0[1] * 1j + 1))

print(P)

plt.figure(len(K) * 2)
w, H = signal.freqresp(K[takenIndex])
plt.plot(H.real, H.imag, "b", label="Charakterystyka")
plt.plot(P[0].real, P[0].imag, "r", marker="o", markersize=10,
         label=r"Punkt odp. pulsacji $\omega_0 = %f$" % w0[0])
plt.plot(P[1].real, P[1].imag, "g", marker="o", markersize=10,
         label=r"Punkt odp. pulsacji $\omega_0 = %f$" % w0[1])
plt.xlabel('Re')
plt.ylabel('Im')
plt.legend()
plt.grid()
plt.title(r"char. amplitudowo-fazowa układu nr 3")
plt.savefig('lab2_docs/figure%d.png' % (len(K) * 2))

for i in plt.get_fignums():
    plt.show()
