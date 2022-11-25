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
A = list()
w = list()
time = np.linspace(0., 150., num=1500, endpoint=False)
u = np.sin(w0 * time)

# print(K)

for i in range(len(K)):
    plt.figure(i)
    tp, yp, xp = signal.lsim(K[i], T=time, U=u)
    
    t.append(tp)
    y.append(yp)
    x.append(xp)
    A.append((yp[1000:].max() - yp[1000:].min()) / 2)

    # w.append(abs(z1 - z0) / np.pi)

    plt.plot(t[i], y[i], 'r', label='Char. wyjściowa')
    plt.plot(t[i], u, 'b', label='Pobudzenie')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(r"k = %f, T = %f, A = %f " %(k[i], T[i], A[i]))
    plt.grid()
    plt.legend()
    plt.savefig('lab2_docs/figure%d.png' % i)

P = k[3] / (T[3] * w0 * 1j  + 1) 

plt.figure(len(K))
w, H = signal.freqresp(K[3])
plt.plot(H.real, H.imag, "b", label="Charakterystyka")
plt.plot(P.real, P.imag, "r", marker = "o", markersize=10, label=r"Punkt odp. pulsacji $\omega_0 $")
plt.xlabel('Re')
plt.ylabel('Im')
plt.legend()
plt.grid()
plt.title(r"char. amplitudowo-fazowa układu nr 4, punkt: $ \omega_0 = %f + %fj $" %(P.real, P.imag))
plt.savefig('lab2_docs/figure%d.png' % (len(K)))

for i in plt.get_fignums():
    plt.show()
