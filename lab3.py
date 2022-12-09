import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import integrate
import os
from mpl_toolkits.mplot3d import Axes3D
# K(s) = Ko * Kr
# Kukl = K/(1+K)
# Ko = 1/(s+1)^3
# Kr = kp + ki/s
# K = (kp + ki/s )* (s+1)^3 = kp*s^3 + (ki + kp + 3)s^2 + 3*s + 1
# K/(1+K) = (s^3 + 3s^2 + (3+kp)*s + (ki + 1))/((kp + 1)*s + ki)


def pi_regulator(kp: float, ki: float, linspaceFinish: float, linspaceSize: int, plot_dest: str, show=True) -> float:
    Tls = np.linspace(0., linspaceFinish, num=linspaceSize)
    num = [0, 0, 0, kp, ki]
    den = [1, 3, 3, (1 + kp), ki]
    K = signal.TransferFunction(num, den)
    p = signal.tf2zpk(num, den)
    print(p)
    if np.ndarray.max((np.real(p[1]))) > 0.0:
        return -1.
    tp, yp = signal.step(K, T=Tls, N=1000)
    eps = yp[-1] - yp
    Q = integrate.simpson(np.power(eps, 2), tp)
    if show:
        plt.figure()
        plt.plot(tp, yp)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title(r"Charakterystyka wyjściowa regulatora PI: $k_p = %f $, $k_i = %f $ " "\n" " $Q = %f $, $\epsilon_{ust} = %f$" % (
            kp, ki, Q, yp[-1]))
        plt.grid()
        plt.savefig(plot_dest)
        plt.draw()
    return yp[-1]


def find_min_by_ki(kp: float, ki: np.ndarray, linspaceFinish: float, linspaceSize: int, plot_dest1: str, plot_dest2: str):
    Tls = np.linspace(0., linspaceFinish, num=linspaceSize)
    Q = []
    epsMin = 10000.0
    i = 0
    # for kps in kp:
    for kis in ki:
        K = signal.TransferFunction(
            [0, 0, 0, kp, kis], [1, 3, 3, (1 + kp), kis])

        tp, yp = signal.step(K, T=Tls, N=1000)

        eps = yp[-1] - yp
        os.system('clear')
        i += 1
        print(i)

        Q.append(integrate.simpson(np.power(eps, 2), tp))
        if Q[-1] < epsMin:
            tpMin = tp
            ypMin = yp
            kiMin = kis
            kpMin = kp
            epsMin = Q[-1]

    print(Q)
    epsIntArr = np.array(Q)
    print(epsIntArr.min())
    plt.figure()
    plt.plot(tpMin, ypMin, label=r"Charakterystyka wyjściowa $k_p=%f$ $k_i = %f$" % (
        kpMin, kiMin))
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(r"Charakterystyka wyjściowa dla minimalnego $Q$")
    plt.grid()
    plt.legend()
    plt.savefig(plot_dest1)
    plt.draw()

    plt.figure()
    plt.plot(ki, epsIntArr)
    plt.xlabel(r'$k_i$')
    plt.ylabel(r'$Q$')
    plt.title(r"Zależność $Q$ od $k_i$ ")
    plt.grid()
    plt.savefig(plot_dest2)
    plt.draw()


l_f = 1500.
l_s = 2500

## ! REGULATOR P : p_i == 0 ! ###

k_ps1 = [0.4, 1.2, 3.0, 4.6, 6.0]
epsis1 = []

k_ps2 = np.arange(0.1, 20., 0.1)
epsis2 = []
k_ps_end = []
for idx, kpp in enumerate(k_ps1):
    epsis1.append(pi_regulator(kpp, 0, l_f, l_s,
                  'lab3_docs/p_regulator_%d.png' % idx))
    if epsis1[-1] == -1.:
        print("Unstable")

for idx, kpp in enumerate(k_ps2):
    p = pi_regulator(kpp, 0, l_f, l_s, "", show=False)
    if p == -1.:
        print("Unstable")
        np.delete(k_ps2, idx)
    else:
        epsis2.append(p)
        k_ps_end.append(kpp)

plt.figure()
plt.plot(k_ps_end, epsis2)
plt.xlabel(r'$k_p$')
plt.ylabel(r'$\epsilon$')
plt.title(r"Zależność $\epsilon$ od $k_p$ dla regulatora P")
plt.grid()
plt.savefig("lab3_docs/p_regulator_esp.png")
plt.draw()

## ! REGULATOR PI ! ###

kp = 0.5

ki_five = [0.1, 0.2, 0.3, 0.4, 0.5]

for idx, kip in enumerate(ki_five):
    pi_regulator(kp, kip, l_f, l_s, 'lab3_docs/pi_regulator_%d.png' % idx)
    print(kip, idx)

ki1 = np.arange(0.01, 5.00, 0.01)
ki2 = np.arange(0.401, 0.500, 0.001)

find_min_by_ki(kp, ki1, l_f, l_s, "lab3_docs/pi_min_1.png", "lab3_docs/pi_Q_1.png")
find_min_by_ki(kp, ki2, l_f, l_s, "lab3_docs/pi_min_2.png", "lab3_docs/pi_Q_2.png")

plt.show()
