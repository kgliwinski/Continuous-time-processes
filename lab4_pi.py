# %%
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


def discrete_pi_regulator(kp: float, ki: float, linspaceFinish: float, linspaceSize: int, plot_dest: str, show=True) -> float:
    Tls = np.linspace(0., linspaceFinish, num=linspaceSize)
    num = [0, 0, 0, kp, ki]
    den = [1, 3, 3, (1 + kp), ki]
    K = signal.lti(num, den)
    K_z = K.to_discrete(dt=0.1)
    p = signal.tf2zpk(num, den)
    print(p)
    if np.ndarray.max((np.real(p[1]))) > 0.0:
        return -1.
    tp, yp = signal.dstep(K_z, n=100)
    # eps = yp[-1] - yp
    # Q = integrate.simpson(np.power(eps, 2), tp)
    if show:
        plt.figure()
        plt.step(tp, np.squeeze(yp), where="post",
                 c='r', label='Char. wyjściowa')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title(r"Charakterystyka wyjściowa regulatora PI")
        plt.grid()
        plt.savefig(plot_dest)
        plt.draw()
    return yp[-1]


l_f = 1500.
l_s = 2500

## ! REGULATOR P : p_i == 0 ! ###

k_ps1 = [0.4, 1.2, 3.0, 4.6, 6.0]
epsis1 = []

k_ps2 = np.arange(0.1, 20., 0.1)
epsis2 = []
k_ps_end = []
for idx, kpp in enumerate(k_ps1):
    discrete_pi_regulator(kpp, 0, l_f, l_s,
                          'lab4_docs/p_regulator_%d.png' % idx)


for idx, kpp in enumerate(k_ps2):
    p = discrete_pi_regulator(kpp, 0, l_f, l_s, "", show=False)


# plt.figure()
# plt.plot(k_ps_end, epsis2)
# plt.xlabel(r'$k_p$')
# plt.ylabel(r'$\epsilon$')
# plt.title(r"Zależność $\epsilon$ od $k_p$ dla regulatora P")
# plt.grid()
# plt.savefig("lab3_docs/p_regulator_esp.png")
# plt.draw()

## ! REGULATOR PI ! ###

kp = 0.5

ki_five = [0.1, 0.2, 0.3, 0.4, 0.5]

for idx, kip in enumerate(ki_five):
    discrete_pi_regulator(
        kp, kip, l_f, l_s, 'lab4_docs/discrete_pi_regulator_%d.png' % idx)
    print(kip, idx)

ki1 = np.arange(0.01, 5.00, 0.01)
ki2 = np.arange(0.401, 0.500, 0.001)

# find_min_by_ki(kp, ki1, l_f, l_s, "lab3_docs/pi_min_1.png", "lab3_docs/pi_Q_1.png")
# find_min_by_ki(kp, ki2, l_f, l_s, "lab3_docs/pi_min_2.png", "lab3_docs/pi_Q_2.png")

plt.show()


# %%
