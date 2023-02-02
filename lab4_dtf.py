# %%
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# Discrete transfer function
# will use the signal.TransferFunction method
# on discrete systems

# 1 / z + 1
T1 = [[1.], [1., 1.]]

# z / z + 2
T2 = [[1., 0.], [1., 2.]]

# z^2 / 3z^2 + 4
T3 = [[1., 0., 0.], [3., 0., 4.]]

K1 = signal.dlti(T1[0], T1[1])
K2 = signal.dlti(T2[0], T2[1])
K3 = signal.dlti(T3[0], T3[1])

print(K1, K2, K3)

l = [K1, K2, K3]

# simulation

time = 100.  # s

for i, k in enumerate(l):
    tp, yp = signal.dstep(k, n=50)
    fig = plt.figure(i)
    plt.step(tp, np.squeeze(yp), where="post", c='r', label='Char. wyjściowa')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(r"")
    plt.grid()
    plt.legend()
    plt.savefig('lab4_docs/figure%d.png' % i)

for i in plt.get_fignums():
    plt.show()

