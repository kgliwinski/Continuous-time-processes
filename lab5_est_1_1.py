# %%
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# z_k = np.random.uniform
N = 1000

a0 = 1.
a1 = 1.

theta = np.array([[a0], [a1]])
print(theta)

lamb = 1.

PN = np.zeros([2, 2])
PN[1, 1] = 100.
PN[0, 0] = 100.

Yk = 0
Uk = 0
teta = np.zeros((2, 1))
print(teta)

phi = np.array([[Yk], [Uk]])
# print(phi, phi.transpose())
Un = np.empty(N, dtype=float)
Un[0] = 0.

print(np.dot(PN, phi))

teta = np.add(teta, np.dot(np.dot(PN, phi),
              (Yk - np.dot((phi.transpose()), teta))))
teta[0] = theta[1]
print(teta)

# yk = a0*Uk + a1*Uk-1 + zk
# Uk = (yk - zk - a1*Uk-1)/a0
for k in range(1, N):
    Zk = np.random.uniform(0., 1.)
    Un[k] = (np.sin(0.1*k) - teta[1]*Un[k-1] - Zk) / teta[0]
    print(Un[k], k)
    phi = np.array([Yk, Un[k]])

    Yk = np.dot((phi.transpose()),theta)+Zk
    PN = (1/lamb)*(PN - (np.dot(np.dot(PN,np.dot(phi, (phi.T))),PN)) /
                   (1+(phi.transpose())*PN*phi))
    teta = teta + PN*phi*(Yk - (phi.transpose())*teta)

plt.figure(1)
plt.scatter(np.linspace(0, N, num=N), Un)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title(r"")
plt.grid()
# plt.legend()
plt.show()

# %%
