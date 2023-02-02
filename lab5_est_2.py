# %%

import numpy as np
import matplotlib.pyplot as plt
N = 2000

ag = 1.0
bg = 1.0
tetag = np.array([ag, bg])

PN = np.zeros((2, 2))
PN[0, 0] = 100
PN[1, 1] = 100

Yk = 0
teta = np.zeros((2, 1))
Uk = np.random.rand() * 10
Zk = 2 * np.random.rand() - 1

phi = np.array([Yk, Uk])

Yk = np.dot(phi.T, tetag) + Zk
PN = (1 / 1) * (PN - np.dot(np.dot(np.dot(PN, phi), phi.T), PN) /
                (1 + np.dot(np.dot(phi.T, PN), phi)))

c = np.dot(PN, phi)
d = Yk - np.dot(phi.T, teta)
print(teta)

f = np.multiply(c, d)
f = np.atleast_2d(f).T
print(f)
teta = np.add(teta, f)
teta[0] = tetag[0]
Un = np.zeros(N)
Un[0] = Uk

print(teta)

for k in range(1, N//2):
    Un[k] = (np.sin(0.1 * k) - teta[1] * Un[k-1]) / teta[0]
    Zk = 2 * np.random.rand() - 1

    phi = np.array([Yk, Un[k]])

    Yk = np.dot(phi.T, tetag) + Zk
    PN = (1 / 1) * (PN - np.dot(np.dot(np.dot(PN, phi), phi.T), PN) /
                    (1 + np.dot(np.dot(phi.T, PN), phi)))
    
    c = np.dot(PN, phi)
    d = Yk - np.dot(phi.T, teta)
    f = np.multiply(c, d)
    f = np.atleast_2d(f).T
    teta = np.add(teta, f)

    Yz = Un[k-1] * teta[1] + Un[k] * teta[0]
    plt.plot(k, Yz * k**0, 'k.')
    plt.plot(k, np.sin(k * 0.1), 'g.')
    plt.plot(k, teta[0] * k**0, 'r.')
    plt.plot(k, tetag[0] * k**0, 'b.')
    plt.legend(['Y', 'Y estymacja', 'wsp. a0', 'wsp. a1'])
    plt.title('Real and estimated Y=sin(0.1*k)')
    plt.xlabel('k')
    plt.ylabel('Yk')

teta[0] = 2.
for k in range(1001, N):
    Un[k] = (np.sin(0.1 * k) - teta[1] * Un[k-1]) / teta[0]
    Zk = 2 * np.random.rand() - 1

    phi = np.array([Yk, Un[k]])

    Yk = np.dot(phi.T, tetag) + Zk
    PN = (1 / 1) * (PN - np.dot(np.dot(np.dot(PN, phi), phi.T), PN) /
                    (1 + np.dot(np.dot(phi.T, PN), phi)))
    
    c = np.dot(PN, phi)
    d = Yk - np.dot(phi.T, teta)
    f = np.multiply(c, d)
    f = np.atleast_2d(f).T
    teta = np.add(teta, f)

    Yz = Un[k-1] * teta[1] + Un[k] * teta[0]
    plt.plot(k, Yz * k**0, 'k.')
    plt.plot(k, np.sin(k * 0.1), 'g.')
    plt.plot(k, teta[0] * k**0, 'r.')
    plt.plot(k, tetag[0] * k**0, 'b.')
    plt.legend(['Y', 'Y estymacja', 'wsp. a0', 'wsp. a1'])
    plt.title('Real and estimated Y=sin(0.1*k)')
    plt.xlabel('k')
    plt.ylabel('Yk')

plt.show()
print(teta[0], teta[1])
# %%
