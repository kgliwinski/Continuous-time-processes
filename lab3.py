import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import integrate
import os
from mpl_toolkits.mplot3d import Axes3D
# K(s) = Ko Kr
# Kukl = K/(1+K)
# Ko = 1/(s+1)^3
# Kr = kp + ki/s
# K = (kp + ki/s )* (s+1)^3 = kp*s^3 + (ki + kp + 3)s^2 + 3*s + 1
# K/(1+K) = (s^3 + 3s^2 + (3+kp)*s + (ki + 1))/((kp + 1)*s + ki)

linspaceSize = 2500
linspaceFinish = 1500.

kp = np.arange(0.01, 2.00, 0.01)
ki = np.arange(0.01, 2.00, 0.01)

epsInt = []
epsMin = 10000.0
i = 0
Tls = np.linspace(0., linspaceFinish, num=linspaceSize)
for kps in kp:
    for kis in ki:
        K = signal.TransferFunction([0, 0, 0, kps, kis], [1, 3, 3, (1 + kps), kis])

        tp, yp = signal.step(K, T=Tls, N=1000)

        eps = 1 - yp
        os.system('clear')
        i += 1
        print(i)

        epsInt.append(integrate.simpson(np.power(eps, 2), tp))
        if epsInt[-1] < epsMin:
            tpMin = tp
            ypMin = yp
            kiMin = kis
            kpMin = kps
            epsMin = epsInt[-1]

print(epsInt)
epsIntArr = np.array(epsInt)
print(epsIntArr.min())
plt.figure(1)
plt.plot(tpMin, ypMin, label="Charakterystyka wyjściowa kp=%f ki%f" % (kpMin, kiMin))
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()
plt.show()

plt.figure(2)
ax = plt.axes(projection='3d')
xline = ki
yline = kp
zline = epsIntArr
ax.set_xlabel('ki')
ax.set_ylabel('kp')
ax.set_zlabel('eps')
ax.scatter(xline, yline, zline)
plt.show()