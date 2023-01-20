from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
# using scipy.signal.cont2discrete to transform
# and scipy.signal.TransferFunction
# K(s) = 1 / (s+1)^3 = 1 / (s^3 + 3s^2 + 3s + 1)
T = [1., 3., 3., 1.]
K_s = signal.lti([1.], T)
print(K_s)

# to dicretize a system we would use a
# forward euler discretization where
# s is replaced by s = (z-1)/dt where dt is sampling time

z_dt = 0.3
z_n = 30
s_linspace_max = z_dt * (z_n - 1)

# to use it with scipy we need to set method to gbt and alpha to 0
K_z = K_s.to_discrete(dt=z_dt, method="foh")

# below is the same, just different method
K_z2 = signal.cont2discrete(([1], T), dt=z_dt, method="impulse")

print("to_discrete method: ", K_z)
print("cont2discrete method: ", K_z2)

tp, yp = signal.dstep(K_z, n=z_n)
tp1, yp1 = signal.step(K_s, T=np.linspace(0., s_linspace_max, num=25000), N = 25000)
plt.figure(0)
plt.plot(tp1, np.squeeze(yp1), 'r', label='Char. układu liniowego')
plt.step(tp, np.squeeze(yp), where="post", c='b', label='Char. dyskretna')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title(r"")
plt.grid()
plt.legend()
plt.show()

