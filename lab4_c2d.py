from scipy import signal
import numpy as np
# using scipy.signal.cont2discrete to transform
# and scipy.signal.TransferFunction
# K(s) = 1 / (s+1)^3 = 1 / (s^3 + 3s^2 + 3s + 1)
T = [1., 3., 3., 1.]
K_s = signal.lti([1.], T)
print(K_s)

# to dicretize a system we would use a
# forward euler discretization where
# s is replaced by s = (z-1)/dt where dt is sampling time

# to use it with scipy we need to set method to gbt and alpha to 0
K_z = K_s.to_discrete(dt=0.5, method="gbt", alpha=0)
K_z2 = signal.cont2discrete(([1], T), dt=0.5, method="gbt", alpha=0)

print("to_discrete method: ", K_z)
print("cont2discrete method: ", K_z2)
