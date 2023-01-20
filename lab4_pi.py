from scipy import signal
import numpy as np

T = [1., 3., 3., 1.]
K_s = signal.lti([1.], T)

z_dt = 0.3
z_n = 30
s_linspace_max = z_dt * (z_n - 1)

K_z = signal.cont2discrete(([1], T), dt = z_dt, method="foh")
# K_R = k_p + k_i(1 / (z - 1)) = (k_p*z + k_i - k_p) / (z - 1)
k_p = 1.
k_i = 2.

PI = signal.dlti([[k_p, (k_i - k_p)], [1., -1.]])



K_r = signal.dlti()