import matplotlib.pyplot as plt
import numpy as np
import prrng
from GooseEPM import SystemAthermal

system = SystemAthermal(
    propagator=...,
    dx=...,
    dy=...,
    sigmay_mean=...,
    sigmay_std=...,
    seed=...,
    failure_rate=...,
    alpha=...,
    sigmabar=...,
    fixed_stress=...,
)

gen = prrng.pcg32(0)
system.sigma = gen.normal(system.shape, mu=0, std=0.1)
system.initSigmaPropogator()

nstep = 1000
sigma = np.empty([nstep])
epsp = np.empty([nstep])
sigma[0] = system.sigmabar
epsp[0] = np.mean(system.epsp)

for i in range(1, nstep):
    system.eventDrivenStep()
    sigma[i] = system.sigmabar
    epsp[i] = np.mean(system.epsp)

fig, ax = plt.subplots()
ax.plot(epsp, sigma)
plt.show()
