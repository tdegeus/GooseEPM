import matplotlib.pyplot as plt
import numpy as np
from GooseEPM import SystemAthermal

system = SystemAthermal(
    propagator=...,
    distances_rows=...,
    distances_cols=...,
    sigmay_mean=...,
    sigmay_std=...,
    seed=...,
    failure_rate=...,
)

nstep = 1000
sigma = np.empty([nstep])  # average stress
epsp = np.empty([nstep])  # average plastic strain
sigma[0] = system.sigmabar
epsp[0] = np.mean(system.epsp)

for i in range(1, nstep):
    system.eventDrivenStep()
    sigma[i] = system.sigmabar
    epsp[i] = np.mean(system.epsp)

fig, ax = plt.subplots()
ax.plot(epsp, sigma)
plt.show()
