import numpy as np
from GooseEPM import elshelby_propagator
from GooseEPM import SystemAthermal
from GooseEPM import SystemThermal

try:
    import matplotlib.pyplot as plt

    plot = True
except ImportError:
    plot = False

L = 100

init = SystemAthermal(
    *elshelby_propagator(L=L, imposed="strain"),
    sigmay_mean=np.ones([L, L]),
    sigmay_std=0.3 * np.ones([L, L]),
    seed=0,
    init_random_stress=True,
    init_relax=True,
)

system = SystemThermal(
    *elshelby_propagator(L=L, imposed="stress"),
    sigmay_mean=np.ones([L, L]),
    sigmay_std=0.3 * np.ones([L, L]),
    seed=0,
    temperature=0.1,
    init_random_stress=False,
    init_relax=False,
)
system.sigma = np.copy(init.sigma)
system.sigmabar = 0.5

nstep = 1000
sigma = np.empty([nstep])  # average stress
epsp = np.empty([nstep])  # average plastic strain
t = np.empty([nstep])  # average plastic strain
sigma[0] = system.sigmabar
epsp[0] = np.mean(system.epsp)
t[0] = system.t

for i in range(1, nstep):
    system.makeThermalFailureSteps(20)
    sigma[i] = system.sigmabar
    epsp[i] = np.mean(system.epsp)
    t[i] = system.t

if plot:

    fig, axes = plt.subplots(ncols=2, figsize=(8 * 2, 6))

    ax = axes[0]
    ax.plot(t[1:], np.diff(epsp) / np.diff(t))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\dot{\gamma}$")
    ax.set_xlim([0, t[-1]])
    ax.set_ylim([0, ax.get_ylim()[1]])

    ax = axes[1]
    cax = ax.imshow(system.epsp, interpolation="nearest")

    cbar = fig.colorbar(cax, aspect=10)
    cbar.set_label(r"$\gamma_p$")

    plt.show()
