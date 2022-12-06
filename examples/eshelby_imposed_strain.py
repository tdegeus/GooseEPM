import pathlib

import h5py
import numpy as np
from GooseEPM import elshelby_propagator
from GooseEPM import SystemAthermal

try:
    import matplotlib.pyplot as plt

    plot = True
except ImportError:
    plot = False

L = 100
system = SystemAthermal(
    *elshelby_propagator(L=L, imposed="strain"),
    sigmay_mean=np.ones([L, L]),
    sigmay_std=0.3 * np.ones([L, L]),
    seed=0,
    init_random_stress=True,
    init_relax=True,
)

nstep = 1000
sigma = np.empty([nstep])  # average stress
epsp = np.empty([nstep])  # average plastic strain
sigma[0] = system.sigmabar
epsp[0] = np.mean(system.epsp)

for i in range(1, nstep):
    system.shiftImposedShear()
    system.relaxAthermal()
    sigma[i] = system.sigmabar
    epsp[i] = np.mean(system.epsp)

base = pathlib.Path(__file__)
with h5py.File(base.parent / (base.stem + ".h5")) as file:
    assert np.allclose(file["epsp"][...], epsp)
    assert np.allclose(file["sigma"][...], sigma)

if plot:

    fig, axes = plt.subplots(ncols=2, figsize=(8 * 2, 6))

    ax = axes[0]
    ax.plot([0, 0.6], [0, 0.6], "r--")
    ax.plot(sigma + epsp, sigma)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\sigma$")

    ax = axes[1]
    cax = ax.imshow(system.epsp, interpolation="nearest")

    cbar = fig.colorbar(cax, aspect=10)
    cbar.set_label(r"$\gamma_p$")

    plt.show()
