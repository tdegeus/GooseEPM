import numpy as np
import scipy.fftpack as fft

from ._GooseEPM import *  # noqa: F401, F403


def elshelby_propagator(L: int, imposed="strain") -> np.ndarray:
    """
    Generates a periodic Eshelby-like propagator.
    The convention that is followed is:

    -   ``dx = dx = [0, 1, 2, ..., L / 2, -L / 2 + 1, ..., -1]``
    -   ``propagator[0, 0] = -1``
    -   Imposed strain: ``np.mean(propagator) = -1 / propagator.size``
    -   Imposed stress: ``np.mean(propagator) = 0``

    See also: Rossi, Biroli, Ozawa, Tarjus, Zamponi (2022), https://arxiv.org/abs/2204.10683

    :param L: Linear size (in pixels) of the square propagator.
    :param imposed: ``"strain"`` or ``"stress"``.
    :return: ``(propagator, dx, dy)``
    """
    qx = fft.fftfreq(L) * 2 * np.pi
    qx, qy = np.meshgrid(qx, qx)

    a = 2 - 2 * np.cos(qx)
    b = 2 - 2 * np.cos(qy)
    q = (a + b) ** 2
    q[:, 0] = 1
    q[0, :] = 1

    G_t = -4 * a * b / q
    G_t[:, 0] = 0
    G_t[0, :] = 0

    if imposed == "strain":
        g = -1 / (L**2 - 1) * (np.sum(G_t) - G_t[0, 0])
    elif imposed == "stress":
        g = -1 / (L**2) * (np.sum(G_t) - G_t[0, 0])
    else:
        raise ValueError("Unknown imposed quantity: " + str(imposed))

    G_t /= g
    G_t[0, 0] = -1

    G = np.copy(fft.ifft2(G_t).real)

    dx = np.arange(L)
    dx = np.where(dx > L / 2, dx - L, dx)

    # fine-tuning: getting rid of rounding errors
    if imposed == "strain":
        for i in range(10):
            G -= (np.sum(G) - G[0, 0]) / G.size
            G[0, 0] = -1
    else:
        for i in range(10):
            G -= (np.sum(G) - G[0, 0] - 1) / G.size
            G[0, 0] = -1

    return G, dx, dx
