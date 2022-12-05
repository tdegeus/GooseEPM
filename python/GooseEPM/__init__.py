from ._GooseEPM import *  # noqa: F401, F403
import scipy.fftpack as fft
import numpy as np

def generate_propagator(L, method = 'rossi'):
    """Generates a periodic Eshelby-like propagator, according to the method described in Rossi et al. (2022). It is meant to be used for strain-controlled protocol.

    Args:
        L (int): length of the system
        method (str, optional): As legacy, 'marko' can be used to generate the propagator like in the original EPM code provided by Marko Popovic. Defaults to 'rossi'.

    Returns:
        ndarray: The propagator in real space, centered in [0,0].
    """
    qx = fft.fftfreq(L) * 2*np.pi
    qy = qx.copy()
    qx, qy = np.meshgrid(qx,qy)

    a = 2-2*np.cos(qx)
    b = 2-2*np.cos(qy)

    #G_E is the Eshelby propagator. Here, we use it through its FT.
    G_E_t = -4*a*b / (a+b)**2
    
    if (method == 'marko'):
        G_t = G_E_t
        G_t[0,0] = 0
    
    if (method == 'rossi'):
        G_t = G_E_t
        G_t[0,0] = 0 #temporary value to avoid NaN
        g = -1/(L**2 - 1) * (np.sum(G_t) - G_t[0,0])
        G_t /= g
        G_t[0,0] = -1

    G = fft.ifft2(G_t)
    return G.real