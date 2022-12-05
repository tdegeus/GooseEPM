import copy
import faulthandler
import unittest

import numpy as np
from GooseEPM import SystemAthermal

faulthandler.enable()


class Test_SystemAthermal(unittest.TestCase):
    """
    GooseEPM.SystemAthermal
    """

    def test_shiftImposedShear(self):
        """
        shiftImposedShear: positive shear
        """

        propagator = np.array(
            [
                [0, 0.25, 0],
                [0.25, -1, 0.25],
                [0, 0.25, 0],
            ]
        )

        system = SystemAthermal(
            propagator=propagator,
            distances_rows=np.array([-1, 0, 1]),
            distances_cols=np.array([-1, 0, 1]),
            sigmay_mean=np.ones([5, 5]),
            sigmay_std=np.zeros([5, 5]),
            seed=0,
        )

        system.sigma = 0.1 * np.ones_like(system.sigma)
        self.assertAlmostEqual(system.sigmabar, 0.1)

        system.shiftImposedShear(direction=1)
        self.assertTrue(np.allclose(system.sigma, 1))

        system.shiftImposedShear(direction=-1)
        self.assertTrue(np.allclose(system.sigma, -1))

    def test_relax(self):
        """
        Try relaxation.
        """

        propagator = np.array(
            [
                [0, 0.25, 0],
                [0.25, -1, 0.25],
                [0, 0.25, 0],
            ]
        )

        system = SystemAthermal(
            propagator=propagator,
            distances_rows=np.array([-1, 0, 1]),
            distances_cols=np.array([-1, 0, 1]),
            sigmay_mean=np.ones([5, 5]),
            sigmay_std=0.1 * np.ones([5, 5]),
            seed=0,
        )

        self.assertAlmostEqual(system.sigmabar, np.mean(system.sigma))

        # holds only if "sigmay" is sufficiently high compare to "sigma" (true here though)
        self.assertTrue(np.allclose(system.epsp, 0))

        system.shiftImposedShear(direction=1)
        system.relax()

        self.assertTrue(np.all(system.epsp >= 0))
        self.assertTrue(np.all(np.abs(system.sigma) < system.sigmay))

    def test_copy(self):
        """
        Try copy.
        """

        propagator = np.array(
            [
                [0, 0.25, 0],
                [0.25, -1, 0.25],
                [0, 0.25, 0],
            ]
        )

        system = SystemAthermal(
            propagator=propagator,
            distances_rows=np.array([-1, 0, 1]),
            distances_cols=np.array([-1, 0, 1]),
            sigmay_mean=np.ones([5, 5]),
            sigmay_std=0.1 * np.ones([5, 5]),
            seed=0,
        )

        t = system.t
        state = system.state
        sigma = np.copy(system.sigma)
        epsp = np.copy(system.epsp)

        mycopy = copy.copy(system)

        system.shiftImposedShear(direction=1)
        system.relax()

        self.assertNotEqual(system.t, t)
        self.assertNotEqual(system.state, state)
        self.assertFalse(np.allclose(system.sigma, sigma))
        self.assertFalse(np.allclose(system.epsp, epsp))

        self.assertEqual(mycopy.t, t)
        self.assertEqual(mycopy.state, state)
        self.assertTrue(np.allclose(mycopy.sigma, sigma))
        self.assertTrue(np.allclose(mycopy.epsp, epsp))

        mycopy.shiftImposedShear(direction=1)
        mycopy.relax()

        self.assertEqual(mycopy.t, system.t)
        self.assertEqual(mycopy.state, system.state)
        self.assertTrue(np.allclose(mycopy.sigma, system.sigma))
        self.assertTrue(np.allclose(mycopy.epsp, system.epsp))


if __name__ == "__main__":

    unittest.main(verbosity=2)
