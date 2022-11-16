import faulthandler
import unittest

import numpy as np
from GooseEPM import SystemAthermal

faulthandler.enable()


class Test_detail(unittest.TestCase):
    def test_simple_propogator(self):

        propagator = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0.25, 0, 0],
            [0, 0.25, -1, 0.25, 0],
            [0, 0, 0.25, 0, 0],
            [0, 0, 0, 0, 0],
        ])

        system = SystemAthermal(
            propagator=propagator,
            distances_rows=np.array([-2, -1, 0, 1, 2]),
            distances_cols=np.array([-2, -1, 0, 1, 2]),
            sigmay_mean=np.ones_like(propagator),
            sigmay_std=0.1 * np.ones_like(propagator),
            seed =0
        )

        system.eventDrivenSteps(100)




if __name__ == "__main__":

    unittest.main(verbosity=2)
