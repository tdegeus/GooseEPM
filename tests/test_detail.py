import unittest

import numpy as np
from GooseEPM import detail


class Test_detail(unittest.TestCase):
    def test_create_distance_lookup(self):

        check = np.array([3, 4, 5, 6, 0, 1, 2, 4, 5, 6, 0, 1, 2])
        distance = np.array([-3, -2, -1, 0, 1, 2, 3])

        self.assertTrue(np.all(np.equal(check, detail.create_distance_lookup(distance))))


if __name__ == "__main__":

    unittest.main(verbosity=2)
