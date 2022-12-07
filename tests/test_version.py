import faulthandler
import unittest

from GooseEPM import version_dependencies

faulthandler.enable()


class Test_version(unittest.TestCase):
    def test_version_dependencies(self):

        deps = version_dependencies()
        deps = [i.split("=")[0] for i in deps]

        self.assertIn("boost", deps)
        self.assertIn("gooseepm", deps)
        self.assertIn("prrng", deps)
        self.assertIn("xtensor-python", deps)
        self.assertIn("xtensor", deps)
        self.assertIn("xtl", deps)


if __name__ == "__main__":

    unittest.main(verbosity=2)
