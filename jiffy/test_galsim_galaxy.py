import unittest
# import numpy as np
# import galsim
import galsim_galaxy

class TestGalSimGalaxyModel(unittest.TestCase):

    def test_init_values(self):
        gg = galsim_galaxy.GalsimGalaxyModel()
        self.assertEqual(gg.psf.__class__.__name__, "Kolmogorov")
        self.assertAlmostEqual(gg.params.nu[0], 0.5)
        self.assertAlmostEqual(gg.params.hlr[0], 1.0)
        self.assertAlmostEqual(gg.params.e1[0], 0.0)
        self.assertAlmostEqual(gg.params.e2[0], 0.0)
        self.assertAlmostEqual(gg.params.flux[0], 1.0)
        self.assertAlmostEqual(gg.params.dx[0], 0.0)
        self.assertAlmostEqual(gg.params.dy[0], 0.0)
        self.assertAlmostEqual(gg.psf.fwhm, 0.6)
        self.assertAlmostEqual(gg.psf.flux, 1.0)

    def test_set_params(self):
        gg = galsim_galaxy.GalsimGalaxyModel()
        gg.set_params([0.1, -0.5])
        self.assertAlmostEqual(gg.params.e1[0], 0.1)
        self.assertAlmostEqual(gg.params.e2[0], -0.5)
        with self.assertRaises(AssertionError):
            gg.set_params([0.1])

    def test_default_image(self):
        gg = galsim_galaxy.GalsimGalaxyModel()
        image = gg.get_image(64, 64)
        self.assertAlmostEqual(image.array.sum(), 0.99832934)
        self.assertAlmostEqual(image.array.max(), 0.0094704079)

if __name__ == "__main__":
    unittest.main()
