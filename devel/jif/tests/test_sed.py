import unittest
import sys
import os
import numpy as np
import galsim

path, filename = os.path.split(__file__)
datapath = os.path.abspath(os.path.join(path, "../jif/input/"))

#
# We depend on specific galsim.SED addition and magnitude operations
# Test here that these perform as we need them to.
#
class TestSEDInstances(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSEDInstances, self).__init__(*args, **kwargs)
        self.redshift = 1.0
        self.filter_name = 'r'

        filter_filename = os.path.join(datapath, 'LSST_{0}.dat'.format(self.filter_name))
        bp = galsim.Bandpass(filter_filename, wave_type='nm')
        self.bp = bp.withZeropoint('AB', effective_diameter=100. * 8.6, exptime=30)

    def test_SED_small_flux(self):
        """
        Test that assigning a large apparent magnitude yields a small output flux of an SED
        """
        mag = 99.

        SED_name = "NGC_0695_spec"
        SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
        SED = galsim.SED(SED_filename, wave_type='Ang', flux_type='flambda')

        ### Set the magnitude at zero and evaluate outputs at a different redshift
        SED = SED.atRedshift(0.).withMagnitude(target_magnitude=mag, bandpass=self.bp)
        SED = SED.atRedshift(self.redshift)

        flux = SED.calculateFlux(self.bp)
        # print "flux: {:8.6e}".format(flux)
        self.assertLess(flux, 1.e-20)

        mag_out = SED.calculateMagnitude(self.bp)
        # print "mag: {:8.6f}".format(mag_out)
        self.assertGreater(mag_out, 90.)

        return None

    def test_SED_add(self):
        """
        Test that adding an SED with small amplitude does not change the output flux/magnitude

        This mimics the operations in galsim_galaxy.py::GalsimGalaxyModel.set_mag_from_obs
        """
        mag_a = 20.
        mag_b = 99.

        SED_name = "NGC_0695_spec"
        SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
        SED_a = galsim.SED(SED_filename, wave_type='Ang', flux_type='flambda')
        SED_a = SED_a.atRedshift(0.).withMagnitude(target_magnitude=mag_a, bandpass=self.bp)
        SED_a = SED_a.atRedshift(self.redshift)

        SED_name = "NGC_4125_spec"
        SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
        SED_b = galsim.SED(SED_filename, wave_type='Ang', flux_type='flambda')
        SED_b = SED_b.atRedshift(0.).withMagnitude(target_magnitude=mag_b, bandpass=self.bp)
        SED_b = SED_b.atRedshift(self.redshift)

        mag_add_out = (SED_a + SED_b).calculateMagnitude(self.bp)

        mag_a_out = SED_a.calculateMagnitude(self.bp)
        mag_b_out = SED_b.calculateMagnitude(self.bp)

        # print "mag_a_out: {:8.6f}, mag_b_out: {:8.6f}, mag_add_out: {:8.6f}".format(
        #     mag_a_out, mag_b_out, mag_add_out)

        self.assertAlmostEqual(mag_add_out, mag_a_out, 4)

        return None


if __name__ == "__main__":
    unittest.main()