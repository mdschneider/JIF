import sys
import os
import numpy as np
import galsim

#
# We depend on specific galsim.SED addition and magnitude operations
# Test here that these perform as we need them to.
#

path, filename = os.path.split(__file__)
datapath = os.path.abspath(os.path.join(path, "../input/"))

redshift = 1.0
filter_name = 'r'

filter_filename = os.path.join(datapath, 'LSST_{0}.dat'.format(filter_name))
bp = galsim.Bandpass(filter_filename)
bp = bp.withZeropoint('AB', effective_diameter=100. * 8.6, exptime=30)


def test_SED_small_flux():
    """
    Test that assigning a large apparent magnitude yields a small output flux of an SED
    """
    mag = 99.

    SED_name = "NGC_0695_spec"
    SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
    SED = galsim.SED(SED_filename, wave_type='Ang')

    ### Set the magnitude at zero and evaluate outputs at a different redshift
    SED = SED.atRedshift(0.).withMagnitude(target_magnitude=mag, bandpass=bp)
    SED = SED.atRedshift(redshift)

    flux = SED.calculateFlux(bp)
    # print "flux: {:8.6e}".format(flux)
    assert flux < 1.e-20

    mag_out = SED.calculateMagnitude(bp)
    # print "mag: {:8.6f}".format(mag_out)
    assert mag_out > 90.

    return None


def test_SED_add():
    """
    Test that adding an SED with small amplitude does not change the output flux/magnitude

    This mimics the operations in galsim_galaxy.py::GalsimGalaxyModel.set_mag_from_obs
    """
    mag_a = 20.
    mag_b = 99.

    SED_name = "NGC_0695_spec"
    SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
    SED_a = galsim.SED(SED_filename, wave_type='Ang')
    SED_a = SED_a.atRedshift(0.).withMagnitude(target_magnitude=mag_a, bandpass=bp)
    SED_a = SED_a.atRedshift(redshift)

    SED_name = "NGC_4125_spec"
    SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
    SED_b = galsim.SED(SED_filename, wave_type='Ang')
    SED_b = SED_b.atRedshift(0.).withMagnitude(target_magnitude=mag_b, bandpass=bp)
    SED_b = SED_b.atRedshift(redshift)

    mag_add_out = (SED_a + SED_b).calculateMagnitude(bp)

    mag_a_out = SED_a.calculateMagnitude(bp)
    mag_b_out = SED_b.calculateMagnitude(bp)

    # print "mag_a_out: {:8.6f}, mag_b_out: {:8.6f}, mag_add_out: {:8.6f}".format(
    #     mag_a_out, mag_b_out, mag_add_out)

    assert np.isclose(mag_add_out, mag_a_out)

    return None


if __name__ == "__main__":
    test_SED_small_flux()
    test_SED_add()
