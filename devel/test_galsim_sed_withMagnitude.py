#!/usr/bin/env python
# encoding: utf-8
"""
test_galsim_sed_withMagnitude.py

Created by Michael Schneider on 2016-02-24
"""

import sys
import os.path
import galsim

def main():
    mag = 20.
    redshift = 1.

    path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(path, "../input/"))

    filter_name = 'r'
    filter_filename = os.path.join(datapath, 'LSST_{0}.dat'.format(filter_name))
    bp = galsim.Bandpass(filter_filename)
    bp = bp.withZeropoint('AB', effective_diameter=100. * 8.6, exptime=30)

    SED_name = 'CWW_E_ext'
    SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
    SED = galsim.SED(SED_filename, wave_type='Ang')

    SED1 = SED.withMagnitude(target_magnitude=mag, bandpass=bp).atRedshift(redshift)
    print "SED1 mag: {:8.6f}".format(SED1.calculateMagnitude(bp))

    SED2 = SED.atRedshift(redshift).withMagnitude(target_magnitude=mag, bandpass=bp)
    print "SED2 mag: {:8.6f}".format(SED2.calculateMagnitude(bp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
