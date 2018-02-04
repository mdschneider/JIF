#!/usr/bin/env python
# encoding: utf-8
"""
roaster_wrapper.py

Wrap the Jiffy roaster module to set initial parameters to known truth values
"""
import numpy as np
import argparse
import os
import numpy as np
from astropy.io import fits
import jiffy

def apply_shear(e, g):
    return (e + g) / (1.0 + g.conjugate() * e)

def get_truths(ifield, igal, workdir):
    """
    Get the true parameter values
    """
    scale = 0.2 # arcseconds

    infile = os.path.join(workdir, "control/ground/constant",
                          "epoch_catalog-{0:0>3}-0.fits".format(ifield))
    hdulist = fits.open(infile)
    tbdata = hdulist[1].data

    nu = tbdata.field('gal_nu')[igal]
    hlr = tbdata.field('gal_hlr')[igal]
    flux = tbdata.field('gal_flux')[igal]

    g1 = tbdata.field('g1')[igal]
    g2 = tbdata.field('g2')[igal]
    # Convert offsets in the truth catalog from pixels to arcseconds 
    # (which are the units expected by the galsim shift() method)
    dx = tbdata.field('dx')[igal] * scale
    dy = tbdata.field('dy')[igal] * scale

    e1int = tbdata.field('gal_e1')[igal]
    e2int = tbdata.field('gal_e2')[igal]

    e_sh = apply_shear(e1int + 1j*e2int, g1 + 1j*g2)

    e1 = e_sh.real
    e2 = e_sh.imag

    truths = np.array([e1, e2, hlr, flux, nu, dx, dy])
    return truths

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str,
                        default="../config/jiffy.yaml",
                        help="Name of a configuration file listing inputs." +
                        "If specified, ignore other command line flags.")

    parser.add_argument("--footprint_number", type=int, default=0,
                        help="The footprint number to load from input")

    parser.add_argument("--field", type=int, default=0)

    parser.add_argument("--workdir", type=str, default="./")

    args = parser.parse_args()

    print "Starting roaster wrapper"
    truths = get_truths(args.field, args.footprint_number, args.workdir)

    rstr = jiffy.roaster.init_roaster(args)
    print "Setting truths: ", truths
    rstr.set_params(truths)

    print "Executing sampler"
    jiffy.roaster.do_sampling(args, rstr)
    print "Finished roaster wrapper"
    return None

if __name__ == '__main__':
    main()
