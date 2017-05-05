import argparse
import os
import numpy as np
from astropy.io import fits

K_TOPDIR = "control/ground/constant"

def get_truths(ifield, igal):
    """
    Get the true parameter values
    """
    scale = 0.2 # arcseconds

    infile = os.path.join(K_TOPDIR,
                          "epoch_catalog-{0:0>3}-0.fits".format(ifield))
    hdulist = fits.open(infile)
    tbdata = hdulist[1].data

    nu = 0.5
    hlr = 1.0
    flux = 1.0
    g1 = tbdata.field('g1')[igal]
    g2 = tbdata.field('g2')[igal]
    # Convert offsets in the truth catalog from pixels to arcseconds 
    # (which are the units expected by the galsim shift() method)
    dx = tbdata.field('dx')[igal] * scale
    dy = tbdata.field('dy')[igal] * scale

    e1int = tbdata.field('gal_e1')[igal]
    e2int = tbdata.field('gal_e2')[igal]

    e1 = g1 + e1int
    e2 = g2 + e2int

    truths = np.array([nu, hlr, e1, e2, flux, dx, dy])
    return truths

def update_config(truths, config_file):
    import ConfigParser

    Config = ConfigParser.ConfigParser()
    Config.add_section('parameters')
    Config.set('parameters', 'nu', truths[0])
    Config.set('parameters', 'hlr', truths[1])
    Config.set('parameters', 'e1', truths[2])
    Config.set('parameters', 'e2', truths[3])
    Config.set('parameters', 'flux', truths[4])
    Config.set('parameters', 'dx', truths[5])
    Config.set('parameters', 'dy', truths[6])
    with open(config_file, 'wb') as cfile:
        Config.write(cfile)
    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--field", type=int, default=0)

    parser.add_argument("--gal", type=int, default=0)

    parser.add_argument("--config_file", type=str, default="jiffy_params.cfg")

    args = parser.parse_args()

    truths = get_truths(args.field, args.gal)
    update_config(truths, args.config_file)

if __name__ == '__main__':
    main()