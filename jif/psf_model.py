#!/usr/bin/env python
# encoding: utf-8
"""
psf_model.py

Parametric PSF models to be included with galsim_galaxy simulated images.

See also galsim_galaxy.py for the expected interface and parameter handling.
"""
import os
import math
import numpy as np
import warnings
import galsim


k_galsim_psf_types = [('fwhm', '<f8'), ('e', '<f8'), ('beta', '<f8')]
k_galsim_psf_defaults = [(0.8, 0.0, 0.0)]


class PSFModel(object):
    """
    Parametric PSF models for marginalization in galaxy model fitting.
    """
    def __init__(self, active_parameters=['fwhm'], gsparams=None):
        self.active_parameters = active_parameters
        self.gsparams = gsparams

        self.params = np.core.records.array(k_galsim_psf_defaults,
            dtype=k_galsim_psf_types)

    def get_params(self):
        """
        Return a list of active model parameter values.
        """
        p = self.params[self.active_parameters].view('<f8').copy()
        return p

    def set_params(self, p):
        """
        Take a list of (active) parameters and set local variables.
        """
        for ip, pname in enumerate(self.active_parameters):
            self.params[pname][0] = p[ip]
        return None

    def validate_params(self):
        """
        Check that all model parameters take values inside allowed ranges.
        """
        valid_params = True
        ### Width must be positive
        if self.params[0].fwhm <=0.:
            valid_params *= False
        ### Ellipticity must be on [0, 1]
        if self.params[0].e < 0. or self.params[0].e > 0.9:
            valid_params *= False
        ### Position angle (in radians) must be on [0, pi]
        if self.params[0].beta < 0.0 or self.params[0].beta > np.pi:
            valid_params *= False
        return valid_params

    def get_psf(self):
        psf = galsim.Kolmogorov(fwhm=self.params[0]['fwhm'],
            gsparams=self.gsparams)
        psf_shape = galsim.Shear(g=self.params[0].e,
            beta=self.params[0].beta*galsim.radians)
        psf = psf.shear(psf_shape)
        return psf

    def get_psf_image(self, ngrid=None, pixel_scale_arcsec=0.2):
        psf = self.get_psf()
        if ngrid is None:
            ngrid = 16
        image_epsf = psf.drawImage(scale=pixel_scale_arcsec, nx=ngrid, ny=ngrid)
        return image_epsf

    def save_image(self, file_name, ngrid=None, pixel_scale_arcsec=0.2):
        image_epsf = self.get_psf_image(ngrid, pixel_scale_arcsec)
        image_epsf.write(file_name)
        return None


def make_test_image():
    psfm = PSFModel()
    filename = "../TestData/test_psf_image.fits"
    print "Saving PSF test image to {}".format(filename)
    psfm.save_image(filename)


if __name__ == "__main__":
    make_test_image()
