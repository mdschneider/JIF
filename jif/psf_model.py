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


k_galsim_psf_types = [('psf_fwhm', '<f8'), ('psf_e', '<f8'), ('psf_beta', '<f8')]
k_galsim_psf_defaults = [(0.6, 0.01, 0.4)]


class PSFModel(object):
    """
    Parametric PSF models for marginalization in galaxy model fitting.

    NOTE: Only models atmospheric + optics PSFs so far.
    An option to only model optics will be a future feature.

    @param active_parameters    List of model parameters for sampling
    @param gsparams             GalSim parameters object for rendering images
    @param lam_over_diam        Wavelength over the primary diameter in arcseconds
    @param obscuration          Fractional obscuration of the telescope entrance pupil
    """
    def __init__(self, active_parameters=['psf_fwhm'], gsparams=None,
                 lam_over_diam=0.012, obscuration=0.548):
        self.active_parameters = active_parameters
        self.gsparams = gsparams
        self.lam_over_diam = lam_over_diam
        self.obscuration = obscuration

        self.params = np.core.records.array(k_galsim_psf_defaults,
            dtype=k_galsim_psf_types)

    def get_params(self):
        """
        Return a list of active model parameter values.
        """
        if len(self.active_parameters) > 0:
            p = self.params[self.active_parameters].view('<f8').copy()
        else:
            p = []
        return p

    def set_param_by_name(self, paramname, value):
        """
        Set a single parameter value using the parameter name as a key.

        Can set 'active' or 'inactive' parameters. So, this routine gives a
        way to set fixed or fiducial values of model parameters that are not
        used in the MCMC sampling in Roaster.
        """
        self.params[paramname][0] = value
        return None

    def get_param_by_name(self, paramname):
        """
        Get a single parameter value using the parameter name as a key.

        Can access 'active' or 'inactive' parameters.
        """
        return self.params[paramname][0]

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
        if self.params[0].psf_fwhm <=0.:
            valid_params *= False
        ### Ellipticity must be on [0, 1]
        if self.params[0].psf_e < 0. or self.params[0].psf_e > 0.9:
            valid_params *= False
        ### Position angle (in radians) must be on [0, pi]
        if self.params[0].psf_beta < 0.0 or self.params[0].psf_beta > np.pi:
            valid_params *= False
        return valid_params

    def get_psf(self):
        optics = galsim.Airy(self.lam_over_diam, obscuration=self.obscuration,
            flux=1., gsparams=self.gsparams)
        atmos = galsim.Kolmogorov(fwhm=self.params[0]['psf_fwhm'],
            gsparams=self.gsparams)
        psf = galsim.Convolve([atmos, optics])
        psf_shape = galsim.Shear(g=self.params[0].psf_e,
            beta=self.params[0].psf_beta*galsim.radians)
        psf = psf.shear(psf_shape)
        return psf

    def get_psf_image(self, ngrid=None, pixel_scale_arcsec=0.2, out_image=None, gain=1.0):
        psf = self.get_psf()
        image_epsf = psf.drawImage(image=out_image, scale=pixel_scale_arcsec, nx=ngrid, ny=ngrid,
                                   gain=gain)
        return image_epsf

    def save_image(self, file_name, ngrid=None, pixel_scale_arcsec=0.2):
        image_epsf = self.get_psf_image(ngrid, pixel_scale_arcsec)
        image_epsf.write(file_name)
        return None


class FlatPriorPSF(object):
    """
    A flat prior for the parametric PSF model
    """
    def __init__(self):
        pass

    def __call__(self, Pi):
        return 0.0


class DefaultPriorPSF(object):
    def __init__(self):
        self.fwhm_mean = 0.6
        self.fwhm_var = 0.25

    def _lnprior_fwhm(self, fwhm):
        return -0.5 * (fwhm - self.fwhm_mean) ** 2 / self.fwhm_var

    def __call__(self, Pi):
        return self._lnprior_fwhm(Pi[0].psf_fwhm)


def make_test_image():
    psfm = PSFModel()
    filename = "../TestData/test_psf_image.fits"
    print "Saving PSF test image to {}".format(filename)
    psfm.save_image(filename)


if __name__ == "__main__":
    make_test_image()
