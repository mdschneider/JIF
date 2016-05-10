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
import parameters as jifparams
import telescopes


class PSFModel(object):
    """
    Parametric PSF models for marginalization in galaxy model fitting.

    NOTE: Only models infinite exposure Kolmogorov atmosphere + unaberrated optics PSFs so far.
    An option to only model optics will be a future feature.

    @param active_parameters    List of model parameters for sampling
    @param gsparams             GalSim parameters object for rendering images
    @param lam_over_diam        Wavelength over the primary diameter in arcseconds
    @param telescope            Telescope model ("LSST" or "WFIRST") [Default: "LSST"]
    @param achromatic           Simulate an achromatic PSF? [Default: True]
    @param SED_name             Name of an SED template in parameters.k_star_SED_names.
                                Not used if modeling achromatic objects.
    """
    ref_filter = 'r'
    def __init__(self, active_parameters=['psf_fwhm'], gsparams=None,
                 lam_over_diam=0.012, telescope="LSST", achromatic=True,
                 SED_name='NGC_0695_spec'):
        self.active_parameters = active_parameters
        self.gsparams = gsparams
        self.lam_over_diam = lam_over_diam
        self.telescope_name = telescope
        self.achromatic = achromatic
        self.SED_name = SED_name

        self.params = np.core.records.array(jifparams.k_galsim_psf_defaults,
            dtype=jifparams.k_galsim_psf_types)

        if not self.achromatic:
            self._load_sed_files()
            self._load_filter_files()

    def _load_sed_files(self):
        """
        Load star SED templates from files.

        Copied from GalSim demo12.py
        """
        path, filename = os.path.split(__file__)
        datapath = os.path.abspath(os.path.join(path, "../input/"))
        self.SEDs = {}
        for SED_name in jifparams.k_star_SED_names:
            SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
            self.SEDs[SED_name] = galsim.SED(SED_filename, wave_type='Ang')

    def _load_filter_files(self):
        self.filters = telescopes.load_filter_files(wavelength_scale=1.0,
                                                    telescope_name=self.telescope_name)
        ### Add the reference filter for defining the magnitude parameters
        path, filename = os.path.split(__file__)
        datapath = os.path.abspath(os.path.join(path, "../input/"))
        ref_filename = os.path.join(datapath, '{}_{}.dat'.format('LSST',
            PSFModel.ref_filter))
        self.filters['ref'] = telescopes.load_filter_file_to_bandpass(ref_filename)        

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
        ### Width must be greater than a small value (in arcseconds)
        ### Note that very small PSF widths will require large GalSim FFTs - so bound it here.
        if self.params[0].psf_fwhm <=0.1 or self.params[0].psf_fwhm > 2.:
            valid_params *= False
        ### Ellipticity must be on [0, 1]. But highly elliptical PSFs probably indicated artifacts
        ### or other failure modes of the fit, so bound to be something less than 1.
        if self.params[0].psf_e < 0. or self.params[0].psf_e > 0.6:
            valid_params *= False
        ### Position angle (in radians) must be on [0, pi]
        if self.params[0].psf_beta < 0.0 or self.params[0].psf_beta > np.pi:
            valid_params *= False
        return valid_params

    def set_mag_from_obs(self, appr_mag, filter_name='r'):
        """
        Set the magnitude model parameter given an apparent magnitude of a star in the specified
        filter.

        @param appr_mag     Apparent magnitude to use in setting the model magnitude parameter
        @param filter_name  Name of the filter to use to calculate magnitudes. (Default: 'r')
        """
        if appr_mag < 98.:
            bp = self.filters[filter_name]
            bp_ref = self.filters['ref']
            SED = self.SEDs[self.SED_name]
            SED = SED.atRedshift(0.).withMagnitude(target_magnitude=appr_mag, bandpass=bp)
            mag_model = SED.atRedshift(0.).calculateMagnitude(bp_ref)
            self.params['psf_mag'][0] = mag_model
        else:
            self.params['psf_mag'][0] = 99.

    def get_SED(self):
        """
        Get the GalSim SED object with amplitude set to the model parameter flux.

        The SED is selected from a library of templates according to the class member variable
        'SED_name'. See the 'parameters' module for the ordered list of SED templates.
        """
        if self.achromatic:
            print "This is an achromatic PSF model - no SED defined"
            return None
        else:
            bp = self.filters['ref']
            SED = self.SEDs[self.SED_name].atRedshift(0.).withMagnitude(self.params[0].psf_mag,
                bandpass=bp)
            return SED

    def get_flux(self, filter_name='r'):
        """
        Get the flux of the star model in the named bandpass

        @param filter_name  Name of the bandpass for the desired magnitude

        @returns the flux in the requested bandpass (in photon counts)
        """
        if self.achromatic:
            return jifparams.flux_from_AB_mag(self.params[0].psf_mag, 
                exposure_time_s=telescopes.k_telescopes[self.telescope_name]["exptime_zeropoint"],
                gain=telescopes.k_telescopes[self.telescope_name]["gain"])
        else:
            SED = self.get_SED()
            flux = SED.calculateFlux(self.filters[filter_name])
        return flux

    def get_magnitude(self, filter_name='r'):
        """
        Get the magnitude of the star model in the named bandpass

        @param filter_name  Name of the bandpass for the desired magnitude

        @returns the magnitude in the requested bandpass
        """
        if self.achromatic:
            return self.params[0].psf_mag
        else:
            SED = self.get_SED()
            mag = SED.calculateMagnitude(self.filters[filter_name])
        return mag

    def get_psf(self):
        """
        Get the GalSim PSF model instance

        Includes unaberrated optics and infinite exposure atmosphere components.

        See GalSim demo12 for the chromatic PSF modeling upon which this routine is based.
        """
        atmos_mono = galsim.Kolmogorov(fwhm=self.params[0]['psf_fwhm'],
                                       gsparams=self.gsparams)
        if self.achromatic:
            atmos = atmos_mono
            optics = galsim.Airy(self.lam_over_diam,
                                 obscuration=telescopes.k_telescopes[self.telescope_name]["obscuration"],
                                 flux=1., gsparams=self.gsparams)
        else:
            ### Point at the zenith to minimize DCR effects in this simplified model. 
            ### But, the ChromaticAtmosphere class should still give us the wavelength dependent 
            ### seeing. 
            atmos = galsim.ChromaticAtmosphere(atmos_mono, 500.,
                                               zenith_angle=0.6 * galsim.radians,
                                               parallactic_angle=2.7 * galsim.radians)
            d = telescopes.k_telescopes[self.telescope_name]["primary_diam_meters"]
            optics = galsim.ChromaticAiry(lam=self.lam_over_diam * d, diam=d)


        if telescopes.k_telescopes[self.telescope_name]["atmosphere"]:
            psf_shape = galsim.Shear(g=self.params[0].psf_e,
                                     beta=self.params[0].psf_beta*galsim.radians)
            atmos = atmos.shear(psf_shape)
            psf = galsim.Convolve([atmos, optics])
        else:
            psf = optics

        return psf

    def get_psf_image(self, filter_name='r', ngrid=None, pixel_scale_arcsec=0.2, out_image=None,
                      gain=1.0):
        """
        Get a GalSim Image instance of the PSF

        This renders images with fluxes set according to the 'psf_mag' model parameter to enable
        simulation of star images. 

        (So this method could be better described as 'get_star_image' perhaps.)
        """
        if ngrid is None and out_image is None:
            raise ValueError("Must specify either ngrid or out_image")
        psf = self.get_psf()
        if self.achromatic:
            psf = psf.withFlux(jifparams.flux_from_AB_mag(self.params[0].psf_mag))
            image_epsf = psf.drawImage(image=out_image, scale=pixel_scale_arcsec,
                                       nx=ngrid, ny=ngrid, gain=gain)
        else:
            ### For chromatic modeling, the PSF 'mag' parameter only has meaning when the PSF model
            ### is convolved with a source model. So we convolve with a star model to render a 
            ### PSF image with a given magnitude.
            SED = self.get_SED()
            mono_star = galsim.Gaussian(fwhm=0.001)
            star = galsim.Chromatic(mono_star, SED)
            final = galsim.Convolve([star, psf])

            image_epsf = final.drawImage(bandpass=self.filters[filter_name], image=out_image,
                                         scale=pixel_scale_arcsec, gain=gain, nx=ngrid, ny=ngrid)
        return image_epsf

    def save_image(self, file_name, ngrid=None, pixel_scale_arcsec=0.2):
        """
        Save the PSF/star image to FITS file using the GalSim 'write' method
        """
        image_epsf = self.get_psf_image(ngrid=ngrid, pixel_scale_arcsec=pixel_scale_arcsec)
        image_epsf.write(file_name)
        return None


class FlatPriorPSF(object):
    """
    A flat prior for the parametric PSF model
    """
    def __init__(self):
        pass

    def __call__(self, Pi, *args, **kwargs):
        return 0.0


class DefaultPriorPSF(object):
    def __init__(self):
        self.fwhm_mean = 0.6
        self.fwhm_var = 0.25

    def _lnprior_fwhm(self, fwhm):
        return -0.5 * (fwhm - self.fwhm_mean) ** 2 / self.fwhm_var

    def __call__(self, Pi, *args, **kwargs):
        return self._lnprior_fwhm(Pi[0].psf_fwhm)


def make_test_image():
    psfm = PSFModel(achromatic=True)
    filename = "../data/TestData/test_psf_image.fits"
    print "Saving PSF test image to {}".format(filename)
    psfm.save_image(filename, ngrid=32)


if __name__ == "__main__":
    make_test_image()
