#!/usr/bin/env python
# encoding: utf-8
"""
galsim_galaxy.py

Wrapper for GalSim galaxy models to use in MCMC.
"""
import os
import math
import copy
import numpy as np
from operator import add
import warnings
###
import galsim
import galsim.wfirst
###
# import segments
import footprints
import parameters as jifparams
import telescopes
import psf_model as pm


class GalSimGalaxyModel(object):
    """
    Parametric galaxy model from GalSim for MCMC.

    This class has methods to store and parse model parameter lists for use in MCMC sampling. The
    working model is that an MCMC chain uses a flat list or array of (unnamed) parameters and this
    class provides a way to transform this flat list into a set of named parameters that are
    generally a subset of all the parameters needed to render a galaxy image.

    As such, this class also holds the methods to render galaxy images (with or without noise) 
    with GalSim.

    Of course, a PSF is also needed to render an image. So this class stores a PSF model,
    which can be a GalSim InterpolatedImage set from the input 'Footprints' file (via the 'Load'
    method) or as a parametric model with parameters that can be sampled just as the galaxy model
    parameters (via the JIF PSFModel class). If PSF sampling is used, then the PSF model parameters
    are appended to the flat array of galaxy model parameters wherever apprpriate.

    **There is only one PSF model (i.e., one epoch or exposure modeled) for any GalsimGalaxyModel
    instance**

    To model multiple epochs, even of the same galaxy model, the user should instantiate a new
    GalsimGalaxyModel instance for each epoch and ensure that the galaxy model parameters are
    appropriately copied for each instance - see how this is done in JIF Roaster.py.

    Derived originally from GalSim examples/demo1.py

    @param galaxy_model         The name of a parametric galaxy model to simulate. Valid values are
                                ['star', 'Spergel', 'Sersic', 'BulgeDisk'].
    @param telescope_model      The name of a telescope model to set the pixel scale, noise model, 
                                etc. Valid models are defined in `parameters.py` and include 
                                ['LSST', 'WFIRST', 'SDSS'].
    @param psf_model            The PSF modeling choice. Can be: 
                                    1) a GalSim image instance with the PSF image  
                                    2) a JIF PSFModel class instance
                                    3) a string or other value -> revert to option 2)
                                If option 3) is selected, then use the PSFModel class to model the 
                                PSF. The components of the PSF model thus used 
                                (e.g., optics, atmosphere) are determined by the value of 
                                `telescope_model`. If option 2), then the supplied PSFModel instance
                                must be compatible with the supplied telescope_model choice.
                                A numpy.ndarray will raise a ValueError. Commonly we make a mistake 
                                of passing a numpy array instead of a galsim.Image instance when 
                                we want to make an InterpolatedImage PSF.
    @param active_parameters    A list of model parameter names for the galaxy model, and 
                                optionally, the PSF model. See the parameters.py file for valid 
                                parameter names to supply.
    """
    def __init__(self,
                 galaxy_model="Spergel",
                 telescope_model="LSST",
                 psf_model='Model',
                 active_parameters=['e1', 'e2'],
                 filter_name='r'):
        assert galaxy_model in ['star', 'Spergel', 'Sersic', 'BulgeDisk']
        assert telescope_model in telescopes.k_telescopes.keys()
        self.galaxy_model = galaxy_model
        self.telescope_model = telescope_model
        self.psf_model = psf_model
        self.active_parameters = active_parameters
        self.filter_name = filter_name

        self._init_gs_params()
        self._init_telescope()
        self._init_model_params()
        self._init_psf()
        return None

    def _init_gs_params(self):
        self.gsparams = jifparams.k_default_gsparams
        return None

    def _init_telescope(self):
        tel = telescopes.k_telescopes[self.telescope_model]
        self.pixel_scale_arcsec = tel['pixel_scale']
        self.gain = tel['gain']
        return None

    def _init_model_params(self):
        ### Set GalSim galaxy model parameters
        self.paramtypes = jifparams.k_galparams_types[self.galaxy_model]
        self.params = np.core.records.array(jifparams.k_galparams_defaults[self.galaxy_model],
            dtype=self.paramtypes)
        self.paramnames = self.active_parameters
        ## Separate the galaxy model parameters from the galaxy+PSF parameter names
        self.active_parameters_galaxy = jifparams.select_galaxy_paramnames(self.active_parameters)
        self.n_params = len(self.active_parameters)
        self.psf_paramnames = jifparams.select_psf_paramnames(self.active_parameters)
        self.n_psf_params = len(self.psf_paramnames)
        return None

    def _init_psf(self):
        if isinstance(self.psf_model, np.ndarray):
            raise ValueError("PSF model was supplied as a numpy array. Did you want a galsim.Image?")
        if isinstance(self.psf_model, galsim.Image):
            if self.galaxy_model == 'star' or (self.n_psf_params > 0):
                raise AttributeError('InterpolatedImage PSF incompatible with star/PSF model')
            self.psf_model_type = 'InterpolatedImage'                
            self.psf_model = galsim.InterpolatedImage(self.psf_model, 
                scale=self.pixel_scale_arcsec, gsparams=self.gsparams, flux=1.0)
        else:
            self.psf_model_type = 'PSFModel class'
            if not isinstance(self.psf_model, pm.PSFModel):
                lam_over_diam = telescopes.tel_lam_over_diam(self.telescope_model, self.filter_name)
                self.psf_model = pm.PSFModel(active_parameters=self.psf_paramnames,
                                             gsparams=self.gsparams,
                                             lam_over_diam=lam_over_diam,
                                             telescope=self.telescope_model,
                                             achromatic=True)
            else:
                ## Check consistency between supplied PSF model and telescope model
                assert self.telescope_model == self.psf_model.telescope_name
                assert self.psf_paramnames == self.psf_model.active_parameters
        return None

    def set_param_by_name(self, paramname, value):
        """
        Set a single parameter value using the parameter name as a key.

        Can set 'active' or 'inactive' parameters. So, this routine gives a
        way to set fixed or fiducial values of model parameters that are not
        used in the MCMC sampling in Roaster.

        @param paramname    The name of the galaxy or PSF model parameter to set
        @param value        The value to assign to the model parameter
        """
        if 'psf' in paramname: 
            if self.psf_model_type == "PSFModel class":
                self.psf_model.params[paramname][0] = value
            else:
                raise ValueError("Cannot set PSF parameter when PSF is an InterpolatedImage")
        else:
            self.params[paramname][0] = value
            # if 'beta' in paramname:
            #     self.params[paramname][0] = np.mod(value, np.pi)
        return None

    def get_param_by_name(self, paramname):
        """
        Get a single parameter value using the parameter name as a key.

        Can access 'active' or 'inactive' parameters.
        """
        if 'psf' in paramname:
            if self.psf_model_type == "PSFModel class":
                p = self.psf_model.params[paramname][0]
            else:
                raise ValueError("Cannot get PSF parameter when PSF is an InterpolatedImage")
        else:
            p = self.params[paramname][0]
        return p

    def set_params(self, p):
        """
        Take a list of (active) parameters and set local variables.

        We assume p is a list or flat numpy array with values listed in the
        same order as the parameter names in self.active_parameters (which
        is supplied on instantiation of a `GalSimGalaxyModel` object).

        If the PSF model for this instance is a `PSFModel` object, then the
        active parameters of the PSFModel should be appended to the list input
        here.

        For use in emcee.

        @param p    A list or array of galaxy (and PSF) model parameter values
        """
        assert len(p) >= self.n_params
        for ip, pname in enumerate(self.active_parameters_galaxy):
            self.params[pname][0] = p[ip]
            # if 'mag_sed' in pname:
            #     ### Transform flux variables with exp -- we sample in ln(Flux)
            #     self.params[pname][0] = p[ip]
            # else:
            #     self.params[pname][0] = p[ip]
            # if 'beta' in pname:
            #     self.params[pname][0] = np.mod(p[ip], np.pi)
        if self.psf_model_type == "PSFModel class":
            ### Assumes the PSF parameters are appended to the galaxy parameters
            self.psf_model.set_params(p[len(self.active_parameters_galaxy):])
        return None

    def get_params(self):
        """
        Return a list of active model parameter values.

        @returns a flat array of model parameter values in the order specified
                 in the `active_parameters` argument to the `GalSimGalaxyModel`
                 constructor
        """
        p = self.params[self.active_parameters_galaxy].view('<f8').copy()
        if self.psf_model_type == "PSFModel class":
            psf_active_params = self.psf_model.get_params()
            if len(psf_active_params) > 0:
                p = np.append(p, self.psf_model.get_params())
        # ### Transform parameters for sampling
        # for ip, pname in enumerate(self.active_parameters):
        #     if 'beta' in pname:
        #         p[ip] = np.mod(p[ip], np.pi)
        return p

    def get_psf_params(self):
        """
        Return a list of active model parameter values if using a parametric PSF model

        If the PSF is modeled as a static image then return an empty list.
        """
        if self.psf_model_type == "PSFModel class":
            return self.psf_model.get_params()
        else:
            return []

    def validate_params(self):
        """
        Check that all model parameters take values inside allowed ranges.

        @returns a boolean indicating the validity of the current model
                 parameters
        """
        def inbounds(p, b):
            return (p >= b[0] and p <= b[1])

        valid_params = 1
        for pname, dtype in self.params.dtype.descr:
            if not inbounds(self.params[pname][0], jifparams.k_param_bounds[pname]):
                valid_params *= 0
        if self.psf_model_type == "PSFModel class":
            valid_params *= self.psf_model.validate_params()

        # if self.params['beta'][0] < 0.:
        #     print "neg beta, valid: ", bool(valid_params)
        return bool(valid_params)

    def get_psf(self):
        """
        Get the PSF as a `GSObject` for use in GalSim image rendering or
        convolutions

        The type of PSF model is determined by the `psf_model` argument to the
        class constructor. The PSF object returned here could be:
            1. a GalSim `InterpolatedImage`
            2. a JIF `PSFModel`
        @returns the PSF model instance
        """
        if self.psf_model_type == 'InterpolatedImage':
            psf = self.psf_model
        elif self.psf_model_type == 'PSFModel class':
            psf = self.psf_model.get_psf()
        else:
            raise AttributeError("Invalid PSF model type name")
        return psf

    def get_image(self, out_image=None):
        if self.galaxy_model == "star":
            return self.get_psf_image(filter_name=self.filter_name, 
                out_image=out_image, gain=self.gain)

        elif self.galaxy_model in ["Spergel", "Sersic"]:
            if self.galaxy_model == "Spergel":
                gal = galsim.Spergel(nu=self.params[0].nu,
                    half_light_radius=self.params[0].hlr,
                    flux=1.0, gsparams=self.gsparams)
            else:
                gal = galsim.Sersic(n=self.params[0].n,
                    half_light_radius=self.params[0].hlr,
                    flux=1.0, gsparams=self.gsparams)
            # flux = jifparams.flux_from_AB_mag(self.params[0].mag_sed1)
            # gal = gal.withFlux(flux)
            gal_shape = galsim.Shear(e1=self.params[0].e1, e2=self.params[0].e2)
            gal = gal.shear(gal_shape)
            gal = gal.shift(self.params[0].dx, self.params[0].dy)
            # print "GG gal HLR: {:12.10g}".format(gal.calculateHLR())

        elif self.galaxy_model == "BulgeDisk":
            bulge = galsim.Spergel(nu=self.params[0].nu_bulge,
                half_light_radius=self.params[0].hlr_bulge,
                flux=1.0,
                gsparams=self.gsparams)
            flux = jifparams.flux_from_AB_mag(self.params[0].mag_sed1_bulge)
            bulge = bulge.withFlux(flux)
            e_disk = np.hypot(self.params[0].e1_bulge, self.params[0].e2_bulge)
            beta_disk = 0.5 * np.arctan2(self.params[0].e2_bulge, 
                                         self.params[0].e1_bulge)            
            bulge_shape = galsim.Shear(g=e_bulge, beta=beta_bulge*galsim.radians)
            bulge = bulge.shear(bulge_shape)
            bulge = bulge.shift(self.params[0].dx_bulge, self.params[0].dy_bulge)

            disk = galsim.Spergel(nu=self.params[0].nu_disk,
                half_light_radius=self.params[0].hlr_disk,
                flux=1.0,
                gsparams=self.gsparams)
            flux = jifparams.flux_from_AB_mag(self.params[0].mag_sed1_disk)
            disk = disk.withFlux(flux)
            e_disk = np.hypot(self.params[0].e1_disk, self.params[0].e2_disk)
            beta_disk = 0.5 * np.arctan2(self.params[0].e2_disk, 
                                         self.params[0].e1_disk)
            disk_shape = galsim.Shear(g=e_disk, beta=beta_disk*galsim.radians)
            disk = disk.shear(disk_shape)
            disk = disk.shift(self.params[0].dx_disk, self.params[0].dy_disk)

            # gal = self.params[0].bulge_frac * bulge + (1 - self.params[0].bulge_frac) * disk
            gal = bulge + disk

        else:
            raise AttributeError("Unimplemented galaxy model")
        # print "GG gal flux: {:12.10g}".format(gal.getFlux())
        # print "GG PSF HLR: {:12.10g}".format(self.get_psf(filter_name).calculateHLR())
        final = galsim.Convolve([gal, self.get_psf()])
        # print "GG final flux: {:12.10g}".format(final.getFlux())
        # print "GG hlr: {:12.10g}".format(final.calculateHLR())
        # wcs = galsim.PixelScale(self.pixel_scale)'
        
        # print "GG pixel scale: {:8.6f}".format(self.pixel_scale)
        # print "GG gain: {:8.6f}".format(gain)

        try:
            image = final.drawImage(image=out_image, scale=self.pixel_scale_arcsec,
                gain=self.gain, add_to_image=False) #, method='fft')
        except RuntimeError:
            print "Trying to make an image that's too big."
            print self.get_params()
            image = None
        return image

    def get_psf_image(self, ngrid=None, out_image=None):
        psf = self.get_psf()
        if self.psf_model_type == 'InterpolatedImage':
            image = psf
        elif self.psf_model_type == 'PSFModel class':
            image = self.psf_model.get_psf_image(out_image=out_image,
                                                 ngrid=ngrid, 
                                                 pixel_scale_arcsec=self.pixel_scale_arcsec,
                                                 gain=self.gain, 
                                                 normalize_flux=True)
        else:
            raise AttributeError("Invalid PSF model type")
        return image

    def save_footprint(self, outfile, out_image=None, noise=None):
        """
        Write image and metadata to the JIF HDF5 'Footprints' file format
        """
        seg = footprints.Footprints(outfile)
        seg_ndx = 0
        
        src_catalog = self.params
        seg.save_source_catalog(src_catalog, segment_index=seg_ndx)

        dummy_mask = 1.0
        dummy_background = 0.0

        im = self.get_image(out_image=out_image)

        noise_var = 1.e-9
        if noise is not None:
            if isinstance(noise, galsim.BaseNoise):
                im.addNoise(noise)
                pix_dat = im.array
                noise_var = noise.getVariance()
            elif isinstance(noise, np.ndarray):
                assert noise.shape[0] == im.array.shape[0]
                assert noise.shape[1] == im.array.shape[1]
                pix_dat = im.array + noise
                noise_var = float(np.var(noise.ravel()))
            else:
                raise TypeError("Invalid noise input: must be galsim Noise instance or ndarray")

        seg.save_images([pix_dat], [noise_var], [dummy_mask], [dummy_background],
            segment_index=seg_ndx, telescope=self.telescope_model,
            filter_name=self.filter_name)
        tel = telescopes.k_telescopes[self.telescope_model]
        seg.save_tel_metadata(telescope=self.telescope_model.lower(),
            primary_diam=tel['primary_diam_meters'],
            pixel_scale_arcsec=tel['pixel_scale'],
            atmosphere=tel['atmosphere'])
        seg.save_psf_images([self.get_psf_image(ngrid=64).array],
            segment_index=seg_ndx,
            telescope=self.telescope_model.lower(),
            filter_name=self.filter_name)
        return None

    def save_image(self, file_name, out_image=None):
        image = self.get_image(out_image=out_image)
        image.write(file_name)
        return None

    def save_psf(self, file_name, ngrid=None):
        image_epsf = self.get_psf_image(ngrid)
        image_epsf.write(file_name)
        return None

    def plot_image(self, file_name, ngrid=None, title=None, noise=None):
        import matplotlib.pyplot as plt
        if ngrid is not None:
            out_image = galsim.Image(ngrid, ngrid)
        else:
            out_image = None

        im = self.get_image(out_image=out_image)
        if noise is not None:
            im.addNoise(noise)
        print "Image rms: ", np.sqrt(np.var(im.array.ravel()))
        ###
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(im.array,
            cmap=plt.get_cmap('pink'), origin='lower',
            interpolation='none',
            extent=[0, ngrid*self.pixel_scale_arcsec, 0, ngrid*self.pixel_scale_arcsec])
        ax.set_xlabel(r"Detector $x$-axis (arcsec.)")
        ax.set_ylabel(r"Detector $y$-axis (arcsec.)")
        if title is not None:
            ax.set_title(title)
        cbar = fig.colorbar(im)
        cbar.set_label(r"photons / pixel")
        fig.savefig(file_name)
        return None

    def plot_psf(self, file_name, ngrid=None, title=None, noise=None):
        import matplotlib.pyplot as plt
        psf = self.get_psf()
        if ngrid is None:
            ngrid = 32
        # image_epsf = psf.drawImage(image=None,
        #     scale=self.pixel_scale, nx=ngrid, ny=ngrid)
        image_epsf = self.get_psf_image(ngrid=ngrid)
        if noise is not None:
            image_epsf.addNoise(noise)
        ###
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(1,1,1)
        ngx, ngy = image_epsf.array.shape
        im = ax.imshow(image_epsf.array,
            cmap=plt.get_cmap('pink'), origin='lower',
            interpolation='none',
            extent=[0, ngx*self.pixel_scale_arcsec, 0, ngy*self.pixel_scale_arcsec])
        ax.set_xlabel(r"Detector $x$-axis (arcsec.)")
        ax.set_ylabel(r"Detector $y$-axis (arcsec.)")
        if title is not None:
            ax.set_title(title)
        cbar = fig.colorbar(im)
        cbar.set_label(r"normalized photons / pixel")
        fig.savefig(file_name)
        return None

    def get_moments(self, add_noise=True):
        results = self.get_image().FindAdaptiveMom()
        print 'HSM reports that the image has observed shape and size:'
        print '    e1 = %.3f, e2 = %.3f, sigma = %.3f (pixels)' % (results.observed_shape.e1,
                    results.observed_shape.e2, results.moments_sigma)


class GalSimGalaxyModelChromatic(GalSimGalaxyModel):
    """
    Parametric galaxy model from GalSim for MCMC.

    This class has methods to store and parse model parameter lists for use in MCMC sampling. The
    working model is that an MCMC chain uses a flat list or array of (unnamed) parameters and this
    class provides a way to transform this flat list into a set of named parameters that are
    generally a subset of all the parameters needed to render a galaxy image.

    As such, this class also holds the methods to render galaxy images (with or without noise),
    in a specified passband with a given SED model, or as a GalSim monochromatic object.

    Of course, a PSF is also needed to render an image. So this class also stores a PSF model,
    which can be a GalSim InterpolatedImage set from the input 'segment' file (via the 'Load'
    method) or as a parametric model with parameters that can be sampled just as the galaxy model
    parameters (via the JIF PSFModel class). If PSF sampling is used, then the PSF model parameters
    are appended to the flat array of galaxy model parameters wherever apprpriate.

    **There is only one PSF model (i.e., one epoch or exposure modeled) for any GalsimGalaxyModel
    instance**

    To model multiple epochs, even of the same galaxy model, the user should instantiate a new
    GalsimGalaxyModel instance for each epoch and ensure that the galaxy model parameters are
    appropriately copied for each instance - see how this is done in JIF Roaster.py.

    Derived originally from GalSim examples/demo1.py

    @param telescope_name       Name of the telescope to model. Used to identify
                                filter curves. [Default: "LSST"]
    @param pixel_scale_arcsec   Pixel scale for image models [Default: 0.11]
    @param noise                GalSim noise model. [Default: None]
    @param galaxy_model         Name of the parametric galaxy model
                                Valid values are 'Sersic', 'Spergel', 'BulgeDisk', or 'star'
                                [Default: "Spergel"]
    @param active_parameters    List of the parameter names for sampling
    @param wavelength_meters    Wavelength in meters to set the scale for the
                                optics PSF [Default: 620e-9]
    @param primary_diam_meters  Diameter of the telescope primary [Default: 2.4]
    @param filters              List of galsim.Bandpass instances. This argument, if not 'None',
                                takes precedence over 'filter_names' and defines which filters can
                                be used with this galaxy model instance.
    @param filter_names         List of filter names to be used if the 'filters' parameter is not
                                specified. If supplied, the names in this list must match those
                                in the 'input/' directory with tables of bandpasses.
                                If neither 'filters' or 'filter_names' ares supplied, a Warning
                                is raised and subsequent execution may produce unexpected results.
    @param filter_wavelength_scale Multiplicative scaling to apply to input filter wavelenghts
    @param atmosphere           Simulate an (infinite exposure) atmosphere PSF? [Default: False]
    @param psf_model            Specification for the PSF model. Can be:
                                    1. a GalSim InterpolatedImage instance
                                    2. a PSFModel instance
                                    3. a name of a parametric model
                                [Default: parametric model]
    @param achromatic_galaxy    If True, don't use the GalSim Chromatic features. Instead, model
                                galaxies with a flux set by the 'mag_sed1' model parameter (with
                                appropriate transformation to flux for AB mags). For multiple
                                bands observed, this effectively assumes a flat SED model if all
                                passbands were of equivalent shapes.
    """
    ### Define a reference filter with respect to which magnitude parameters are defined
    ref_filter = 'r'
    def __init__(self,
                 telescope_model="LSST",
                 pixel_scale_arcsec=0.11, ### arcseconds
                 noise=None,
                 galaxy_model="Spergel",
                 active_parameters=['hlr'], #, 'e', 'beta'],
                 primary_diam_meters=2.4,
                 filters=None,
                 filter_names=None,
                 filter_wavelength_scale=1.0,
                 atmosphere=False,
                 psf_model=None,
                 achromatic_galaxy=False):
        super(GalSimGalaxyModelChromatic, self).__init__(galaxy_model=galaxy_model,
            telescope_model=telescope_model, psf_model=psf_model,
            active_parameters=active_parameters)
        self.pixel_scale = pixel_scale_arcsec
        self.noise = noise
        self.primary_diam_meters = primary_diam_meters
        self.filters = copy.deepcopy(filters)
        self.filter_names = filter_names
        self.atmosphere = atmosphere

        self.achromatic_galaxy = achromatic_galaxy

        if self.psf_model in ['star', 'Model', 'InterpolatedImage']:
            super(GalSimGalaxyModelChromatic, self)._init_psf()
        else:
            self.psf_model_type = 'Parametric'

        if not self.achromatic_galaxy:
            ### Set GalSim SED model parameters
            self._load_sed_files()
        ### Load the filters that can be used to draw galaxy images
        if self.filters is None:
            if self.filter_names is not None:
                self._load_filter_files(filter_wavelength_scale)
            else:
                warnings.warn("No filters available in GalSimGalaxyModel: supply"
                              + "'filters' or 'filter_names' argument")
        else:
            self.filter_names = self.filters.keys()
        ### Add the reference filter for defining the magnitude parameters
        path, filename = os.path.split(__file__)
        datapath = os.path.abspath(os.path.join(path, "input/"))
        ref_filename = os.path.join(datapath, '{}_{}.dat'.format('LSST',
            GalSimGalaxyModel.ref_filter))
        self.filters['ref'] = telescopes.load_filter_file_to_bandpass(ref_filename)
        return None


    def _load_sed_files(self):
        """
        Load SED templates from files.

        Copied from GalSim demo12.py
        """
        path, filename = os.path.split(__file__)
        datapath = os.path.abspath(os.path.join(path, "input/"))
        self.SEDs = {}
        for SED_name in jifparams.k_SED_names:
            SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
            self.SEDs[SED_name] = galsim.SED(SED_filename, wave_type='Ang',
                                             flux_type='flambda')
        return None

    def _load_filter_files(self, wavelength_scale=1.0):
        """
        Load filters for drawing chromatic objects.

        @param wavelength_scale     Multiplicative scaling of the wavelengths
                                    input from the filter files to get
                                    nanometers from whatever the input units are
        """
        self.filters = telescopes.load_filter_files(wavelength_scale, self.telescope_name)

    def get_psf(self, filter_name='r'):
        """
        Get the PSF as a `GSObject` for use in GalSim image rendering or
        convolutions

        The type of PSF model is determined by the `psf_model` argument to the
        class constructor. The PSF object returned here could be:
            1. a GalSim `InterpolatedImage`
            2. a JIF `PSFModel`
            3. a GalSim model PSF composed of optics and, optionally,
               atmosphere components

        @returns the PSF model instance
        """
        if self.psf_model_type == 'InterpolatedImage':
            psf = self.psf_model
        elif self.psf_model_type == 'PSFModel class':
            psf = self.psf_model.get_psf()
        else:
            lam_over_diam = self.filters[filter_name].effective_wavelength*1.e-9 / self.primary_diam_meters
            lam_over_diam *= 206264.8 # arcsec
            optics = galsim.Airy(lam_over_diam, 
                                 obscuration=telescopes.k_telescopes[self.telescope_name]['obscuration'], 
                                 flux=1.,
                                 gsparams=self.gsparams)
            if self.atmosphere:
                atmos = galsim.Kolmogorov(fwhm=0.6, gsparams=self.gsparams)
                psf = galsim.Convolve([atmos, optics])
            else:
                psf = optics
        return psf

    def set_mag_from_obs(self, sed_index, appr_mag, redshift=0.0, filter_name='r', gal_comp=''):
        """
        Set the magnitude model parameter given an apparent magnitude at a specified redshift in
        a specified filter.

        @param sed_index    Index into the SED template name list
        @param appr_mag     Apparent magnitude to use in setting the model magnitude parameter
        @param redshift     Redshift at which the input apparent magnitude is defined.
        @param filter_name  Name of the filter to use to calculate magnitudes. (Default: 'r')
        @param gal_comp     Name of the galaxy component (bulge,disk) to select. Can be the empty
                            string to get the composite galaxy model SED.
        """
        if self.galaxy_model == "star":
            self.psf_model.set_mag_from_obs(appr_mag, filter_name=filter_name)
        else:
            if appr_mag < 98.:
                bp = self.filters[filter_name]
                bp_ref = self.filters['ref']
                SED = self.SEDs[jifparams.k_SED_names[sed_index]]
                SED = SED.atRedshift(redshift).withMagnitude(target_magnitude=appr_mag, bandpass=bp)
                mag_model = SED.atRedshift(0.).calculateMagnitude(bp_ref)
                self.params['mag_sed{:d}'.format(sed_index+1)][0] = mag_model
            else:
                self.params['mag_sed{:d}'.format(sed_index+1)][0] = 99.
        return None

    def get_SED(self, gal_comp=''):
        """
        Get the GalSim SED object given the SED parameters and redshift.

        This routine passes galsim_galaxy magnitude parameters to the GalSim
        SED.withMagnitude() method.

        The magnitude GalSimGalaxyModel magnitude parameters are defined for redshift zero. If a
        model is requested for a different redshift, then the SED amplitude is set before the
        redshift, resulting in output apparent magnitudes that may not match the input apparent
        magnitude parameter (unless z=0).

        @param gal_comp             Name of the galaxy component (bulge,disk) to
                                    select. Can be the empty string to get the
                                    composite galaxy model SED.
        """
        if len(gal_comp) > 0:
            gal_comp = '_' + gal_comp
        bp = self.filters['ref']
        SEDs = [self.SEDs[SED_name].atRedshift(0.).withMagnitude(
            target_magnitude=self.params[0]['mag_sed{:d}{}'.format(i+1, gal_comp)],
            bandpass=bp).atRedshift(self.params[0].redshift)
                for i, SED_name in enumerate(jifparams.k_SED_names)]
        return reduce(add, SEDs)

    def get_flux(self, filter_name='r'):
        """
        Get the flux of the galaxy model in the named bandpass

        @param filter_name  Name of the bandpass for the desired magnitude

        @returns the flux in the requested bandpass (in photon counts)
        """
        if self.galaxy_model == "star":
            return self.psf_model.get_flux(filter_name)
        else:
            if self.achromatic_galaxy:
                raise NotImplementedError()
            else:
                SED = self.get_SED()
                flux = SED.calculateFlux(self.filters[filter_name])
            return flux

    def get_magnitude(self, filter_name='r'):
        """
        Get the magnitude of the galaxy model in the named bandpass

        @param filter_name  Name of the bandpass for the desired magnitude

        @returns the magnitude in the requested bandpass
        """
        if self.galaxy_model == "star":
            return self.psf_model.get_magnitude(filter_name)
        else:
            if self.achromatic_galaxy:
                raise NotImplementedError()
            else:
                SED = self.get_SED()
                mag = SED.calculateMagnitude(self.filters[filter_name])
            return mag

    def get_image(self, out_image=None, add_noise=False,
                  filter_name='r', gain=1.0, snr=None):
        if self.galaxy_model == "star":
            return self.get_psf_image(filter_name=filter_name,out_image=out_image, gain=gain)
        elif self.galaxy_model == "Gaussian":
            # gal = galsim.Gaussian(flux=self.params.gal_flux, sigma=self.params.gal_sigma)
            # gal_shape = galsim.Shear(g=self.params.e, beta=self.params.beta*galsim.radians)
            # gal = gal.shear(gal_shape)
            raise AttributeError("Unimplemented galaxy model")

        elif self.galaxy_model == "Spergel":
            # print "GG hlr param: {:12.10f}".format(self.params[0].hlr)
            mono_gal = galsim.Spergel(nu=self.params[0].nu,
                half_light_radius=self.params[0].hlr,
                # flux=self.params[0].gal_flux,
                flux=1.0,
                gsparams=self.gsparams)
            # print "GG mono_gal HLR: {:12.10g}".format(mono_gal.calculateHLR())
            if self.achromatic_galaxy:
                gal = mono_gal
                flux = jifparams.flux_from_AB_mag(self.params[0].mag_sed1)
                # print "GG flux: {:12.10g}".format(flux)
                gal = gal.withFlux(flux)
            else:
                SED = self.get_SED()
                gal = galsim.Chromatic(mono_gal, SED)
            gal_shape = galsim.Shear(g=self.params[0].e,
                beta=self.params[0].beta*galsim.radians)
            gal = gal.shear(gal_shape)
            gal = gal.shift(self.params[0].dx, self.params[0].dy)
            # print "GG gal HLR: {:12.10g}".format(gal.calculateHLR())

        elif self.galaxy_model == "Sersic":
            mono_gal = galsim.Sersic(n=self.params[0].n,
                half_light_radius=self.params[0].hlr,
                # flux=self.params[0].gal_flux,
                flux=1.0,
                gsparams=self.gsparams)
            if self.achromatic_galaxy:
                gal = mono_gal
                gal = gal.withFlux(jifparams.flux_from_AB_mag(
                    self.params[0].mag_sed1,
                    pixel_scale_arcsec=self.pixel_scale_arcsec))
            else:
                SED = self.get_SED()
                gal = galsim.Chromatic(mono_gal, SED)
            gal_shape = galsim.Shear(g=self.params[0].e,
                beta=self.params[0].beta*galsim.radians)
            gal = gal.shear(gal_shape)
            gal = gal.shift(self.params[0].dx, self.params[0].dy)

        elif self.galaxy_model == "BulgeDisk":
            mono_bulge = galsim.Spergel(nu=self.params[0].nu_bulge,
                half_light_radius=self.params[0].hlr_bulge,
                flux=1.0,
                gsparams=self.gsparams)
            SED_bulge = self.get_SED(gal_comp='bulge')
            bulge = galsim.Chromatic(mono_bulge, SED_bulge)
            bulge_shape = galsim.Shear(g=self.params[0].e_bulge,
                beta=self.params[0].beta_bulge*galsim.radians)
            bulge = bulge.shear(bulge_shape)
            bulge = bulge.shift(self.params[0].dx_bulge, self.params[0].dy_bulge)

            mono_disk = galsim.Spergel(nu=self.params[0].nu_disk,
                half_light_radius=self.params[0].hlr_disk,
                flux=1.0,
                gsparams=self.gsparams)
            SED_disk = self.get_SED(gal_comp='disk')
            disk = galsim.Chromatic(mono_disk, SED_disk)
            disk_shape = galsim.Shear(g=self.params[0].e_disk,
                beta=self.params[0].beta_disk*galsim.radians)
            disk = disk.shear(disk_shape)
            disk = disk.shift(self.params[0].dx_disk, self.params[0].dy_disk)

            # gal = self.params[0].bulge_frac * bulge + (1 - self.params[0].bulge_frac) * disk
            gal = bulge + disk
            gal = gal.shift(dx, dy)

        else:
            raise AttributeError("Unimplemented galaxy model")
        # print "GG gal flux: {:12.10g}".format(gal.getFlux())
        # print "GG PSF HLR: {:12.10g}".format(self.get_psf(filter_name).calculateHLR())
        final = galsim.Convolve([gal, self.get_psf(filter_name)])
        # print "GG final flux: {:12.10g}".format(final.getFlux())
        # print "GG hlr: {:12.10g}".format(final.calculateHLR())
        # wcs = galsim.PixelScale(self.pixel_scale)'
        
        # print "GG pixel scale: {:8.6f}".format(self.pixel_scale)
        # print "GG gain: {:8.6f}".format(gain)

        try:
            if self.achromatic_galaxy:
                image = final.drawImage(image=out_image, scale=self.pixel_scale,
                    gain=gain, add_to_image=False, method='fft')
                # print "GG image range: {:10.8g}, {:10.8g}".format(np.min(image.array), 
                    # np.max(image.array))
            else:
                image = final.drawImage(bandpass=self.filters[filter_name],
                    image=out_image, scale=self.pixel_scale, gain=gain,
                    add_to_image=False,
                    method='fft')
            if add_noise:
                if self.telescope_name == "WFIRST":
                    sky_level = telescopes.wfirst_sky_background(filter_name, self.filters[filter_name])
                    image += sky_level
                    galsim.wfirst.allDetectorEffects(image)
                    image -= (sky_level + galsim.wfirst.dark_current*galsim.wfirst.exptime) / galsim.wfirst.gain
                else:
                    if self.noise is not None:
                        if snr is None:
                            image.addNoise(self.noise)
                        else:
                            image.addNoiseSNR(self.noise, snr=snr)
                    else:
                        raise AttributeError("A GalSim noise model must be specified to add noise to an image.")
        except RuntimeError:
            print "Trying to make an image that's too big."
            print self.get_params()
            image = None
        return image

    def get_psf_image(self, filter_name='r', ngrid=None, out_image=None, gain=1.0, add_noise=False):
        psf = self.get_psf(filter_name)
        if self.psf_model_type == 'InterpolatedImage':
            image = psf
        elif self.psf_model_type == 'PSFModel class':
            image = self.psf_model.get_psf_image(filter_name=filter_name, out_image=out_image,
                                                 ngrid=ngrid, pixel_scale_arcsec=self.pixel_scale,
                                                 gain=gain, normalize_flux=True)
            if add_noise:
                image.addNoise(self.noise)
        else:
            image_epsf = psf.drawImage(image=out_image, scale=self.pixel_scale, nx=ngrid, ny=ngrid,
                                       gain=gain)
            image = image_epsf
            if add_noise:
                image.addNoise(self.noise)
        return image

    def get_segment(self):
        pass


def save_bandpasses_to_segment(seg, gg, filter_names, telescope_name="LSST", scale=1):
    """
    Read filter files and copy to a 'segment' HDF5 file
    """
    path, filename = os.path.split(__file__)
    filter_names = list(filter_names)
    waves_nm_list = []
    throughputs_list = []
    effective_wavelengths = []
    for i, f in enumerate(filter_names):
        bp = np.loadtxt(os.path.join(path, "input/{}_{}.dat".format(
            telescope_name, f)))
        waves_nm_list.append(bp[:,0]*scale)
        throughputs_list.append(bp[:,1])
        effective_wavelengths.append(gg.filters[f].effective_wavelength)
    # print "effective wavelengths (nm):", effective_wavelengths
    seg.save_bandpasses(filter_names,
        waves_nm_list, throughputs_list,
        effective_wavelengths=effective_wavelengths,
        telescope=telescope_name.lower())
    return None


def make_test_image():
    gg = GalSimGalaxyModel()
    noise = telescopes.lsst_noise(random_seed=58979832601)
    ###
    segfile = os.path.join(os.path.dirname(__file__),
        '../data/TestData/test_image_data.h5')
    print("Writing {}".format(segfile))
    gg.save_footprint(segfile, noise=noise)
    return None


def make_lsst_images(filter_name_ground='r', filter_name_space='F184',
                     file_lab='', galaxy_model="Spergel",
                     achromatic_galaxy=False):
    """
    Use the GalSimGalaxyModel class to make test images of a galaxy for LSST and WFIRST.
    """
    import os
    import h5py

    ngrid_lsst = 70

    print("Making test images for LSST and WFIRST")

    # LSST
    print("\n----- LSST -----")
    lsst = GalSimGalaxyModel(galaxy_model="Spergel",
        telescope_model="LSST",
        psf_model="Model",
        active_parameters=[])

    lsst.set_param_by_name('redshift', 1.0)
    lsst.set_param_by_name('nu', 0.3)
    lsst.set_param_by_name('hlr', 1.0)
    lsst.set_param_by_name('e', 0.26)
    lsst.set_param_by_name('beta', 0.5236)
    lsst.set_param_by_name('mag_sed1', 28.5)
    lsst.set_param_by_name('dx', 0.7)
    lsst.set_param_by_name('dy', -0.3)
    lsst.set_param_by_name('psf_fwhm', 0.6)

    lsst_noise = telescopes.lsst_noise(random_seed=15983217)

    # Save the image
    lsst.save_image("../data/TestData/test_lsst_image" + file_lab + ".fits",
        filter_name=filter_name_ground,
        out_image=galsim.Image(ngrid_lsst, ngrid_lsst))
    lsst.plot_image("../data/TestData/test_lsst_image" + file_lab + ".png",
        ngrid=ngrid_lsst,
        filter_name=filter_name_ground, title="LSST " + filter_name_ground,
        noise=lsst_noise)
    # Save the corresponding PSF
    lsst.save_psf("../data/TestData/test_lsst_psf" + file_lab + ".fits",
        ngrid=ngrid_lsst/2, filter_name=filter_name_ground)
    lsst.plot_psf("../data/TestData/test_lsst_psf" + file_lab + ".png",
        ngrid=ngrid_lsst/2, title="LSST " + filter_name_ground,
        filter_name=filter_name_ground,
        noise=None)

    lsst_data = lsst.get_image(out_image=galsim.Image(ngrid_lsst, ngrid_lsst), 
                               filter_name=filter_name_ground)
    lsst_data.addNoise(lsst_noise)

    print "noise rms: {:12.10g}".format(np.sqrt(lsst_noise.getVariance()))

    # -------------------------------------------------------------------------
    ### Save a file with image data for input to the Roaster
    segfile = os.path.join(os.path.dirname(__file__),
        '../data/TestData/test_lsst_data' + file_lab + '.h5')
    print("Writing {}".format(segfile))
    seg = footprints.Footprints(segfile)

    seg_ndx = 0
    src_catalog = lsst.params
    seg.save_source_catalog(src_catalog, segment_index=seg_ndx)

    dummy_mask = 1.0
    dummy_background = 0.0

    ### Ground data
    seg.save_images([lsst_data.array], [lsst_noise.getVariance()], [dummy_mask],
        [dummy_background], segment_index=seg_ndx,
        telescope='lsst',
        filter_name=filter_name_ground)
    tel = telescopes.k_telescopes["LSST"]
    seg.save_tel_metadata(telescope='lsst',
        primary_diam=tel['primary_diam_meters'],
        pixel_scale_arcsec=tel['pixel_scale'],
        atmosphere=tel['atmosphere'])
    seg.save_psf_images([lsst.get_psf_image(filter_name_ground, ngrid=64).array],
        segment_index=seg_ndx,
        telescope='lsst',
        filter_name=filter_name_ground)
    # save_bandpasses_to_segment(seg, lsst, telescopes.k_lsst_filter_names, "LSST")

    return None


def make_blended_test_image(num_sources=3, random_seed=75256611, 
                            galaxy_model="Spergel", achromatic_galaxy=True):
    lsst_pixel_scale_arcsec = 0.2

    ellipticities = [0.05, 0.3, 0.16]
    hlrs = [1.8, 1.0, 2.0]
    orientations = np.array([0.1, 0.25, -0.3]) * np.pi

    noise_model = telescopes.lsst_noise(82357)

    filter_name = 'y'

    ### Setup the 'segment' image that will contain all the galaxies in the blend
    npix_segment = 128
    # segment_pos = galsim.CelectialCoord(ra=90.*galsim.degrees, dec=-10.*galsim.degrees)
    segment_image = galsim.ImageF(npix_segment, npix_segment, scale=lsst_pixel_scale_arcsec)

    ### Define the galaxy positions in the segment (relative to the center of the segment image)
    ### (see galsim demo13.py)
    pos_rng = galsim.UniformDeviate(random_seed)
    x_gal = (0.4 + 0.1 * np.array([pos_rng() for i in xrange(num_sources)])) * npix_segment
    y_gal = (0.4 + 0.1 * np.array([pos_rng() for i in xrange(num_sources)])) * npix_segment

    npix_gal = 100

    for isrcs in xrange(num_sources):
        # ### Draw every source using the full output array
        # b = galsim.BoundsI(1, nx, 1, ny)
        # sub_image = segment_image[b]

        sub_image = galsim.Image(npix_gal, npix_gal, scale=lsst_pixel_scale_arcsec)

        src_model = GalSimGalaxyModel(
            telescope_name="LSST",
            pixel_scale_arcsec=telescopes.k_telescopes['LSST']['pixel_scale'],
            noise=telescopes.lsst_noise(82357),
            galaxy_model=galaxy_model,
            primary_diam_meters=8.4,
            filter_names=telescopes.k_lsst_filter_names,
            filter_wavelength_scale=1.0,
            atmosphere=True,
            achromatic_galaxy=achromatic_galaxy)

        src_model.params[0]["e"] = ellipticities[isrcs]
        src_model.params[0]["beta"] = orientations[isrcs]
        src_model.params[0]["hlr"] = hlrs[isrcs]
        # p = src_model.get_params()

        # src_model.set_params(p)

        gal_image = src_model.get_image(out_image=sub_image, filter_name=filter_name)

        ix = int(math.floor(x_gal[isrcs]+0.5))
        iy = int(math.floor(y_gal[isrcs]+0.5))

        # Create a nominal bound for the postage stamp given the integer part of the
        # position.
        sub_bounds = galsim.BoundsI(ix-0.5*npix_gal, ix+0.5*npix_gal-1,
                                    iy-0.5*npix_gal, iy+0.5*npix_gal-1)
        sub_image.setOrigin(galsim.PositionI(sub_bounds.xmin, sub_bounds.ymin))

        # Find the overlapping bounds between the large image and the individual postage
        # stamp.
        bounds = sub_image.bounds & segment_image.bounds

        segment_image[bounds] += sub_image[bounds]

    segment_image.addNoise(noise_model)

    outfile = "../data/TestData/test_lsst_blended_image.fits"
    print("Saving to {}".format(outfile))
    segment_image.write(outfile)


if __name__ == "__main__":
    make_test_image()
    # make_blended_test_image(num_sources=2)
