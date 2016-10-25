#!/usr/bin/env python
# encoding: utf-8
"""
Roaster.py

Draw samples of source model parameters given the pixel data for image segments.

This script is intended to run on one footprint at a time (i.e., call multiple
instances of the script to process multiple footprint).
"""

import argparse
import sys
import os.path
import string
import copy
import numpy as np
from scipy.optimize import minimize as sp_minimize
from scipy.optimize import basinhopping
import h5py
import emcee
# from emcee.utils import MPIPool
###
import config as jifconf
import telescopes as jiftel
import parameters as jifparams
import galsim_galaxy
import psf_model as pm
import galsim

import logging


# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
#logging.basicConfig(filename='logs/Roaster.log',
#                     level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


# Store the pixel data as global (module scope) variables so emcee
# mutlithreading doesn't need to repeatedly pickle these all the time.
# self.pixel_data = []
# self.pix_noise_var = []
# self.src_models = []


class EmptyPrior(object):
    def __init__(self):
        pass
    def __call__(self, *args):
        return 0.0


class Roaster(object):
    """
    Draw samples of source model parameters via MCMC.

    We allow for 3 types of multiplicity in the data:
        (a) Multiple instruments observing the same source,
        (b) Multiple epochs of the same sources for a given instrument,
        (c) Multiple sources in a single 'cutout' (e.g., for blended sources)
    For scenarios (a) and (b), the likelihood is a product of the likelihoods
    for each epoch or instrument.
    For scenario (c), we need a list of source models that generate a model
    image that is fed to the likelihood function for a single epoch or
    instrument.

    @param lnprior_omega      Prior class for the galaxy model parameters
    @param lnprior_Pi         Prior class for the PSF model parameters
    @param data_format        Format for the input data file.
    @param galaxy_model_type  Type of parametric galaxy model - see
                              galsim_galaxy types.
                              ['Sersic', 'Spergel', 'BulgeDisk' (default), 
                               'star']
                              If 'star' is specified, then any PSF model 
                              sampling is turned off. This takes precedence over
                              any other PSF sampling arguments or                 settings.
    @param telescope          Select only this telescope observations from the
                              input, if provided.
                              If not provided, then get data for all available
                              telescopes.
    @param filters_to_load    Select only data for these named filters from the
                              input, if provided.
                              If not provided, then get data for all available
                              filters (for each telescope).
    @param debug              Save debugging outputs (including model images
                              per step?)
    @param model_paramnames   Names of the galaxy model parameters to sample in.
                              These must match names in a galsim_galaxy model
                              and/or a psf_model.
    @param achromatic_galaxy  Use an achromatic galaxy model (or, by default use
                              chromatic GalSim features)
    """
    def __init__(self, lnprior_omega=None,
                 lnprior_Pi=None,
                 data_format='test_galsim_galaxy',
                 galaxy_model_type='Spergel',
                 telescope=None,
                 filters_to_load=None,
                 debug=False,
                 model_paramnames=['hlr', 'e', 'beta'],
                 achromatic_galaxy=False):
        if lnprior_omega is None:
            self.lnprior_omega = EmptyPrior()
        else:
            self.lnprior_omega = lnprior_omega
        if lnprior_Pi is None:
            self.lnprior_Pi = pm.FlatPriorPSF()
        else:
            self.lnprior_Pi = lnprior_Pi
        self.data_format = data_format
        self.galaxy_model_type = galaxy_model_type
        self.telescope = telescope
        self.filters_to_load = filters_to_load
        self.debug = debug
        self.model_paramnames = model_paramnames
        self.achromatic_galaxy = achromatic_galaxy
        ### Check if any of the active parameters are for a PSF model.
        ### If so, we will sample in the PSF and need to setup accordingly
        ### when the Load() function is called.
        self.sample_psf = False
        self.psf_model_paramnames = []
        # if galaxy_model_type == 'star':
        #     if not np.all(['psf' in p for p in model_paramnames]):
        #         raise ValueError("All parameters must have 'psf' label for 'star' model_type")
        # else:
        if np.any(['psf' in p for p in model_paramnames]):
            self.sample_psf = True
            self.psf_model_paramnames = jifparams.select_psf_paramnames(
                model_paramnames)
        ### The complimentary set of parameters are those for the galaxy model(s)
        self.galaxy_model_paramnames = jifparams.select_galaxy_paramnames(
            model_paramnames)


        ### Count the number of calls to self.lnlike
        self.istep = 0

    def _get_branch_name(self, i):
        branches = ['ground', 'space']
        # define the parent branch (i.e. telescope)
        if self.epoch is None:
            branch = branches[i]
        else:
            branch = branches[self.epoch]
        return branch

    def Load(self, infile, segment=None, epoch_num=None, use_PSFModel=False):
        """
        Load image cutouts from file for the given segment, where segment is
        an integer reference.

        The input file should contain:
            1. Pixel data for a cutout of N galaxies
            2. A segmentation mask for the cutout
            3. The pixel noise model (e.g., variance per pixel)
            4. The WCS information for the image
            5. Background model(s)
        This information should be replicated for each epoch and/or instrument.

        Each source in a blend group should contain information on:
            1. a, b, theta (ellipticity semi-major and semi-minor axes,
            orientation angle)
            2. centroid position (x,y)
            3. flux
        These values will be used to initialize any model-fitting (e.g., MCMC)
        algorithm.

        @param infiles      List of input filenames to load.
        @param segment      Index of the segment to load. Choose segment 0 if not supplied.
        @param epoch_num    Specify a single epoch number to load (indexed from 0). If not supplied
                            load all available epochs for each available telescope and filter.
                            Requires specification of a single telescope and filter_to_load at
                            instantiation of the Roaster instance.
        @param use_PSFModel Force the use of a PSFModel class instance to model PSFs even if not
                            sampling any PSF model parameters
        """
        # global self.pixel_data
        # global self.pix_noise_var
        # global self.src_models

        ### Reset the global data lists in case Roaster had been used before
        self.pixel_data = []
        self.pix_noise_var = []

        if epoch_num is not None:
            if self.telescope is None:
                raise ValueError(
                    "Must set telescope before requesting 1 epoch to Load")
            elif not self.achromatic_galaxy and self.filters_to_load is None:
                raise ValueError(
                    "Must set filters_to_load before requesting 1 epoch to Load")
            else:
                epochs = [epoch_num]

        logging.info("<Roaster> Loading image data")
        try:
            if self.data_format == "jif_segment" or self.data_format == "test_galsim_galaxy":
                f = h5py.File(infile, 'r')
                if self.telescope is None:
                    self.num_telescopes = len(f['telescopes'])
                    telescopes = f['telescopes'].keys()
                else:
                    self.num_telescopes = 1
                    telescopes = [self.telescope]
                if self.debug:
                    print("Num. telescopes: {:d}".format(self.num_telescopes))
                    print("Telescope names: {}".format(telescopes))
                if segment == None:
                    segment = 0
                self.num_sources = f['Footprints/seg{:d}'.format(segment)].attrs['num_sources']
                if self.debug:
                    print("Num. sources: {:d}".format(self.num_sources))

                instruments = telescopes

                have_bandpasses = False

                pixel_scales = []
                primary_diams = []
                atmospheres = []
                tel_names = []
                psfs = []
                psf_types = []
                self.filters = {}
                self.filter_names = []
                for itel, tel in enumerate(telescopes):
                    g = 'Footprints/seg{:d}/{}'.format(segment, tel.lower())
                    filter_names = f[g].keys()
                    if self.filters_to_load is not None:
                        filter_names = [filt for filt in filter_names
                                        if filt in self.filters_to_load]
                    if len(filter_names) == 0:
                        raise ValueError("No data available in the requested filters")
                    for ifilt, filter_name in enumerate(filter_names):
                        logging.debug("Loading data for filter {} in telescope {}".format(
                            filter_name, tel))
                        ### If present in the segments file,
                        ### load filter information and instantiate the galsim Bandpass.
                        ### If the filters section is not found, then revert to using
                        ### a known filter selection given the 'telescope' name in
                        ### the segments file. If the telescope name is not known by
                        ### galsim_galaxy then raise an error and stop.
                        fg = 'telescopes/{}/filters/{}'.format(tel, filter_name)
                        try:
                            waves_nm = f[fg + '/waves_nm']
                            throughput = f[fg + '/throughput']
                            table = galsim.LookupTable(x=waves_nm, f=throughput)
                            bp = jiftel.load_filter_file_to_bandpass(table,
                                effective_diameter_meters=jiftel.k_telescopes[tel]['effective_diameter'],
                                exptime_sec=jiftel.k_telescopes[tel]['exptime_zeropoint'])
                            self.filters[filter_name] = bp
                            have_bandpasses = True
                        except KeyError:
                            have_bandpasses = False
                            self.filters = None

                        tel_model = jiftel.k_telescopes[tel]
                        lam_over_diam = (tel_model["filter_central_wavelengths"][filter_name] * 1e-9 /
                                         tel_model["primary_diam_meters"]) * 180*3600/np.pi
                        print "lam_over_diam: {:5.4g} (arcseconds)".format(lam_over_diam)

                        h = 'Footprints/seg{:d}/{}/{}'.format(segment, tel.lower(), filter_name)
                        if epoch_num is None:
                            nepochs = len(f[h])
                            epochs = range(nepochs)
                            if self.debug:
                                print("Number of epochs for {}: {:d}".format(tel, nepochs))
                        for iepoch in epochs:
                            seg = f[h + '/epoch_{:d}'.format(iepoch)]
                            # obs = f[branch+'/observation']
                            dat = seg['image']
                            noise = seg['noise']
                            if self.debug:
                                print itel, ifilt, iepoch, "dat shape:", dat.shape
                            self.pixel_data.append(np.array(dat))
                            self.pix_noise_var.append(seg.attrs['variance'])
                            ###
                            if self.sample_psf:
                                psf_types.append('PSFModel class')
                                psfs.append(pm.PSFModel(
                                    active_parameters=self.psf_model_paramnames,
                                    telescope=self.telescope,
                                    achromatic=self.achromatic_galaxy,
                                    lam_over_diam=lam_over_diam,
                                    gsparams=None))
                            elif use_PSFModel:
                                psf_types.append('PSFModel class')
                                psfs.append(pm.PSFModel(
                                    active_parameters=[],
                                    telescope=self.telescope,
                                    achromatic=self.achromatic_galaxy,
                                    lam_over_diam=lam_over_diam,
                                    gsparams=None))
                            else:
                                psf_types.append(seg.attrs['psf_type'])
                                psfs.append(galsim.Image(seg['psf'][...]))
                            ###
                            tel_group = f['telescopes/{}'.format(tel)]
                            pixel_scales.append(tel_group.attrs['pixel_scale_arcsec'])
                            primary_diams.append(tel_group.attrs['primary_diam'])
                            atmospheres.append(tel_group.attrs['atmosphere'])
                            tel_names.append(tel)
                            self.filter_names.append(filter_name)
                print "Have data for instruments:", instruments
                print "pixel noise variances:", self.pix_noise_var
                print "PSF types:", psf_types
            else:
                raise KeyError("Unsupported input data format in Roaster")
                # if segment == None:
                #     logging.info("<Roaster> Must specify a segment number as an integer")
                # print "Have data for instruments:", instruments
        except KeyError:
            print "Error parsing the input file -- Perhaps it's in a deprecated format? Try rerunning footprint extraction."
            sys.exit(1)

        if have_bandpasses:
            print "Filters:", self.filters.keys()
        else:
            self.filters = None

        nimages = len(self.pixel_data)
        ### Here 'epoch' refers to any repeat observation of a source, whether it's in a different
        ### filter, telescope, or just a different time with the same telescope and filter.
        self.num_epochs = nimages
        self.nx = np.zeros(nimages, dtype=int)
        self.ny = np.zeros(nimages, dtype=int)
        for i in xrange(nimages):
            self.nx[i], self.ny[i] = self.pixel_data[i].shape
            if self.debug:
                print "nx, ny, i:", self.nx[i], self.ny[i], i
                print "tel_name:", tel_names[i]
                print "pixel_scale (arcsec):", pixel_scales[i]
                print "primary_diam (m):", primary_diams[i]
                print "atmosphere:", atmospheres[i]

        logging.debug("<Roaster> num_epochs: {:d}, num_sources: {:d}".format(
            self.num_epochs, self.num_sources))
        print "Active parameters: ", self.model_paramnames

        self._init_galaxy_models(nimages, tel_names, pixel_scales,
                                 primary_diams, atmospheres, psfs)
        logging.debug("<Roaster> Finished loading data")
        if self.debug:
            print "\npixel data shapes:", [dat.shape for dat in self.pixel_data]
        return None

    def _init_galaxy_models(self, nimages, tel_names, pixel_scales, 
                            primary_diams, atmospheres, psfs):
        self.src_models = []
        self.src_models = [[galsim_galaxy.GalSimGalaxyModel(
                                telescope_model=tel_names[idat],
                                galaxy_model=self.galaxy_model_type,
                                psf_model=psfs[idat],
                                active_parameters=self.model_paramnames)
                            for idat in xrange(nimages)]
                           for isrcs in xrange(self.num_sources)]
        ### Count galaxy 'active' parameters plus distinct PSF parameters for all epochs.
        self.n_psf_params = self.src_models[0][0].n_psf_params
        self.n_gal_params = self.src_models[0][0].n_params - self.n_psf_params
        self.n_params = self.n_gal_params + self.num_epochs * self.n_psf_params
        return None

    def initialize_param_values(self, param_file_name):
        """
        Initialize model parameter values from a config file
        """
        import ConfigParser
        config = ConfigParser.RawConfigParser()
        config.read(param_file_name)

        params = config.items("parameters")
        for paramname, val in params:
            vals = str.split(val, ' ')
            if len(vals) > 1: ### Assume multiple sources
                fval = [float(v) for v in vals[0:self.num_sources]]
            else:
                fval = float(val)
            self.set_param_by_name(paramname, fval)
        return None

    def _get_noise_var(self, i=0):
        return self.pix_noise_var[i]

    def _get_raw_params(self):
        return self.src_models[0][0].params

    def get_params(self):
        """
        Make a flat array of active model parameters for all sources

        If sampling PSF parameters as well as galaxy parameters, then append distinct PSF model
        parameters for each observation of the given segment.

        For use in MCMC sampling.
        """
        ### Get the galaxy model parameters, and, if PSF sampling, the epoch-0 PSF model parameters
        p = np.array([m[0].get_params() for m in self.src_models]).ravel()
        ### If PSF sampling, append the PSF model parameters for the remaining epochs
        if self.sample_psf and self.num_epochs > 1:
            p_psf = np.array([m[i].get_psf_params() for m in self.src_models
                              for i in xrange(1, self.num_epochs)]).ravel()
            p = np.concatenate((p, p_psf))
        return p

    def set_params(self, p):
        """
        Set the active galaxy model parameters for all galaxies in a segment
        from a flattened array `p`.

        `p` is assumed to be packed as [(p1_gal1, ..., pn_gal1), ...,
        (p1_gal_m, ..., pn_gal_m)]
        for n parameters per galaxy and m galaxies in the segment.

        For use in MCMC sampling.
        """
        valid_params = True
        for isrcs in xrange(self.num_sources):
            imin = isrcs * self.n_params
            imax = (isrcs + 1) * self.n_params
            p_set = p[imin:imax]
            ### Assign the parameters for source model isrcs in epoch iepochs.
            ### The galaxy parameters are the same across epochs, but PSF parameters are 
            ### different.
            p_gal_set = p_set[0:self.n_gal_params]
            for iepochs in xrange(self.num_epochs):
                jmin = self.n_gal_params + iepochs * self.n_psf_params
                jmax = jmin + self.n_psf_params

                p_psf_set = p_set[jmin:jmax]
                p_set_iepoch = np.concatenate((p_gal_set, p_psf_set))

                self.src_models[isrcs][iepochs].set_params(p_set_iepoch)
                valid_params *= self.src_models[isrcs][iepochs].validate_params()
        return valid_params

    def set_param_by_name(self, paramname, value):
        """
        Set a galaxy or PSF model parameter by name.

        Can pass a single value that will be set for all source models, or a
        list of length num_sources with unique values for each source (but
        common across all epochs).

        But, if the named parameter is a PSF model parameter, then set the
        value only for the matching epoch (passing a list is not defined in
        this case)
        """
        if isinstance(value, list):
            if len(value) == self.num_sources:
                for isrc in xrange(self.num_sources):
                    for idat in xrange(self.num_epochs):
                        self.src_models[isrc][idat].set_param_by_name(paramname,
                                                                      value[isrc])
            else:
                raise ValueError("If passing list, must be of length num_sources")
        elif isinstance(value, float):
            logging.debug("Setting {} to {:8.6f}".format(paramname, value))
            if 'psf' in paramname:
                p = paramname.split("_")
                if len(p) != 3:
                    raise ValueError("PSF parameter names must have the epoch index appended")
                epoch_num = int(p[2]) - 1
                pname = string.join(p[0:2], "_")
                for isrcs in xrange(self.num_sources):
                    self.src_models[isrcs][epoch_num].set_param_by_name(pname, value)
            else:
                for isrcs in xrange(self.num_sources):
                    for idat in xrange(self.num_epochs):
                        self.src_models[isrcs][idat].set_param_by_name(paramname, value)
        else:
            raise ValueError("Unsupported type for input value")

    def lnprior(self, omega):
        valid_params = self.set_params(omega)
        if valid_params:
            ### Iterate over distinct galaxy models in the segment and evaluate the
            ### prior for each one.
            lnp = 0.0

            for isrcs in xrange(self.num_sources):
                # imin = isrcs * self.n_params
                # imax = (isrcs + 1) * self.n_params
                ### Pass active + inactive parameters, with names included
                ### We index only the first 'epoch' here, because the source model parameters should
                ### be the same for all epochs.
                p = copy.deepcopy(self.src_models[isrcs][0].params)
                lnp += self.lnprior_omega(p)
                if self.sample_psf:
                    ### Iterate over epochs and evaluate the PSF model prior for each epoch.
                    ### Each epoch has a different PSF, so the PSF model parameters are distinct.
                    for iepoch in xrange(self.num_epochs):
                        ppsf = copy.deepcopy(self.src_models[isrcs][iepoch].psf_model.params)
                        lnp += self.lnprior_Pi(ppsf, epoch_num=iepoch)
        else:
            lnp = -np.inf
        return lnp

    def _get_model_image(self, iepochs):
        """
        Create a galsim.Image from the source model(s) for epcoh iepochs
        """
        nx = self.nx[iepochs]
        ny = self.ny[iepochs]
        model_image = galsim.ImageF(nx, ny,
            scale=self.src_models[0][iepochs].pixel_scale_arcsec, init_value=0.)
        # print "Roaster model image pixel scale:", self.src_models[0][iepochs].pixel_scale

        for isrcs in xrange(self.num_sources):
            # print "Roaster model parameters:", self.src_models[isrcs][iepochs].params            
            sub_image = galsim.Image(nx, ny, init_value=0.)
            model = self.src_models[isrcs][iepochs].get_image(out_image=sub_image,
                filter_name=self.filter_names[iepochs])
            ix = nx/2
            iy = ny/2
            # print isrcs, ix, iy, nx, ny
            sub_bounds = galsim.BoundsI(int(ix-0.5*nx), int(ix+0.5*nx-1),
                                        int(iy-0.5*ny), int(iy+0.5*ny-1))
            sub_image.setOrigin(galsim.PositionI(sub_bounds.xmin,
                                                 sub_bounds.ymin))

            # Find the overlapping bounds between the large image and the 
            # individual stamp.
            bounds = sub_image.bounds & model_image.bounds
            model_image[bounds] += sub_image[bounds]


            # ### Draw every source using the full output array
            # b = galsim.BoundsI(1, self.nx[iepochs], 1, self.ny[iepochs])
            # sub_image = model_image[b]
            # model = self.src_models[isrcs][iepochs].get_image(sub_image,
            #     filter_name=self.filter_names[iepochs], add_noise=add_noise)
            # model_image[b] + sub_image[b]
        return model_image

    def lnlike(self, omega, *args, **kwargs):
        """
        Evaluate the log-likelihood function for joint pixel data for all
        galaxies in a blended group given all available imaging and epochs.

        See GalSim/examples/demo5.py for how to add multiple sources to a single
        image.
        """

        self.istep += 1
        valid_params = self.set_params(omega)

        if valid_params:
            lnlike = 0.0
            for iepochs in xrange(self.num_epochs):
                model_image = self._get_model_image(iepochs)
                if model_image is None:
                    lnlike = -np.inf
                else:
                    if self.debug:
                        if not os.path.exists('debug'):
                            os.makedirs('debug')
                        model_image_file_name = os.path.join('debug',
                            'model_image_iepoch%d_istep%d.fits' % (iepochs,
                                self.istep))
                        model_image.write(model_image_file_name)
                        logging.debug('Wrote model image to %r',
                            model_image_file_name)

                        import matplotlib.pyplot as plt
                        fig = plt.figure(figsize=(10,10))
                        plt.imshow(self.pixel_data[iepochs] - model_image.array,
                                   interpolation='none', cmap=plt.cm.pink)
                        plt.colorbar()
                        plt.savefig(os.path.join('debug', 'resid_iepoch%d_istep%d.png' % (iepochs,
                            self.istep)))

                    delta = (self.pixel_data[iepochs] - model_image.array)
                    lnlike += (-0.5 * np.sum(delta ** 2) /
                        self.pix_noise_var[iepochs])
        else:
            lnlike = -np.inf
        return lnlike

    def __call__(self, omega, *args, **kwargs):
        return self.lnlike(omega, *args, **kwargs) + self.lnprior(omega)

# ---------------------------------------------------------------------------------------
# MCMC routines
# ---------------------------------------------------------------------------------------
# def walker_ball(omega, spread, nwalkers):
#     return [omega+(np.random.rand(len(omega))*spread-0.5*spread)
#             for i in xrange(nwalkers)]

def optimize_params(omega_interim, roaster, quiet=False):
    """
    Optimize the ln-posterior function
    """
    def neg_lnp(omega, *args, **kwargs):
        return -roaster(omega, *args, **kwargs)

    ### Replicate the parameter names for each source we're fitting
    paramnames = roaster.model_paramnames * roaster.num_sources

    print "\n"
    print "--optimize_params-- paramnames: ", paramnames
    print "--optimize_params-- param bounds:", jifparams.get_bounds(paramnames)
    print "--optimize_params-- omega_interim:", omega_interim
    print "\n"

    res = sp_minimize(fun=neg_lnp,
                      x0=omega_interim,
                      # method='L-BFGS-B',
                      method='SLSQP',
                      jac=False, # Estimate Jacobian numerically
                      # tol=1e-10,
                      bounds=jifparams.get_bounds(paramnames),
                      options={
                          'ftol': 1e-10,
                          # 'eps': 1.0e-9,
                          'maxiter': 200,
                          'disp': not quiet # Set True to print convergence messages
                      })
    # #----------------------------
    # res = sp_minimize(fun=neg_lnp,
    #               x0=omega_interim,
    #               method='L-BFGS-B',
    #               jac=False, # Estimate Jacobian numerically
    #               bounds=jifparams.get_bounds(paramnames),
    #               options={
    #                   'ftol': 1e-12,
    #                   'gtol': 1e-8,
    #                   'factr': 1e1,
    #                   'maxcor': 10,
    #                   # 'eps': 1.0e-6,
    #                   # 'maxiter': 200,
    #                   'disp': True # Set True to print convergence messages
    #               })
    #----------------------------
    # res = sp_minimize(fun=neg_lnp,
    #               x0=omega_interim,
    #               method='TNC',
    #               jac=False, # Estimate Jacobian numerically
    #               bounds=jifparams.get_bounds(paramnames),
    #               options={
    #                   'disp': True # Set True to print convergence messages
    #               })
    print "Optimization result:", res.success, res.x
    if res.success:
        omega_interim = res.x
    return omega_interim, res.success
    # -----------------------------
    # minimizer_kwargs = dict(method="L-BFGS-B", bounds=jifparams.get_bounds(paramnames), options={'disp': False})
    # res = basinhopping(neg_lnp, omega_interim, niter=100, minimizer_kwargs=minimizer_kwargs)
    # # print "Optimization result: \n", res
    # print "Optimization result:", res.x
    # return res.x, True

def cluster_walkers(pps, lnps, thresh_multiplier=1):
    """
    Down-select emcee walkers to those with the largest mean posteriors

    Follows the algorithm of Hou, Goodman, Hogg et al. (2012)
    """
    logging.debug("Clustering emcee walkers with threshold multiplier {:3.2f}".format(
        thresh_multiplier))
    pps = np.array(pps)
    lnps = np.array(lnps)
    ### lnps.shape == (Nsteps, Nwalkers) => lk.shape == (Nwalkers,)
    lk = -np.mean(np.array(lnps), axis=0)
    nwalkers = len(lk)
    ndx = np.argsort(lk)
    lks = lk[ndx]
    d = np.diff(lks)
    thresh = np.cumsum(d) / np.arange(1, nwalkers)
    selection = d > (thresh_multiplier * thresh)
    if np.any(selection):
        nkeep = np.argmax(selection)
    else:
        nkeep = nwalkers
    print "pps, lnps:", pps.shape, lnps.shape
    pps = pps[:, ndx[0:nkeep], :]
    lnps = lnps[:, ndx[0:nkeep]]
    print "New pps, lnps:", pps.shape, lnps.shape
    return pps, lnps


def run_emcee_sampler(omega_interim, args, roaster, use_MPI=False):
    """
    Run emcee MCMC algorithm to select interim posterior samples
    """
    logging.debug("Using emcee sampler")
    nvars = len(omega_interim)

    # p0 = walker_ball(omega_interim, 0.01, args.nwalkers)
    p0 = emcee.utils.sample_ball(omega_interim, np.ones_like(omega_interim) * 0.01, args.nwalkers)

    if use_MPI:
        pool = MPIPool(loadbalance=True)
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        pool = None

    # logging.debug("Initializing parameters for MCMC to yield finite posterior values")
    # while not all([np.isfinite(roaster(p)) for p in p0]):
    #     p0 = walker_ball(omega_interim, 0.02, args.nwalkers)
    sampler = emcee.EnsembleSampler(args.nwalkers,
                                    nvars,
                                    roaster,
                                    threads=args.nthreads,
                                    pool=pool)
    nburn = max([1,args.nburn])
    logging.info("Burning with {:d} steps".format(nburn))
    pp, lnp, rstate = sampler.run_mcmc(p0, nburn)
    sampler.reset()
    pps = []
    lnps = []
    lnpriors = []
    logging.info("Sampling")
    for i in range(args.nsamples):
        if np.mod(i+1, 20) == 0:
            print "\tStep {:d} / {:d}, lnp: {:5.4g}".format(i+1, args.nsamples,
                np.mean(pp))
        pp, lnp, rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
        if not args.quiet:
            print i, np.mean(lnp)
            print np.mean(pp, axis=0)
            print np.std(pp, axis=0)
        lnprior = np.array([roaster.lnprior(omega) for omega in pp])
        pps.append(np.column_stack((pp.copy(), lnprior)))
        lnps.append(lnp.copy())

    pps, lnps = cluster_walkers(pps, lnps, thresh_multiplier=4)

    if use_MPI:
        pool.close()
    return pps, lnps


def run_SIRS_sampler(omega_interim, args, roaster):
    """
    Run Sampling Importance-Resampling sampler to select interim posterior samples
    """
    def dmvnorm(x, mu, var, norm):
        if norm is None:
            norm = -0.5 * np.sum(np.log(var))
        delta = x - mu
        return -0.5 * np.sum(delta * delta / var) + norm

    logging.debug("Using SIRS sampler")
    ### Define the mean and cov for the Gaussian proposal distribution
    mu = omega_interim
    var = np.array([jifparams.k_param_vars[p] for p in roaster.model_paramnames])
    cov = np.diag(var)
    # cov = np.loadtxt("output/great3/CGC/000/roaster_CGC_000_seg0_LSST_param_cov.txt")
    # var = np.diag(cov)
    norm = -0.5 * np.sum(np.log(var))
    ### Draw samples
    x = np.random.multivariate_normal(mu, cov, size=args.nsamples)

    ### Evaluate importance sampling weights
    ww = np.array([roaster(v) - dmvnorm(v, mu, var, norm) for v in x])
    ww_norm = np.logaddexp.reduce(ww, axis=0)
    qq = np.array([w - ww_norm for w in ww])    

    ### Sample with replacement using probabilities qq
    n = args.nsamples
    ndx = np.random.choice(n, 100, replace=True, p=np.exp(qq))
    logging.debug("Number of unique samples in SIRS sampler: {:d}".format(len(np.unique(ndx))))

    pps = None
    lnps = None
    return pps, lnps


def do_sampling(args, roaster, return_samples=False, sampler='emcee'):
    """
    Execute the MCMC chain to fit galaxy (and PSF) model parameters to a segment

    Save the MCMC chain steps to an HDF5 file, and optionally also return the
    steps if `return_samples` is True.
    """
    omega_interim = roaster.get_params()
    logging.info("Have {:d} sampling parameters".format(len(omega_interim)))
    print "Starting parameters: ", omega_interim

    # omega_interim, opt_success = optimize_params(omega_interim, roaster, args.quiet)
    ## Optimization needs fixing - turn off for now [2016-10-22] 
    opt_success = True

    if sampler == 'emcee':
        ### Double the burn-in period if optimization failed
        if not opt_success:
            args.nburn *= 2
        pps, lnps = run_emcee_sampler(omega_interim, args, roaster)
    elif sampler == 'sirs':
        pps, lnps = run_SIRS_sampler(omega_interim, args, roaster)
    else:
        raise KeyError("Unsupported sampler type in Roaster: {}".format(sampler))

    write_results(args, pps, lnps, roaster)
    if return_samples:
        return pps, lnps
    else:
        return None


def write_results(args, pps, lnps, roaster):
    if args.telescope is None:
        tel_lab = ""
    else:
        tel_lab = "_{}".format(args.telescope)
    outdir = os.path.dirname(args.outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = args.outfile + tel_lab + ".h5"
    logging.info("Writing MCMC results to %s" % outfile)
    f = h5py.File(outfile, 'w')

    if roaster.telescope is not None:
        telescope = roaster.telescope
    else:
        telescope = 'None'

    ### Store outputs in an HDF5 (sub-)group so we don't always 
    ### need a separate HDF5 file for every segment.
    group_name='Samples/seg{:d}'.format(args.segment_number)
    g = f.create_group(group_name)

    ### Save attributes so we can later instantiate Roaster with the same
    ### settings.
    g.attrs['infile'] = args.infiles[0]
    if isinstance(args.segment_number, int):
        g.attrs['segment_number'] = args.segment_number
    else:
        g.attrs['segment_number'] = 'all'
    g.attrs['epoch_num'] = args.epoch_num
    g.attrs['galaxy_model_type'] = roaster.galaxy_model_type
    if roaster.filters_to_load is not None:
        g.attrs['filters_to_load'] = roaster.filters_to_load
    else:
        g.attrs['filters_to_load'] = 'None'
    g.attrs['telescope'] = telescope
    g.attrs['model_paramnames'] = roaster.model_paramnames
    g.attrs['achromatic_galaxy'] = roaster.achromatic_galaxy

    ### Collect the galaxy model parameter names, appending source indices if 
    ### needed.
    ### If sampling in PSF parameters, handle these separately from the galaxy
    ### parameters.
    if roaster.num_sources == 1:
        paramnames = roaster.galaxy_model_paramnames
    elif roaster.num_sources > 1:
        paramnames = [p + '_src{:d}'.format(isrc) for isrc in xrange(roaster.num_sources)
                      for p in roaster.galaxy_model_paramnames]
    if roaster.sample_psf:
        if roaster.num_epochs == 1:
            paramnames += roaster.psf_model_paramnames
        elif roaster.num_epochs > 1:
            paramnames += [p + '_{:d}'.format(iepoch) for iepoch in xrange(roaster.num_epochs)
                           for p in roaster.psf_model_paramnames]

    ### Write the MCMC samples and log probabilities
    if "post" in g:
        del g["post"]
    post = g.create_dataset("post", data=np.transpose(np.dstack(pps), [2,0,1]))
    # pnames = np.array(roaster.src_models[0][0].paramnames)
    post.attrs['paramnames'] = paramnames
    if "logprobs" in g:
        del g["logprobs"]
    logprobs = g.create_dataset("logprobs", data=np.vstack(lnps))
    g.attrs["nburn"] = args.nburn
    f.close()
    return None

# ---------------------------------------------------------------------------------------
# Prior distributions for interim sampling of galaxy model parameters
# ---------------------------------------------------------------------------------------
class DefaultPriorSpergel(object):
    """
    A default prior for a single-component Spergel galaxy
    """
    def __init__(self):
        ### Gaussian mixture in 'nu' parameter
        self.nu_mean_1 = -0.6 ### ~ de Vacouleur profile
        self.nu_mean_2 = 0.5 ### ~ exponential profile
        self.nu_var_1 = 0.05
        self.nu_var_2 = 0.01
        ### Gamma distribution keeping half-light radius from becoming
        ### much larger than 1 arcsecond or too close to zero.
        self.hlr_shape = 2.
        self.hlr_scale = 0.15
        ### Gaussian distribution in log flux
        self.mag_mean = 20.0
        self.mag_var = 7.0
        ### Beta distribution in ellipticity magnitude
        self.e_beta_a = 1.5
        self.e_beta_b = 5.0
        ### Gaussian priors in centroid parameters
        self.pos_var = 0.5

    def _lnprior_nu(self, nu):
        d1 = (nu - self.nu_mean_1)
        d2 = (nu - self.nu_mean_2)
        return -0.5 * (d1*d1/self.nu_var_1 + d2*d2/self.nu_var_2)

    def _lnprior_hlr(self, hlr):
        return (self.hlr_shape-1.)*np.log(hlr) - (hlr / self.hlr_scale)

    def _lnprior_mag(self, mag):
        delta = mag - self.mag_mean
        return -0.5 * delta * delta / self.mag_var

    def __call__(self, omega):
        lnp = 0.0
        ### 'nu' parameter - peaked at exponential and de Vacouleur profile values
        lnp += self._lnprior_nu(omega[0].nu)
        ### Half-light radius
        lnp += self._lnprior_hlr(omega[0].hlr)
        ### Flux
        lnp += self._lnprior_mag(omega[0].mag_sed1)
        lnp += self._lnprior_mag(omega[0].mag_sed2)
        lnp += self._lnprior_mag(omega[0].mag_sed3)
        lnp += self._lnprior_mag(omega[0].mag_sed4)
        ### Ellipticity magnitude
        e = omega[0].e
        lnp += (self.e_beta_a-1.)*np.log(e) + (self.e_beta_b-1.)*np.log(1.-e)
        ### Centroid (x,y) perturbations
        dx = omega[0].dx
        dy = omega[0].dy
        lnp += -0.5 * (dx*dx + dy*dy) / self.pos_var
        return lnp


class DefaultPriorBulgeDisk(object):
    """
    A default prior for the parameters of a bulge+disk galaxy model
    """
    def __init__(self, z_mean=1.0):
        self.z_mean = z_mean
        self.z_var = 0.05 * (1. + z_mean)
        ### ellipticity variance for bulge/disk components
        self.e_var_bulge = 0.05 ** 2
        self.e_var_disk = 0.3 ** 3

    def __call__(self, omega):
        """
        Evaluate the prior on Bulge + Disk galaxy model parameters for a single
        galaxy.
        """
        ### Redshift
        lnp = -0.5 * (omega[0] - self.z_mean) ** 2 / self.z_var
        ### Ellipticity
        lnp += -0.5 * (omega[3]) ** 2 / self.e_var_bulge
        lnp += -0.5 * (omega[7]) ** 2 / self.e_var_disk
        return lnp

# ------------------------------------------------------------------------------
# For inspection: skip sampling and just save the model initialized in Roaster
# ------------------------------------------------------------------------------
def save_model_image(args, roaster):
    """
    Save the initial Roaster model image to FITS and png files

    This routine is intended for inspection to see what kind of model image is 
    created once Roaster is initialized but before any MCMC sampling. 
    If the model image looks very different from the input data then it may 
    take a long time for the MCMC chain to settle in to a high probability 
    region of parameter space, or there may be a bug somewhere. 
    """
    epoch_num = 0 # FIXME: loop over this index

    model_image = roaster._get_model_image(iepochs=epoch_num)

    dat = roaster.pixel_data[epoch_num]
    mdat = model_image.array

    ### Save a FITS image using the GalSim 'write' method for Image() class
    outfile = args.outfile + "_initial_model_image.fits"
    model_image.write(outfile)
    logging.debug('Wrote model image to %r', outfile)

    ### Save a PNG plot using matplotlib
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    colors = ['#348ABD', '#7A68A6', '#A60628', '#467821', '#CF4457', '#188487']
    fig = plt.figure(figsize=(12, 12/1.618))
    ### Data
    plt.subplot(2, 2, 1)
    plt.imshow(dat, interpolation='none',
               cmap=plt.cm.pink)
    plt.colorbar()
    plt.title("Data")
    ### Model
    plt.subplot(2, 2, 2)
    plt.imshow(mdat, interpolation='none',
               cmap=plt.cm.pink)
    plt.colorbar()
    plt.title("Model")
    ### Data - Model
    resid = dat - mdat
    plt.subplot(2, 2, 4)
    plt.imshow(resid, interpolation='none', cmap=plt.cm.BrBG)
    plt.colorbar()
    plt.title("Residual")
    ### Cross-sections of the profile
    nx, ny = dat.shape
    x = (np.arange(0, nx) - nx/2.) * model_image.scale
    y = (np.arange(0, ny) - ny/2.) * model_image.scale
    plt.subplot(2, 2, 3)
    plt.plot(y, dat[(nx/2), :], color=colors[0], linestyle='solid', label="data")
    plt.plot(x, dat[:, (ny/2)], color=colors[0], linestyle='dashed', label=None)
    plt.plot(y, mdat[(nx/2), :], color=colors[1], linestyle='solid', label="model")
    plt.plot(x, mdat[:, (ny/2)], color=colors[1], linestyle='dashed', label=None)
    plt.legend()
    plt.xlabel(r"position (arcsec.)")
    plt.ylabel(r"surface brightness (ADU/pix$^2$)")
    plt.title("x,y cross-sections")

    outfile = args.outfile + "_initial_model_image.png"
    plt.savefig(outfile, bbox_inches='tight')
    return np.sqrt(np.var(resid.ravel()) / roaster.pix_noise_var[epoch_num])

# ------------------------------------------------------------------------------
class ConfigFileParser(object):
    """
    Parse a configuration file for this script 
    """
    def __init__(self, config_file_name):
        self.config_file = config_file_name

        config = jifconf.DefConfigParser()
        config.read(config_file_name)

        infiles = config.items("infiles")
        print "infiles:"
        for key, infile in infiles:
            print infile
        self.infiles = [infile for key, infile in infiles]

        segment_number = config.get("data", "segment_number")
        self.segment_number = int(segment_number)

        self.outfile = config.get("metadata", "outfile",
                                  default="../output/roasting/roaster_out")

        self.galaxy_model_type = config.get("model", "galaxy_model_type", 
                                            default="Spergel")

        model_params = config.get("model", "model_params", default='e beta')
        self.model_params = str.split(model_params, ' ')

        self.num_sources = int(config.get("model", "num_sources", default=1))

        self.telescope = config.get("data", "telescope")

        self.data_format = config.get("data", "data_format")

        filters = config.get("data", "filters")
        if filters is not None:
            self.filters = str.split(filters, ' ')

        achromatic = config.get("model", "achromatic")
        self.achromatic = achromatic == 'True'

        epoch_num = config.get("data", "epoch_num", default=-1)
        self.epoch_num = int(epoch_num)

        self.init_param_file = config.get("init", "init_param_file")

        output_model = config.get("run", "output_model", default="False")
        self.output_model = (output_model == "True")

        self.seed = config.get("init", "seed")
        if self.seed is not None:
            self.seed = int(self.seed)

        self.sampler = config.get("sampling", "sampler", default='emcee')

        self.nsamples = int(config.get("sampling", "nsamples", default=100))
        self.nwalkers = int(config.get("sampling", "nwalkers", default=32))
        self.nburn = int(config.get("sampling", "nburn", default=50))
        self.nthreads = int(config.get("sampling", "nthreads", default=1))

        quiet = config.get("run", "quiet")
        self.quiet = (quiet == "True")
        debug = config.get("run", "debug")
        self.debug = (debug == "True")

        return None


# ------------------------------------------------------------------------------
def InitRoaster(args):
    args.outfile += '_seg{:d}'.format(args.segment_number)

    np.random.seed(args.seed)

    logging.debug('--- Roaster started')

    if args.segment_number is None:
        args.segment_number = 0

    if args.epoch_num >= 0:
        epoch_num = args.epoch_num
    else:
        epoch_num = None

    ### Set galaxy priors
    if args.galaxy_model_type == "Spergel":
        # lnprior_omega = DefaultPriorSpergel()
        lnprior_omega = EmptyPrior()
    elif args.galaxy_model_type == "BulgeDisk":
        lnprior_omega = DefaultPriorBulgeDisk(z_mean=1.0)
    else:
        lnprior_omega = EmptyPrior()

    ### Set PSF priors
    lnprior_Pi = pm.DefaultPriorPSF()

    roaster = Roaster(debug=args.debug, data_format=args.data_format,
                      lnprior_omega=lnprior_omega,
                      lnprior_Pi=lnprior_Pi,
                      galaxy_model_type=args.galaxy_model_type,
                      model_paramnames=args.model_params,
                      telescope=args.telescope,
                      filters_to_load=args.filters,
                      achromatic_galaxy=args.achromatic)
    roaster.Load(args.infiles[0], segment=args.segment_number,
                 epoch_num=epoch_num, use_PSFModel=False)
    if args.init_param_file is not None:
        roaster.initialize_param_values(args.init_param_file)
    return roaster, args


# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Draw interim samples of source model parameters via MCMC.')

    parser.add_argument("--infiles", type=str,
                        help="input image files to roast", nargs='+')

    parser.add_argument('--config_file', type=str, default=None,
                        help="Name of a configuration file listing inputs." +
                             "If specified, ignore other command line flags." +
                             "(Default: None)")

    parser.add_argument("--segment_number", type=int, default=None,
                        help="Index of the segment to load")

    parser.add_argument("-o", "--outfile",
                        default="../output/roasting/roaster_out",
                        help="output HDF5 to record posterior samples and "+
                             "loglikes. (Default: `roaster_out`)")

    parser.add_argument("--galaxy_model_type", type=str, default="Spergel",
                        help="Type of parametric galaxy model "+
                             "(Default: 'Spergel')")

    parser.add_argument("--data_format", type=str, default="jif_segment",
                        help="Format of the input image data file " +
                             "(Default: 'jif_segment')")

    parser.add_argument("--model_params", type=str, nargs='+',
                        default=['nu', 'hlr', 'e', 'beta', 'mag_sed1', 'dx', 
                                 'dy', 'psf_fwhm'],
                        help="Names of the galaxy model parameters for sampling.")

    parser.add_argument("--telescope", type=str, default=None,
                        help="Select only a single telescope from the input data file " + 
                             "(Default: None - get all telescopes data)")

    parser.add_argument("--filters", type=str, nargs='+',
                        help="Names of a subset of filters to load from the input data file " + 
                             "(Default: None - use data in all available filters)")

    parser.add_argument("--achromatic", action="store_true",
                        help="Use an achromatic galaxy or star model")

    parser.add_argument("--epoch_num", type=int, default=-1,
                        help="Index of single epoch to fit. If not supplied, "+
                             "then fit all epochs.")

    parser.add_argument("--init_param_file", type=str, default=None,
                        help="Name of a config file with parameter values to "+
                             "initialize the chain")

    parser.add_argument("--output_model", action="store_true",
                        help="Just save the initial model image to file. "+
                             "Don't sample")

    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for pseudo-random number generator")

    parser.add_argument("--sampler", type=str, default='emcee',
                        help="Type of MC sampler to use (Default: 'emcee')")

    parser.add_argument("--nsamples", default=100, type=int,
                        help="Number of samples for each emcee walker "+
                             "(Default: 100)")

    parser.add_argument("--nwalkers", default=32, type=int,
                        help="Number of emcee walkers (Default: 16)")

    parser.add_argument("--nburn", default=50, type=int,
                        help="Number of burn-in steps (Default: 50)")

    parser.add_argument("--nthreads", default=1, type=int,
                        help="Number of threads to use (Default: 1)")

    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    ### Parse some parameters that can override those in the config_file, if present
    segment_number = args.segment_number
    outfile = args.outfile

    ###
    ### Get the parameters for input/output from configuration file or argument list
    ###
    if isinstance(args.config_file, str):
        logging.info('Reading from configuration file {}'.format(args.config_file))
        args = ConfigFileParser(args.config_file)
        if segment_number is not None:
            args.segment_number = segment_number
    elif not isinstance(args.infiles, list):
        raise ValueError("Must specify either 'config_file' or 'infiles' argument")

    roaster, args = InitRoaster(args)

    import pprint
    pp = pprint.PrettyPrinter(indent=4)

    if not args.quiet:
        print("\nsource model 0:")
        pp.pprint(roaster.src_models[0][0].__dict__)

    if args.output_model:
        resid_rms = save_model_image(args, roaster)
        if resid_rms > 1:
            print "\n======== WARNING: large model residual ==========="
        print "Residual r.m.s. for initial model: {:12.10g}\n".format(resid_rms)
    else:
        do_sampling(args, roaster, sampler=args.sampler)

    logging.debug('--- Roaster finished')
    return 0


if __name__ == "__main__":
    sys.exit(main())
