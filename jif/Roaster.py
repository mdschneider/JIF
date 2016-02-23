#!/usr/bin/env python
# encoding: utf-8
"""
Roaster.py

Draw samples of source model parameters given the pixel data for image segments.

This script is intended to run on one segment at a time (i.e., call multiple
instances of the script to process multiple segments).
"""

import argparse
import sys
import os.path
import copy
import numpy as np
import h5py
import emcee
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
                              ['Sersic', 'Spergel', 'BulgeDisk' (default)]
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
    """
    def __init__(self, lnprior_omega=None,
                 lnprior_Pi=None,
                 data_format='test_galsim_galaxy',
                 galaxy_model_type='BulgeDisk',
                 telescope=None,
                 filters_to_load=None,
                 debug=False,
                 model_paramnames=['hlr', 'e', 'beta']):
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
        ### Check if any of the active parameters are for a PSF model.
        ### If so, we will sample in the PSF and need to setup accordingly
        ### when the Load() function is called.
        self.sample_psf = False
        self.psf_model_paramnames = []
        if np.any(['psf' in p for p in model_paramnames]):
            self.sample_psf = True
            self.psf_model_paramnames = galsim_galaxy.select_psf_paramnames(
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

    def Load(self, infile, segment=None, use_PSFModel=False):
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

        @param infiles  List of input filenames to load.
        @param segment  Index of the segment to load. Choose segment 0 if not
        supplied.
        @param use_PSFModel Force the use of a PSFModel class instance to model
        PSFs even if not sampling any PSF model parameters
        """
        # global self.pixel_data
        # global self.pix_noise_var
        # global self.src_models

        ### Reset the global data lists in case Roaster had been used before
        self.pixel_data = []
        self.pix_noise_var = []
        self.src_models = []

        logging.info("<Roaster> Loading image data")
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
            self.num_sources = f['segments/seg{:d}'.format(segment)].attrs['num_sources']

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
                g = 'segments/seg{:d}/{}'.format(segment, tel.lower())
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
                        bp = galsim_galaxy.load_filter_file_to_bandpass(table,
                            effective_diameter_meters=galsim_galaxy.k_telescopes[tel]['effective_diameter'],
                            exptime_sec=galsim_galaxy.k_telescopes[tel]['exptime_zeropoint'])
                        self.filters[filter_name] = bp
                        have_bandpasses = True
                    except KeyError:
                        have_bandpasses = False

                    h = 'segments/seg{:d}/{}/{}'.format(segment, tel.lower(), filter_name)
                    nepochs = len(f[h])
                    if self.debug:
                        print("Number of epochs for {}: {:d}".format(tel, nepochs))
                    for iepoch in xrange(nepochs):
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
                                gsparams=None))
                        elif use_PSFModel:
                            psf_types.append('PSFModel class')
                            psfs.append(pm.PSFModel(
                                active_parameters=[],
                                gsparams=None))
                        else:
                            psf_types.append(seg.attrs['psf_type'])
                            psfs.append(seg['psf'])
                        ###
                        tel_group = f['telescopes/{}'.format(tel)]
                        pixel_scales.append(tel_group.attrs['pixel_scale_arcsec'])
                        primary_diams.append(tel_group.attrs['primary_diam'])
                        atmospheres.append(tel_group.attrs['atmosphere'])
                        tel_names.append(tel)
                        self.filter_names.append(filter_name)
            print "Have data for instruments:", instruments
            print "pixel noise variances:", self.pix_noise_var
        else:
            raise KeyError("Unsupported input data format in Roaster")
            # if segment == None:
            #     logging.info("<Roaster> Must specify a segment number as an integer")
            # print "Have data for instruments:", instruments

        if have_bandpasses:
            print "Filters:", self.filters.keys()
        else:
            self.filters = None

        nimages = len(self.pixel_data)
        self.num_epochs = nimages
        self.nx = np.zeros(nimages, dtype=int)
        self.ny = np.zeros(nimages, dtype=int)
        for i in xrange(nimages):
            self.nx[i], self.ny[i] = self.pixel_data[i].shape
            if self.debug:
                print "nx, ny, i:", self.nx[i], self.ny[i], i
                print "tel_name:", tel_names[i]
                print "pixel_scale (arcsec):", pixel_scales[i]
                print "wavelength (nm):", wavelengths[i]
                print "primary_diam (m):", primary_diams[i]
                print "atmosphere:", atmospheres[i]

        logging.debug("<Roaster> num_epochs: {:d}, num_sources: {:d}".format(
            self.num_epochs, self.num_sources))
        self.src_models = [[galsim_galaxy.GalSimGalaxyModel(
                                telescope_name=tel_names[idat],
                                galaxy_model=self.galaxy_model_type,
                                active_parameters=self.model_paramnames,
                                pixel_scale_arcsec=pixel_scales[idat],
                                primary_diam_meters=primary_diams[idat],
                                filters=self.filters,
                                filter_names=self.filter_names[idat],
                                atmosphere=atmospheres[idat],
                                psf_model=psfs[idat])
                            for idat in xrange(nimages)]
                           for isrcs in xrange(self.num_sources)]
        self.n_params = self.src_models[0][0].n_params
        logging.debug("<Roaster> Finished loading data")
        if self.debug:
            print "\npixel data shapes:", [dat.shape for dat in self.pixel_data]
        return None

    def _get_noise_var(self, i=0):
        return self.pix_noise_var[i]

    def _get_raw_params(self):
        return self.src_models[0][0].params

    def get_params(self):
        """
        Make a flat array of active model parameters for all sources

        For use in MCMC sampling.
        """
        p = np.array([m[0].get_params() for m in self.src_models]).ravel()
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

            p_set = self.src_models[isrcs][0].get_params()
            if self.debug:
                print "input p:", p
                print "p_set before indexing:", p_set
            p_set = p[imin:imax]
            # p_set = p[isrcs]
            if self.debug:
                print "p_set after indexing:", p_set

            for iepochs in xrange(self.num_epochs):
                self.src_models[isrcs][iepochs].set_params(p_set)
                valid_params *= self.src_models[isrcs][iepochs].validate_params()
        return valid_params

    def set_param_by_name(self, paramname, value):
        """
        Set a galaxy or PSF model parameter by name.

        Can pass a single value that will be set for all source models, or a
        list of length num_sources with unique values for each source (but
        common across all epochs).
        """
        if isinstance(value, list):
            if len(value) == self.num_sources:
                for isrcs in xrange(self.num_sources):
                    for idat in xrange(self.num_epochs):
                        self.src_models[isrcs][idat].set_param_by_name(paramname, value[isrcs])
            else:
                raise ValueError("If passing list, must be of length num_sources")
        elif isinstance(value, float):
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
                imin = isrcs * self.n_params
                imax = (isrcs + 1) * self.n_params
                ### Pass active + inactive parameters, with names included
                p = copy.deepcopy(self.src_models[isrcs][0].params)
                lnp += self.lnprior_omega(p)
                if self.sample_psf:
                    ppsf = copy.deepcopy(self.src_models[isrcs][0].psf_model.params)
                    lnp += self.lnprior_Pi(ppsf)
        else:
            lnp = -np.inf
        return lnp

    def _get_model_image(self, iepochs, add_noise=False):
        """
        Create a galsim.Image from the source model(s)
        """
        model_image = galsim.ImageF(self.nx[iepochs], self.ny[iepochs],
            scale=self.src_models[0][iepochs].pixel_scale)

        for isrcs in xrange(self.num_sources):
            ### Draw every source using the full output array
            b = galsim.BoundsI(1, self.nx[iepochs], 1, self.ny[iepochs])
            sub_image = model_image[b]
            model = self.src_models[isrcs][iepochs].get_image(sub_image,
                filter_name=self.filter_names[iepochs], add_noise=add_noise)
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
                        model_image_file_name = os.path.join('debug',
                            'model_image_iepoch%d_istep%d.fits' % (iepochs,
                                self.istep))
                        model_image.write(model_image_file_name)
                        logging.debug('Wrote model image to %r',
                            model_image_file_name)

                    lnlike += (-0.5 * np.sum((self.pixel_data[iepochs] -
                        model_image.array) ** 2) /
                        self.pix_noise_var[iepochs])
        else:
            lnlike = -np.inf
        return lnlike

    def __call__(self, omega, *args, **kwargs):
        return self.lnlike(omega, *args, **kwargs) + self.lnprior(omega)

# ---------------------------------------------------------------------------------------
# MCMC routines
# ---------------------------------------------------------------------------------------
def walker_ball(omega, spread, nwalkers):
    return [omega+(np.random.rand(len(omega))*spread-0.5*spread)
            for i in xrange(nwalkers)]


def do_sampling(args, roaster, return_samples=False):
    """
    Execute the MCMC chain to fit galaxy (and PSF) model parameters to a segment

    Save the MCMC chain steps to an HDF5 file, and optionally also return the
    steps if `return_samples` is True.
    """
    omega_interim = roaster.get_params()
    logging.info("Have {:d} sampling parameters".format(len(omega_interim)))

    nvars = len(omega_interim)
    p0 = walker_ball(omega_interim, 0.02, args.nwalkers)

    # logging.debug("Initializing parameters for MCMC to yield finite posterior values")
    # while not all([np.isfinite(roaster(p)) for p in p0]):
    #     p0 = walker_ball(omega_interim, 0.02, args.nwalkers)
    sampler = emcee.EnsembleSampler(args.nwalkers,
                                    nvars,
                                    roaster,
                                    threads=args.nthreads)
    nburn = max([1,args.nburn])
    logging.info("Burning")
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

    ### Save attributes so we can later instantiate Roaster with the same
    ### settings.
    f.attrs['infile'] = args.infiles[0]
    f.attrs['segment_number'] = args.segment_numbers[0]
    f.attrs['galaxy_model_type'] = roaster.galaxy_model_type
    if roaster.telescope is not None:
        f.attrs['telescope'] = roaster.telescope
    else:
        f.attrs['telescope'] = 'None'

    f.attrs['model_paramnames'] = roaster.model_paramnames

    if roaster.num_sources == 1:
        paramnames = roaster.model_paramnames
    elif roaster.num_sources > 1:
        paramnames = [[p + '_src{:d}'.format(isrc) for p in roaster.model_paramnames]
                      for isrc in xrange(roaster.num_sources)]

    ### Write the MCMC samples and log probabilities
    if "post" in f:
        del f["post"]
    post = f.create_dataset("post", data=np.transpose(np.dstack(pps), [2,0,1]))
    # pnames = np.array(roaster.src_models[0][0].paramnames)
    post.attrs['paramnames'] = paramnames
    if "logprobs" in f:
        del f["logprobs"]
    logprobs = f.create_dataset("logprobs", data=np.vstack(lnps))
    f.attrs["nburn"] = args.nburn
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

# ---------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Draw interim samples of source model parameters via MCMC.')

    parser.add_argument("infiles",
                        help="input image files to roast", nargs='+')

    parser.add_argument("--segment_numbers", type=int, nargs='+',
                        help="Index of the segments to load from each infile")

    parser.add_argument("-o", "--outfile", default="../output/roasting/roaster_out",
                        help="output HDF5 to record posterior samples and loglikes."
                             +"(Default: `roaster_out`)")

    parser.add_argument("--galaxy_model_type", type=str, default="Spergel",
                        help="Type of parametric galaxy model (Default: 'Spergel')")

    parser.add_argument("--data_format", type=str, default="jif_segment",
                        help="Format of the input image data file (Default: \
                             'jif_segment')")

    parser.add_argument("--model_params", type=str, nargs='+',
                        default=['nu', 'hlr', 'e', 'beta', 'mag_sed1', 'dx', 'dy', 'psf_fwhm'],
                        help="Names of the galaxy model parameters for sampling.")

    parser.add_argument("--telescope", type=str, default=None,
                        help="Select only a single telescope from the input data file \
                        (Default: None - get all telescopes data)")

    parser.add_argument("--filters", type=str, nargs='+',
                        help="Names of a subset of filters to load from the input data file \
                        (Default: None - use data in all available filters)")

    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for pseudo-random number generator")

    parser.add_argument("--nsamples", default=100, type=int,
                        help="Number of samples for each emcee walker \
                              (Default: 100)")

    parser.add_argument("--nwalkers", default=16, type=int,
                        help="Number of emcee walkers (Default: 16)")

    parser.add_argument("--nburn", default=50, type=int,
                        help="Number of burn-in steps (Default: 50)")

    parser.add_argument("--nthreads", default=1, type=int,
                        help="Number of threads to use (Default: 1)")

    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    np.random.seed(args.seed)

    logging.debug('--- Roaster started')

    if args.segment_numbers is None:
        args.segment_numbers = [0 for f in args.infiles]

    ### Set galaxy priors
    if args.galaxy_model_type == "Spergel":
        lnprior_omega = DefaultPriorSpergel()
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
                      filters_to_load=args.filters)
    roaster.Load(args.infiles[0], segment=args.segment_numbers[0])

    import pprint
    pp = pprint.PrettyPrinter(indent=4)

    if not args.quiet:
        print("\nsource model 0:")
        pp.pprint(roaster.src_models[0][0].__dict__)

    do_sampling(args, roaster)

    logging.debug('--- Roaster finished')
    return 0


if __name__ == "__main__":
    sys.exit(main())
