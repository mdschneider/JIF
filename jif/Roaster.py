#!/usr/bin/env python
# encoding: utf-8
"""
Roaster.py

Draw samples of source model parameters given the pixel data for image cutouts.
April 1, 2015
"""

import argparse
import sys
import os.path
import copy
import numpy as np
import h5py
import emcee
import galsim_galaxy
import galsim

import logging


# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
#logging.basicConfig(filename='logs/Roaster.log',
#                     level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


# Store the pixel data as global (module scope) variables so emcee mutlithreading doesn't need to
# repeatedly pickle these all the time.
pixel_data = []
pix_noise_var = []
src_models = []


class EmptyPrior(object):
    def __init__(self):
        pass
    def __call__(self, *args):
        return 0.0


class Roaster(object):
    """
    Draw samples of source model parameters via MCMC.

    We allow for 2 types of multiplicity in the data:
        (a) Multiple epochs or instruments observing the same source
        (b) Multiple sources in a single 'cutout' (e.g., for blended sources)
    For scenario (a), the likelihood is a product of the likelihoods for each epoch or instrument.
    For scenario (b), we need a list of source models that generate a model image that is 
    fed to the likelihood function for a single epoch or instrument.

    @lnprior_omega      Prior class for the galaxy model parameters
    @param data_format  Format for the input data file.
    @param debug        Save debugging outputs (including model images per step?)
    """
    def __init__(self, lnprior_omega=None, 
                 data_format='test_galsim_galaxy', debug=False):
        if lnprior_omega is None:
            self.lnprior_omega = EmptyPrior()
        else:
            self.lnprior_omega = lnprior_omega
        self.data_format = data_format
        self.debug = debug

        ### Count the number of calls to self.lnlike
        self.istep = 0
    
    def Load(self, infile):
        """
        Load image cutouts from file.

        The input file should contain: 
            1. Pixel data for a cutout of N galaxies
            2. A segmentation mask for the cutout
            3. The pixel noise model (e.g., variance per pixel)
            4. The WCS information for the image
            5. Background model(s)
        This information should be replicated for each epoch and/or instrument.

        Each source in a blend group should contain information on:
            1. a, b, theta (ellipticity semi-major and semi-minor axes, orientation angle)
            2. centroid position (x,y)
            3. flux
        These values will be used to initialize any model-fitting (e.g., MCMC) algorithm.

        @param infiles  List of input filenames to load.
        """
        global pixel_data
        global pix_noise_var
        global src_models

        logging.info("<Roaster> Loading image data")
        if self.data_format == "test_galsim_galaxy":
            f = h5py.File(infile, 'r')
            self.num_epochs = len(f) ### FIXME: What's the right HDF5 method to get num groups?
            self.num_sources = f.attrs['num_sources']

            instruments = []
            pixel_scales = []
            wavelengths = []
            primary_diams = []
            atmospheres = []
            for i in xrange(self.num_epochs):
                cutout = f['cutout%d' % (i+1)]
                dat = cutout['pixel_data']
                print i, "dat shape:", dat.shape
                print "\t", np.array(dat).shape
                pixel_data.append(np.array(dat))
                # pixel_data.append(np.core.records.array(np.array(dat), dtype=float, shape=dat.shape))
                # pixel_data.append(np.array(cutout['pixel_data']))
                pix_noise_var.append(cutout['noise_model'])
                instruments.append(cutout.attrs['instrument'])
                pixel_scales.append(cutout.attrs['pixel_scale'])
                wavelengths.append(cutout.attrs['wavelength'])
                primary_diams.append(cutout.attrs['primary_diam'])
                atmospheres.append(cutout.attrs['atmosphere'])
            print "Have data for instruments:", instruments
        else:
            raise AttributeError("Unsupported input data format to Roaster")

        self.nx = np.zeros(self.num_epochs, dtype=int)
        self.ny = np.zeros(self.num_epochs, dtype=int)
        for i in xrange(self.num_epochs):
            dat = f['cutout%d'%(i+1)]['pixel_data']
            self.nx[i], self.ny[i] = dat.shape

        src_models = [[galsim_galaxy.GalSimGalaxyModel(galaxy_model="Sersic",
                                pixel_scale=pixel_scales[iepochs], 
                                wavelength=wavelengths[iepochs],
                                primary_diam_meters=primary_diams[iepochs],
                                atmosphere=atmospheres[iepochs]) 
                            for iepochs in xrange(self.num_epochs)] 
                           for isrcs in xrange(self.num_sources)]

        logging.debug("<Roaster> Finished loading data")
        print "\npixel data shapes:", [dat.shape for dat in pixel_data]
        return None

    def get_params(self):
        """
        Make a flat array of model parameters for all sources
        """
        return np.array([m[0].get_params() for m in src_models]).ravel()

    def set_params(self, p):
        for isrcs in xrange(self.num_sources):
            imin = isrcs * self.num_sources
            imax = (isrcs + 1) * self.num_sources
            for iepochs in xrange(self.num_epochs):
                src_models[isrcs][iepochs].set_params(p[imin:imax])
        return None

    def lnprior(self, omega):
        return self.lnprior_omega(omega)

    def lnlike(self, omega, *args, **kwargs):
        """
        Evaluate the log-likelihood function for joint pixel data for all 
        galaxies in a blended group given all available imaging and epochs.

        See GalSim/examples/demo5.py for how to add multiple sources to a single image.
        """
        self.istep += 1
        self.set_params(omega)

        lnlike = 0.0
        for iepochs in xrange(self.num_epochs):
            model_image = galsim.ImageF(self.nx[iepochs], self.ny[iepochs], 
                scale=src_models[0][iepochs].pixel_scale)

            for isrcs in xrange(self.num_sources):
                ### Draw every source using the full output array
                b = galsim.BoundsI(1, self.nx[iepochs], 1, self.ny[iepochs])
                sub_image = model_image[b]
                model = src_models[isrcs][iepochs].get_image(sub_image)

            if self.debug:
                model_image_file_name = os.path.join('debug', 
                    'model_image_iepoch%d_istep%d.fits' % (iepochs, self.istep))
                model_image.write(model_image_file_name)
                logging.debug('Wrote model image to %r', model_image_file_name)

            lnlike += (-0.5 * np.sum((pixel_data[iepochs] - model_image.array) ** 2) / 
                pix_noise_var[iepochs])
        return lnlike

    def __call__(self, omega, *args, **kwargs):
        return self.lnlike(omega, *args, **kwargs) + self.lnprior(omega)

# ---------------------------------------------------------------------------------------
# MCMC routines
# ---------------------------------------------------------------------------------------
def walker_ball(omega, spread, nwalkers):
    return [omega+(np.random.rand(len(omega))*spread-0.5*spread) for i in xrange(nwalkers)]


def do_sampling(args, roaster):
    omega_interim = roaster.get_params()
    print "omega_interim:", omega_interim

    nvars = len(omega_interim)
    p0 = walker_ball(omega_interim, 0.02, args.nwalkers)

    logging.debug("Initializing parameters for MCMC to yield finite posterior values")
    while not all([np.isfinite(roaster(p)) for p in p0]):
        p0 = walker_ball(omega_interim, 0.02, args.nwalkers)
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
    logging.info("Sampling")
    for i in range(args.nsamples):
        pp, lnp, rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
        if not args.quiet:
            print i, np.mean(lnp)
            print np.mean(pp, axis=0)
            print np.std(pp, axis=0)
        pps.append(pp.copy())
        lnps.append(lnp.copy())

    write_results(args, pps, lnps)
    return None


def write_results(args, pps, lnps):
    logging.info("Writing MCMC results to %s" % args.outfile)
    f = h5py.File(args.outfile, 'w')
    if "post" in f:
        del f["post"]
    post = f.create_dataset("post", data=np.transpose(np.dstack(pps), [2,0,1]))
    paramnames = ['x0', 'y0', 'phi0', 'ell0', 'L', 'psfwidth']
    post.attrs['paramnames'] = paramnames
    if "logprobs" in f:
        del f["logprobs"]
    logprobs = f.create_dataset("logprobs", data=np.vstack(lnps))
    f.attrs["nburn"] = args.nburn
    f.close()
    return None

# ---------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Draw interim samples of source model parameters via MCMC.')
    parser.add_argument("infiles",
                        help="input image files to roast", nargs='+')    
    parser.add_argument("-o", "--outfile", default="../output/roasting/roaster_out.h5",
                        help="output HDF5 to record posterior samples and loglikes."
                             +"(Default: `roaster_out.h5`)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for pseudo-random number generator")
    parser.add_argument("--nsamples", default=250, type=int,
                        help="Number of samples for each emcee walker (Default: 250)")
    parser.add_argument("--nwalkers", default=64, type=int,
                        help="Number of emcee walkers (Default: 64)")
    parser.add_argument("--nburn", default=50, type=int,
                        help="Number of burn-in steps (Default: 50)")
    parser.add_argument("--nthreads", default=1, type=int,
                        help="Number of threads to use (Default: 8)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    np.random.seed(args.seed)

    logging.debug('--- Roaster started')

    roaster = Roaster(debug=args.debug)
    roaster.Load(args.infiles[0])

    print "\nRoaster:", roaster.__dict__, "\n"

    print "\nsource models:",src_models[0][0].__dict__, "\n"

    do_sampling(args, roaster)

    logging.debug('--- Roaster finished')
    return 0


if __name__ == "__main__":
    sys.exit(main())
