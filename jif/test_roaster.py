#!/usr/bin/env python
# encoding: utf-8
"""
test_roaster.py

A minimal implementation of Roaster to test mutlithreading.
February 24, 2015
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


class EmptyPrior(object):
    def __init__(self):
        pass
    def __call__(self, *args):
        return 0.0


class toymodel(object):
    """toymodel for an image"""
    def __init__(self, nx, ny, imin, imax):
        self.nx = nx
        self.ny = ny
        self.imin = imin
        self.imax = imax

        self.model_image = np.zeros((nx, ny), dtype=float)

    def set_params(self, p):
        self.model_image[self.imin:self.imax, self.imin:self.imax] = p
        return None

    def get_image(self):
        return self.model_image
        

class Roaster(object):
    """
    Draw samples of source model parameters via MCMC.

    We allow for 2 types of multiplicity in the data:
        (a) Multiple epochs or instruments observing the same source
        (b) Multiple sources in a single 'cutout' (e.g., for blended sources)
    For scenario (a), the likelihood is a product of the likelihoods for each epoch or instrument.
    For scenario (b), we need a list of source models that generate a model image that is 
    fed to the likelihood function for a single epoch or instrument.

    @param debug        Save debugging outputs (including model images per step?)
    """
    def __init__(self, debug=False):
        self.debug = debug

        self.num_epochs = 1
        self.npix = 64
        self.params = np.zeros(self.num_epochs, dtype=float)

    def Load(self, infile=None):
        global pixel_data

        f = h5py.File(infile, 'r')
        i = 0
        cutout = f['cutout%d' % (i+1)]
        dat = cutout['pixel_data']
        pixel_data.append(np.array(dat))

        # pixel_data.append(np.zeros((self.npix,self.npix), dtype=float))
        # pixel_data[0][12:18, 12:18] = 1.0
        self.nx = [self.npix]
        self.ny = [self.npix]

        self.src_models = [toymodel(self.nx[i], self.ny[i], 12, 18) 
                           for i in xrange(self.num_epochs)]
        return None

    def get_params(self):
        return self.params

    def set_params(self, p):
        for iepochs in xrange(self.num_epochs):
            self.params[iepochs] = p[iepochs]
            self.src_models[iepochs].set_params(p[iepochs])

    def lnlike(self, omega, *args, **kwargs):
        self.set_params(omega)

        lnlike = 0.0
        for iepochs in xrange(self.num_epochs):
            model_image = self.src_models[iepochs].get_image()
            lnlike += (-0.5 * np.sum((pixel_data[iepochs] - model_image) ** 2))
        return lnlike

    def __call__(self, omega, *args, **kwargs):
        return self.lnlike(omega, *args, **kwargs)


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
    parser.add_argument("-o", "--outfile", default="../output/roasting/test_roaster_out.h5",
                        help="output HDF5 to record posterior samples and loglikes."
                             +"(Default: `roaster_out.h5`)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for pseudo-random number generator")
    parser.add_argument("--nsamples", default=1, type=int,
                        help="Number of samples for each emcee walker (Default: 250)")
    parser.add_argument("--nwalkers", default=8, type=int,
                        help="Number of emcee walkers (Default: 64)")
    parser.add_argument("--nburn", default=1, type=int,
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

    do_sampling(args, roaster)

    logging.debug('--- Roaster finished')
    return 0


if __name__ == "__main__":
    sys.exit(main())
