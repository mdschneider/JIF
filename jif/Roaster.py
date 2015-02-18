#!/usr/bin/env python
# encoding: utf-8
"""
Roaster.py

Draw samples of source model parameters given the pixel data for image cutouts.
"""

import argparse
import sys
import os.path
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


class EmptyPrior(object):
    def __init__(self):
        pass
    def __call__(self, *args):
        return 0.0


class Roaster(object):
    """
    Draw samples of source model parameters via MCMC.
    """
    def __init__(self, pix_noise_var, src_model=None, lnprior_omega=None):
        self.pix_noise_var = pix_noise_var
        self.src_model = src_model
        if lnprior_omega is None:
            self.lnprior_omega = EmptyPrior()
        else:
            self.lnprior_omega = lnprior_omega
    
    def Load(self, infiles):
        """
        Load image cutouts from files.

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

        @param infiles  List of input filenames to load.
        """
        ### TODO: ingest test image file produced by galsim_galaxy script
        ### TODO: set self.nx, self.ny from input data pixel grid dimensions
        raise NotImplementedError()

    def lnprior(self, omega):
        return NotImplementedError()

    def lnlike(self, omega, *args, **kwargs):
        """
        Evaluate the log-likelihood function for joint pixel data for all 
        galaxies in a blended group given all available imaging and epochs.
        """
        self.src_model.set_params(omega)
        out_image = galsim.Image(self.nx, self.ny)
        model = self.src_model.get_image(out_image).array
        return -0.5 * np.sum((self.data - model) ** 2) / self.pix_noise_var

    def __call__(self, omega, *args, **kwargs):
        return self.lnlike(omega, *args, **kwargs) + self.lnprior(omega)

# ---------------------------------------------------------------------------------------
# MCMC routines
# ---------------------------------------------------------------------------------------
def walker_ball(omega, spread, nwalkers):
    return [omega+(np.random.rand(len(omega))*spread-0.5*spread) for i in xrange(nwalkers)]


def do_sampling(args, roaster):
    omega_interim = roaster.src_model.get_params()

    nvars = len(omega_interim)
    p0 = walker_ball(omega_interim, 0.02, args.nwalkers)

    while not all([np.isfinite(roaster(p)) for p in p0]):
        p0 = walker_ball(omega_interim, 0.02, args.nwalkers)
    sampler = emcee.EnsembleSampler(args.nwalkers,
                                    nvars,
                                    roaster,
                                    threads=args.nthreads)
    nburn = max([1,args.nburn])
    print "Burning"
    pp, lnp, rstate = sampler.run_mcmc(p0, nburn)
    sampler.reset()
    pps = []
    lnps = []
    print "Sampling"
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
    print "<Reaper> Writing MCMC results to %s" % args.outfile
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
                             +"(Default: `out.h5`)")
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

    args = parser.parse_args()
    np.random.seed(args.seed)

    logging.debug('--- Roaster started')

    noise_model = galsim_galaxy.wfirst_noise(-1)
    pix_noise_var = noise_model.getVariance()

    roaster = Roaster(pix_noise_var=pix_noise_var, 
        src_model=galsim_galaxy.GalSimGalaxyModel(noise=noise_model))
    roaster.Load(args.infiles)

    do_sampling(args, roaster)

    logging.debug('--- Roaster finished')
    return 0


if __name__ == "__main__":
    sys.exit(main())
