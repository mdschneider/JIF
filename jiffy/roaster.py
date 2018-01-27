#!/usr/bin/env python
# encoding: utf-8
# 
# Copyright (c) 2017, Lawrence Livermore National Security, LLC. 
# Produced at the Lawrence Livermore National Laboratory. Written by 
# Michael D. Schneider schneider42@llnl.gov. 
# LLNL-CODE-742321. All rights reserved. 
# 
# This file is part of JIF. For details, see https://github.com/mdschneider/JIF 
# 
# Please also read this link – Our Notice and GNU Lesser General Public License
# https://github.com/mdschneider/JIF/blob/master/LICENSE 
# 
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License (as published by the Free Software
# Foundation) version 2.1 dated February 1999. 
# 
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the terms and conditions of the GNU General
# Public License for more details. 
# 
# You should have received a copy of the GNU Lesser General Public License along
# with this program; if not, write to the Free Software Foundation, Inc., 59 
# Temple Place, Suite 330, Boston, MA 02111-1307 USA 
"""
@file jiffy roaster.py

Draw posterior samples of image source model parameters given the
likelihood functxion of an image footprint
"""
import numpy as np
import galsim
import galsim_galaxy
import galsim_psf

class Roaster(object):
    """
    Likelihood model for footprint pixel data given a parametric source model

    Only single epoch images are allowed.
    """
    def __init__(self, config="../config/jiffy.yaml"):
        if isinstance(config, str):
            import yaml
            config = yaml.load(open(config))
        self.config = config

        np.random.seed(self.config["init"]["seed"])

        self.num_sources = self.config['model']['num_sources']
        actv_params = self.config["model"]["model_params"].split(" ")

        model_class_name = self.config["model"]["model_class"]
        args = {"active_parameters": actv_params}
        if model_class_name is "GalsimGalaxyModel":
            args["psf_model_class_name"] = self.config["model"]["psf_class"]

        model_modules = __import__('galsim_galaxy', 'galsim_psf')
        self.src_models = [getattr(model_modules, model_class_name)(**args)
                           for i in xrange(self.num_sources)]

        self.n_params = len(actv_params)

        # Initialize objects describing the pixel data in a footprint
        self.ngrid_x = 64
        self.ngrid_y = 64
        self.noise_var = 3e-10
        self.scale = 0.2
        self.gain = 1.0
        self.data = None

    def get_params(self):
        """
        Make a flat array of active model parameters for all sources

        For use in MCMC sampling
        """
        return np.array([m.get_params() for m in self.src_models]).ravel()

    def set_params(self, params):
        """
        Set the active parameters for all sources from a flattened array
        """
        valid_params = True
        for isrc in xrange(self.num_sources):
            imin = isrc * self.n_params
            imax = (isrc + 1) * self.n_params
            p_set = params[imin:imax]
            valid_params *= self.src_models[isrc].set_params(p_set)
        return valid_params

    def set_param_by_name(self, paramname, value):
        """
        Set a galaxy or PSF model parameter by name

        Can pass a single value that will be set for all source models, or a
        list of length num_sources with unique values for each source.
        """
        if isinstance(value, list):
            if len(value) == self.num_sources:
                for isrc, v in enumerate(value):
                    self.src_models[isrc].set_param_by_name(paramname, v)
            else:
                raise ValueError("If passing list, must have length num_sources")
        elif isinstance(value, float):
            for isrc in xrange(self.num_sources):
                self.src_models[isrc].set_param_by_name(paramname, value)
        else:
            raise ValueError("Unsupported type for input value")
        return None

    def make_data(self):
        """
        Make fake data from the current stored galaxy model
        """
        image = self._get_model_image()
        noise = galsim.GaussianNoise(sigma=np.sqrt(self.noise_var))
        image.addNoise(noise)
        self.data = image.array
        return None

    def import_data(self, pix_dat_array, noise_var, scale=0.2, gain=1.0):
        """
        Import the pixel data and noise variance for a footprint
        """
        self.ngrid_x, self.ngrid_y = pix_dat_array.shape
        self.data = pix_dat_array
        self.noise_var = noise_var
        self.scale = scale
        self.gain = gain

    def initialize_param_values(self, param_file_name):
        """
        Initialize model parameter values from config file
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

    def _get_model_image(self):
        model_image = galsim.ImageF(self.ngrid_x, self.ngrid_y,
                                    scale=self.scale)
        for isrc in xrange(self.num_sources):
            model = self.src_models[isrc].get_image(image=model_image,
                                                    gain=self.gain)
        return model_image

    def lnprior(self, params):
        """
        Evaluate the log-prior of the model parameters
        """
        return 0.0

    def lnlike(self, params):
        """
        Evaluate the log-likelihood of the pixel data in a footprint
        """
        res = -np.inf
        valid_params = self.src_models[0].set_params(params)
        if valid_params:
            model = self._get_model_image()
            if model is None:
                res = -np.inf
            else:
                delta = (model.array - self.data)**2
                lnnorm = (- 0.5 * self.ngrid_x * self.ngrid_y *
                          np.sqrt(self.noise_var * 2 * np.pi))
                res = -0.5*np.sum(delta / self.noise_var) + lnnorm
        return res

    def __call__(self, params):
        return self.lnlike(params)


def init_roaster(args):
    """
    Initialize Roaster object, load data, and setup model
    """
    import yaml
    import footprints

    config = yaml.load(open(args.config_file))

    rstr = Roaster(config)

    dat, noise_var, scale, gain = footprints.load_image(config["io"]["infile"],
        segment=args.footprint_number)

    rstr.import_data(dat, noise_var, scale=scale, gain=gain)

    rstr.initialize_param_values(config["init"]["init_param_file"])

    return rstr

def do_sampling(args, rstr):
    """
    Execute MCMC sampling for posterior model inference
    """
    import emcee
    omega_interim = rstr.get_params()

    nvars = len(omega_interim)
    nsamples = rstr.config["sampling"]["nsamples"]
    nwalkers = rstr.config["sampling"]["nwalkers"]
    nthreads = rstr.config["sampling"]["nthreads"]

    p0 = emcee.utils.sample_ball(omega_interim, 
                                 np.ones_like(omega_interim) * 0.01, nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers,
                                    nvars,
                                    rstr,
                                    threads=nthreads)

    nburn = max([1, rstr.config["sampling"]["nburn"]])
    print "Burning with {:d} steps".format(nburn)
    pp, lnp, rstate = sampler.run_mcmc(p0, nburn)
    sampler.reset()

    pps = []
    lnps = []
    # lnpriors = []
    print "Sampling"
    for istep in range(nsamples):
        if np.mod(istep + 1, 20) == 0:
            print "\tStep {:d} / {:d}, lnp: {:5.4g}".format(istep + 1, nsamples,
                np.mean(pp))
        pp, lnp, rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
        lnprior = np.array([rstr.lnprior(omega) for omega in pp])
        pps.append(np.column_stack((pp.copy(), lnprior)))
        lnps.append(lnp.copy())

    pps, lnps = cluster_walkers(pps, lnps, thresh_multiplier=4)

    write_results(args, pps, lnps, rstr)
    return None

def cluster_walkers(pps, lnps, thresh_multiplier=1):
    """
    Down-select emcee walkers to those with the largest mean posteriors

    Follows the algorithm of Hou, Goodman, Hogg et al. (2012)
    """
    print "Clustering emcee walkers with threshold multiplier {:3.2f}".format(
        thresh_multiplier)
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

def write_results(args, pps, lnps, rstr):
    """
    Save and HDF5 file with posterior samples from Roaster
    """
    import os
    import h5py
    outfile = rstr.config["io"]["roaster_outfile"] + '_seg{:d}.h5'.format(args.footprint_number)
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    hfile = h5py.File(outfile, 'w')

    ### Store outputs in an HDF5 (sub-)group so we don't always
    ### need a separate HDF5 file for every segment.
    group_name = 'Samples/footprint{:d}'.format(args.footprint_number)
    grp = hfile.create_group(group_name)

    paramnames = rstr.config["model"]["model_params"].split()

    ## Write the MCMC samples and log probabilities
    if "post" in grp:
        del grp["post"]
    post = grp.create_dataset("post",
                              data=np.transpose(np.dstack(pps), [2, 0, 1]))
    # pnames = np.array(rstr.src_models[0][0].paramnames)
    post.attrs['paramnames'] = paramnames
    if "logprobs" in grp:
        del grp["logprobs"]
    _ = grp.create_dataset("logprobs", data=np.vstack(lnps))
    hfile.close()
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Draw interim samples of source model parameters via MCMC.')

    parser.add_argument('--config_file', type=str,
                        default="../config/jiffy.yaml",
                        help="Name of a configuration file listing inputs." +
                        "If specified, ignore other command line flags.")

    parser.add_argument("--footprint_number", type=int, default=0,
                        help="The footprint number to load from input")

    args = parser.parse_args()

    rstr = init_roaster(args)

    do_sampling(args, rstr)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
