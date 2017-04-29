#!/usr/bin/env python
# encoding: utf-8
"""
jiffy rstr.py

Draw posterior samples of image source model parameters given the
likelihood functxion of an image footprint
"""
import numpy as np
import galsim
import galsim_galaxy

class Roaster(object):
    """
    Likelihood model for footprint pixel data given a parametric source model
    """

    def __init__(self, config="../config/jiffy.yaml"):
        if isinstance(config, str):
            import yaml
            config = yaml.load(open(config))
        self.config = config

        actv_params = self.config["model"]["model_params"].split(" ")
        self.src_models = [galsim_galaxy.GalsimGalaxyModel(actv_params)]

        # Initialize objects describing the pixel data in a footprint
        self.ngrid_x = 64
        self.ngrid_y = 64
        self.noise_var = 3e-10
        self.scale = 0.2
        self.gain = 1.0
        self.data = None

    def _get_model_image(self):
        return self.src_models[0].get_image(self.ngrid_x, self.ngrid_y)

    def get_params(self):
        return self.src_models[0].get_params()

    def set_param_by_name(self, paramname, value):
        """
        Set a galaxy or PSF model parameter by name
        """
        self.src_models[0].params[paramname] = value
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

    def lnprior(self, params):
        """
        Evaluate the log-prior of the model parameters
        """
        return 0.0

    def lnlike(self, params):
        """
        Evaluate the log-likelihood of the pixel data in a footprint
        """
        self.src_models[0].set_params(params)
        model = self.src_models[0].get_image(self.ngrid_x, self.ngrid_y,
                                             scale=self.scale, gain=self.gain)
        delta = (model.array - self.data)**2
        lnnorm = (- 0.5 * self.ngrid_x * self.ngrid_y *
                  np.sqrt(self.noise_var * 2 * np.pi))
        return -0.5*np.sum(delta / self.noise_var) + lnnorm

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

    p0 = emcee.utils.sample_ball(omega_interim, 
                                 np.ones_like(omega_interim) * 0.01, nwalkers)

    sampler = emcee.EnsembleSampler(nwalkers,
                                    nvars,
                                    rstr)

    nburn = max([1, rstr.config["sampling"]["nburn"]])
    print "Burning with {:d} steps".format(nburn)
    pp, lnp, rstate = sampler.run_mcmc(p0, nburn)
    sampler.reset()

    pps = []
    lnps = []
    lnpriors = []
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
    outfile = rstr.config["io"]["roaster_outfile"]
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    hfile = h5py.File(outfile, 'w')

    ### Store outputs in an HDF5 (sub-)group so we don't always 
    ### need a separate HDF5 file for every segment.
    group_name='Samples/footprint{:d}'.format(args.footprint_number)
    grp = hfile.create_group(group_name)

    paramnames = rstr.config["model"]["model_params"].split()   

    ## Write the MCMC samples and log probabilities
    if "post" in grp:
        del grp["post"]
    post = grp.create_dataset("post", data=np.transpose(np.dstack(pps), [2,0,1]))
    # pnames = np.array(rstr.src_models[0][0].paramnames)
    post.attrs['paramnames'] = paramnames
    if "logprobs" in grp:
        del grp["logprobs"]
    logprobs = grp.create_dataset("logprobs", data=np.vstack(lnps))
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
