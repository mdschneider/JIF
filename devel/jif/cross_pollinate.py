#!/usr/bin/env python
# encoding: utf-8
"""
cross_pollinate.py

Combine multiple epochs from Roaster with importance sampling weights
"""
from __future__ import print_function
import argparse
import sys
import os.path
import numpy as np
import h5py
import Roaster

import logging


# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
#logging.basicConfig(filename='logs/cross_pollinate.log',
#                     level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


def init_roasters(config_files):
    """
    Initialize the Roaster objects for all epochs

    @param config_files List of config file names for Roaster setup

    @return List of Roaster instances for each entry in config_files
    """
    args = [Roaster.ConfigFileParser(cf) for cf in config_files]
    r = [Roaster.InitRoaster(a) for a in args]
    roasters = [x[0] for x in r]
    args = [x[1] for x in r]
    return roasters, args


def load_interim_samples(args, segment_number=0):
    """
    Load the Roaster outputs for all epochs

    @param args Argument list for Roaster as output from init_roasters()
    """
    infiles = [a.outfile + ".h5" for a in args]
    res = []
    for infile in infiles:
        f = h5py.File(infile, 'r')
        g = f['Samples/seg{:d}'.format(segment_number)]
        samples = g['post'][...]
        lnp = g['logprobs'][...]
        res.append({"samples":samples, "lnp":lnp})
        f.close()
    return res


def cross_pollinate(args, roasters, epoch_samples):
    """
    Calculate the MIS weights
    """
    print("\n===== Cross-pollinating =====")
    ### Loop over samples for each epoch and calculate IS weights for every sample
    for i, epoch in enumerate(epoch_samples): # select the interim samples for epoch 'i'
        print("\tevaluating likelihoods for epoch {:d} samples".format(i))
        omega = epoch["samples"]
        # print("cp, omega:", omega.shape)
        lnp = []
        for j, epoch_num_j in enumerate(epochs): # select the likelihood for epoch 'j'
            if i == j:
                lnp_val = epoch["lnp"]
                # print("lnp: ", lnp_val)
            else:
                lnp_func = epoch_samples[j]["roaster"]

                # TODO: get PSF params from epoch j and append to galaxy params from epoch i.
                # The PSF params could be resampled from the existing samples for epoch j.

                ### Evaluate the ln-posterior of the i'th samples given the j'th data set
                lnp_val = np.array([
                    np.array([lnp_func(omega[istep, iwalk, :])
                        for iwalk in xrange(args.nwalkers)])
                    for istep in xrange(nsamples)])
                # print("lnp: ", lnp_val)
            lnp.append(lnp_val)
        lnp = np.array(lnp)
        ### ln(prod(p) / sum(p))
        weights = np.sum(lnp, axis=0) - np.logaddexp.reduce(lnp, axis=0)
        # print("weights: ", weights)
        epoch_samples[i]['weights'] = weights
    return epoch_samples


def write_cross_pollinate_results(args, epoch_samples, paramnames):
    """
    Write the combined Roaster samples and MIS weights to HDF5 file
    """
    ### Concatenate all samples and weights into a single 2d array
    samples = np.concatenate([e["samples"] for e in epoch_samples])
    # print("samples:", samples.shape)
    wts = np.concatenate([e["weights"] for e in epoch_samples])
    # print("weights:", wts.shape)

    nsteps, nwalkers, nparams = samples.shape
    dat = np.empty((nsteps, nwalkers, nparams+1), dtype=np.float64)
    dat[:, :, 0:nparams] = samples
    dat[:, :, nparams] = wts
    # print("dat:", dat.shape)

    paramnames.append('lnprior')
    paramnames.append('wt')
    # print("paramnames:", paramnames)

    outfile = 'output/cross_pollinate_out.h5'
    logging.info("\tWriting to {}".format(outfile))
    f = h5py.File(outfile, 'w')
    if "post" in f:
        del f["post"]
    post = f.create_dataset("post", data=dat)
    post.attrs['paramnames'] = paramnames
    f.close()
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Combine multiple epochs from Roaster with importance '+
                    'sampling weights')

    parser.add_argument("--config_files", type=str, nargs='+',
                        "Names of the Roaster configuration files to process")

    args = parser.parse_args()

    logging.debug('Cross-pollinate started')

    roasters, args = init_roasters(config_files)
    epoch_samples = load_interim_samples(args, segment_number=0)
    cross_pollinate(args, roasters, epoch_samples)
    write_cross_pollinate_results(args, epoch_samples, model_paramnames)

    logging.debug('Cross-pollinate finished')
    return 0


if __name__ == "__main__":
    sys.exit(main())
