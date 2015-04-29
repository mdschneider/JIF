#!/usr/bin/env python
# encoding: utf-8
"""
gpp_space-ground.py

Galaxy Posterior Plot (GPP) script.

Generate a plot of the marginal posteriors for a galaxy image for a single model parameter.

The plot can compare the posteriors obtained with:
    1) only space data
    2) only ground data
    3) the combination of space & ground data
    4) parameter estimate(s) from the LSST DM stack
"""
import argparse
import sys
import os.path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import galsim_galaxy

import logging


# Print log messages to file.
logging.basicConfig(filename='logs/gpp_space-ground.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class GalaxyPosteriorPlot(object):
    """
    Galaxy Posterior Plot
    """
    def __init__(self, args):
        self.args = args

        ### Load the combined epoch MCMC samples
        self.pnames_epoch_all, self.data_epoch_all = self.load(args.outhead + ".h5")
        print self.data_epoch_all.shape

        ### Load epoch 0 (ground) MCMC samples
        self.pnames_epoch0, self.data_epoch0 = self.load(args.outhead + "_epoch0.h5")

        ### Load epoch 1 (space) MCMC samples
        self.pnames_epoch1, self.data_epoch1 = self.load(args.outhead + "_epoch1.h5")

    def load(self, infile):
        if os.path.isfile(infile):
            logging.info("Reading data from {}".format(infile))
            f = h5py.File(infile, 'r')
            paramnames = f['post'].attrs['paramnames']
            data = f['post'][...]
            data = data[self.args.nburn:, :, :]
            f.close()
            return paramnames, data
        else:
            logging.debug("Cannot find {}".format(infile))
            return None, None

    def plot(self):
        """
        Make the marginal posterior plot
        """
        logging.debug("Making plot")
        fig = plt.figure(figsize=(8, 8/1.618))
        ax = fig.add_subplot(1, 1, 1)

        param_ndx = np.argwhere(self.pnames_epoch_all == self.args.param_name)[0]
        print "param_ndx:", param_ndx, len(param_ndx)
        if len(param_ndx) != 1:
            print self.pnames_epoch_all
            raise ValueError("Parameter name {} does not match any in input file".format(
                self.args.param_name))

        pmin = np.min(self.data_epoch_all[:, :, param_ndx])
        pmax = np.max(self.data_epoch_all[:, :, param_ndx])

        density = gaussian_kde(self.data_epoch_all[:, :, param_ndx].ravel())
        xs = np.linspace(pmin, pmax, 200)
        ax.plot(xs, density(xs), label="Combined", color='black', linewidth=3)

        if self.pnames_epoch0 is not None:
            pmin = np.min(self.data_epoch0[:, :, param_ndx])
            pmax = np.max(self.data_epoch0[:, :, param_ndx])

            density = gaussian_kde(self.data_epoch0[:, :, param_ndx].ravel())
            xs = np.linspace(pmin, pmax, 200)
            ax.plot(xs, density(xs), label="Ground")

        if self.pnames_epoch1 is not None:
            pmin = np.min(self.data_epoch1[:, :, param_ndx])
            pmax = np.max(self.data_epoch1[:, :, param_ndx])

            density = gaussian_kde(self.data_epoch1[:, :, param_ndx].ravel())
            xs = np.linspace(pmin, pmax, 200)
            ax.plot(xs, density(xs), label="Space")            

        if self.args.truth is not None:
            ax.axvline(self.args.truth, linestyle='dashed', color='grey')

        ax.set_xlabel(r"{}".format(self.args.param_name))
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()

        plotfile = self.args.outhead + '_gpp_{}.png'.format(self.args.param_name)
        logging.info("Saving plot to {}".format(plotfile))
        fig.savefig(plotfile, bbox_inches='tight')
        return None
        

def main():
    parser = argparse.ArgumentParser(
        description='Galaxy Posterior Plot for space and ground imaging of a single galaxy.')

    parser.add_argument("--outhead", type=str, default="../output/roasting/roaster_out",
                        help="Head of the output file(s) from Roaster \
                        (Default: ../output/roasting/roaster_out)")

    parser.add_argument("--param_name", type=str, default="e_disk",
                        help="Name of the parameter to plot (Default: 'e_disk')")

    parser.add_argument("--truth", type=float, default=None, help="True value of the parameter")

    parser.add_argument("--nburn", type=int, default=100, 
                        help="Drop nburn samples from the start of each chain.")

    args = parser.parse_args()

    gpp = GalaxyPosteriorPlot(args)

    gpp.plot()

    return 0


if __name__ == "__main__":
    sys.exit(main())
