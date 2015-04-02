import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import triangle

import Roaster


class RoasterInspector(object):
    """Compute summary statistics and plots for Roaster outputs"""
    def __init__(self, args):
        self.args = args
        f = h5py.File(args.infile, 'r')
        self.paramnames = f['post'].attrs['paramnames']        
        self.data = f['post'][...]
        self.logprob = f['logprobs'][...]
        self.nburn = f.attrs['nburn']        
        f.close()

    def __str__(self):
        return ("<RoasterInspector>\n" + "Input file: %s" % self.args.infile)       

    def summary(self):
        print self.__str__()
        # print "Parameter names:", self.paramnames
        print "data: ", self.data.shape
        # print self.data
        # print self.logprob
        return None

    def report(self):
        print "\n"
        for i, p in enumerate(self.paramnames):
            print "%s = %4.3g +/- %4.3g" % (p, np.mean(self.data[:, :, i]), np.std(self.data[:, :, i]))
        print "\n"

    def plot(self):
        # Triangle plot
        fig = triangle.corner(np.vstack(self.data[-self.args.keeplast:,...]),
                              labels=self.paramnames, truths=self.args.truths)
        outfile = ''.join([self.args.outprefix, "roaster_inspector_triangle.png"])
        print "Saving {}".format(outfile)
        fig.savefig(outfile)

        # Walkers plot

        # First determine size of plot
        nparams = len(self.paramnames) + 1  # +1 for lnprob plot
        # Try to make plot aspect ratio near golden
        ncols = int(np.ceil(np.sqrt(nparams*1.618)))
        nrows = int(np.ceil(1.0*nparams/ncols))

        fig = plt.figure(figsize = (3.0*ncols,3.0*nrows))

        for i, p in enumerate(self.paramnames):
            ax = fig.add_subplot(nrows, ncols, i+1)
            ax.plot(self.data[:, :, i])
            ax.set_ylabel(p)
        ax = fig.add_subplot(nrows, ncols, i+2)
        ax.plot(self.logprob)
        ax.set_ylabel('ln(prob)')
        fig.tight_layout()
        outfile = ''.join([self.args.outprefix, 'roaster_inspector_walkers.png'])
        print "Saving {}".format(outfile)
        fig.savefig(outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="input HDF5 file with samples from Roaster")
    parser.add_argument("--truths", type=float,
                        help="true value of hyperparameters: {Omega_m, sigma_8, ...}",
                        nargs='+')
    parser.add_argument("--keeplast", type=int, help="Keep last N samples.", default=0)
    parser.add_argument("--outprefix", default='../output/roasting/', type=str,
                        help="Prefix to apply to output figures.")
    args = parser.parse_args()

    inspector = RoasterInspector(args)
    inspector.summary()
    inspector.report()
    inspector.plot()
