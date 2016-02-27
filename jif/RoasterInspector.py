# from __future__ import print_function
import argparse
import os
import sys
import copy

import h5py
import numpy as np
import matplotlib.pyplot as plt
import corner

import galsim
import Roaster


class RoasterInspector(object):
    """
    Compute summary statistics and plots for Roaster outputs
    """
    def __init__(self, args):
        self.args = args
        f = h5py.File(args.infile, 'r')
        self.infile = args.infile
        self.roaster_infile = f.attrs['infile']
        self.segment_number = f.attrs['segment_number']
        self.galaxy_model_type = f.attrs['galaxy_model_type']
        self.filters_to_load = f.attrs['filters_to_load']
        if isinstance(self.filters_to_load, str):
            if self.filters_to_load == 'None':
                self.filters_to_load = None
            else:
                self.filters_to_load = [self.filters_to_load]
        self.telescope = f.attrs['telescope']
        if self.telescope == 'None':
            self.telescope = None
        self.model_paramnames = f.attrs['model_paramnames']
        self.achromatic_galaxy = f.attrs['achromatic_galaxy']
        self.paramnames = f['post'].attrs['paramnames']
        if len(self.paramnames.shape) > 1:
            self.paramnames = np.array(self.paramnames).ravel()
        #     self.paramnames = self.paramnames[0]
        self.data = f['post'][...]
        self.logprob = f['logprobs'][...]
        self.nburn = f.attrs['nburn']
        f.close()

        outdir = os.path.dirname(args.infile)[0]
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        ### The lists input from HDF5 can lose the commas between entries.
        ### This seems to fix it:
        self.model_paramnames = [m for m in self.model_paramnames]

    def __str__(self):
        return ("<RoasterInspector>\n" + "Input file: %s" % self.args.infile)

    def _outfile_head(self):
        return os.path.splitext(self.infile)[0]

    def summary(self):
        print self.__str__()
        # print "Parameter names:", self.paramnames
        print "Roaster input file: ", self.roaster_infile
        print "Segment number: {:d}".format(self.segment_number)
        print "data: ", self.data.shape
        print "galaxy model: {}".format(self.galaxy_model_type)
        print "filter subset: ", self.filters_to_load
        print "paramnames:", self.paramnames.shape, "\n", self.paramnames
        # print self.data
        # print self.logprob
        return None

    def report(self):
        print "\n"
        for i, p in enumerate(self.paramnames):
            print("%s = %4.3g +/- %4.3g" % (p, np.mean(self.data[:, :, i]),
                np.std(self.data[:, :, i])))
        print "\n"

    def _get_opt_params(self):
        ndx = np.argmax(self.logprob[-self.args.keeplast:,...])
        opt_params = np.vstack(self.data[-self.args.keeplast:,...])[ndx,...]
        # opt_params = np.median(np.vstack(self.data[-self.args.keeplast:,...]), axis=0)
        return opt_params

    def _load_roaster_input_data(self):
        self.roaster = Roaster.Roaster(galaxy_model_type=self.galaxy_model_type,
            telescope=self.telescope,
            model_paramnames=self.model_paramnames,
            filters_to_load=self.filters_to_load,
            debug=False,
            achromatic_galaxy=self.achromatic_galaxy)
        ### The following puts data in self.roaster.pixel_data
        self.roaster.Load(self.roaster_infile, segment=self.segment_number)
        print "Length of Roaster pixel data list: {:d}".format(len(self.roaster.pixel_data))
        return None

    def plot(self):
        n = len(self.paramnames)

        # Triangle plot
        # try:
        fig = corner.corner(np.vstack(self.data[-self.args.keeplast:,:, 0:n]),
                              labels=self.paramnames, truths=self.args.truths)
        outfile = (self._outfile_head() +
            "_roaster_inspector_triangle.png")
        print "Saving {}".format(outfile)
        fig.savefig(outfile)
        # except ValueError:
        #     pass

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
        outfile = (self._outfile_head() +
            '_roaster_inspector_walkers.png')
        print "Saving {}".format(outfile)
        fig.savefig(outfile)
        return None

    def plot_data_and_model(self):
        """
        Plot panels of pixel data, model, and residuals
        """
        self._load_roaster_input_data()
        for idat, dat in enumerate(self.roaster.pixel_data):
            fig = plt.figure(figsize=(10, 10/1.618))

            vmin = dat.min()
            vmax = dat.max()

            ### pixel data
            ax = fig.add_subplot(2, 2, 1)
            ax.set_title("Image")
            cax = ax.imshow(dat, interpolation='none',
                            cmap=plt.cm.pink, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(cax)

            ### model
            ax = fig.add_subplot(2, 2, 2)
            ax.set_title("Opt Model")
            opt_params = self._get_opt_params()
            valid_params = self.roaster.set_params(opt_params)
            model_image = self.roaster._get_model_image(idat)
            cax = ax.imshow(model_image.array, interpolation='none',
                            cmap=plt.cm.pink, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(cax)

            resid = dat - model_image.array
            noisy_image = copy.deepcopy(model_image)

            ### model + noise
            noise_var = self.roaster._get_noise_var(idat)
            print "noise variance: ", noise_var
            ax = fig.add_subplot(2, 2, 3)
            ax.set_title("Model + Noise")
            noise = galsim.GaussianNoise(sigma=np.sqrt(noise_var))
            noisy_image.addNoise(noise)
            cax = ax.imshow(noisy_image.array, interpolation='none',
                            cmap=plt.cm.pink, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(cax)

            ### residual (chi)
            ax = fig.add_subplot(2, 2, 4)
            ax.set_title("Residual")
            cax = ax.imshow(resid, interpolation='none',
                            cmap=plt.cm.BrBG)
            cbar = fig.colorbar(cax)

            plt.tight_layout()
            outfile = (self._outfile_head() +
                '_data_and_model_epoch{:d}.png'.format(idat))
            print "Saving {}".format(outfile)
            fig.savefig(outfile)
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("infile",
                        help="input HDF5 file with samples from Roaster")

    parser.add_argument("--truths", type=float,
                        help="true values of hyperparameters: \
                              {Omega_m, sigma_8, ...}",
                        nargs='+')

    parser.add_argument("--keeplast", type=int, default=0,
                        help="Keep last N samples.")

    # parser.add_argument("--outprefix", default='../output/roasting/', type=str,
    #                     help="Prefix to apply to output figures.")

    args = parser.parse_args()

    inspector = RoasterInspector(args)
    inspector.summary()
    inspector.report()
    inspector.plot()
    inspector.plot_data_and_model()
