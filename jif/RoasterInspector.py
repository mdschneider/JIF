from __future__ import print_function
import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
import triangle

import Roaster


class RoasterInspector(object):
    """
    Compute summary statistics and plots for Roaster outputs
    """
    def __init__(self, args):
        self.args = args
        f = h5py.File(args.infile, 'r')
        self.roaster_infile = f.attrs['infile']
        self.segment_number = f.attrs['segment_number']
        self.galaxy_model_type = f.attrs['galaxy_model_type']
        self.telescope = f.attrs['telescope']
        self.model_paramnames = f.attrs['model_paramnames']
        self.paramnames = f['post'].attrs['paramnames']
        if len(self.paramnames.shape) > 1:
            self.paramnames = self.paramnames[0]
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
        opt_params = self.data[-self.args.keeplast:,...][ndx]
        return opt_params

    def _load_roaster_input_data(self):
        self.roaster = Roaster.Roaster(galaxy_model_type=self.galaxy_model_type,
            telescope=self.telescope,
            model_paramname=self.model_paramname)
        self.roaster.Load(self.roaster_infile) ### puts data in Roaster.pixel_data
        return None

    def plot(self):
        if not os.path.exists(self.args.outprefix):
            os.makedirs(self.args.outprefix)

        # Triangle plot
        fig = triangle.corner(np.vstack(self.data[-self.args.keeplast:,...]),
                              labels=self.paramnames, truths=self.args.truths)
        outfile = os.path.join(self.args.outprefix,
            "roaster_inspector_triangle.png")
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
        outfile = os.path.join(self.args.outprefix,
            'roaster_inspector_walkers.png')
        print "Saving {}".format(outfile)
        fig.savefig(outfile)
        return None

    def plot_data_and_model(self):
        """
        Plot panels of pixel data, model, and residuals
        """
        self._load_roaster_input_data()
        for idat, dat in enumerate(Roaster.pixel_data):
            fig = plt.figure(figsize=(10, 10/1.618))

            ### pixel data
            ax = fig.add_subplot(2, 2, 1)
            ax.title("Image")
            ax.imshow(dat, interpolation='none', origin='lower',
                cmap=plt.cm.gray)
            x.add_colorbar()

            ### model
            ax = fig.add_subplot(2, 2, 2)
            ax.title("Opt Model")
            opt_params = self._get_opt_params()
            valid_params = self.roaster.set_params(opt_params)
            model_image = self.roaster._get_model_image(idat)
            ax.imshow(model_image.array, interpolation='none', origin='lower',
                      cmap=plt.cm.gray)
            x.add_colorbar()

            resid = dat - model_image.array

            ### model + noise
            ax = fig.add_subplot(2, 2, 3)
            ax.title("Model + Noise")
            noise = galsim.GaussianNoise(
                sigma=np.sqrt(Roaster.pix_noise_var[idat]))
            model_image.addNoise(noise)
            ax.imshow(model_image.array, interpolation='none', orign='lower',
                      cmap=plt.cm.gray)
            x.add_colorbar()

            ### residual (chi)
            ax = fig.add_subplot(2, 2, 4)
            ax.title("Residual")
            ax.imshow(resid, interpolation='none', origin='lower',
                      cmap=plt.cm.BrBG)
            ax.add_colorbar()

            plt.tight_layout()
            outfile = os.path.join(self.args.outprefix,
                'roaster_data_and_model_{:d}.png'.format(idat))
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

    parser.add_argument("--outprefix", default='../output/roasting/', type=str,
                        help="Prefix to apply to output figures.")

    args = parser.parse_args()

    inspector = RoasterInspector(args)
    inspector.summary()
    inspector.report()
    inspector.plot()
