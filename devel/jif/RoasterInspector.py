# from __future__ import print_function
import argparse
import os
import sys
import copy

import h5py
import numpy as np
import yaml
import matplotlib.pyplot as plt
import corner

import galsim
import Roaster

plt.style.use('ggplot')


def gelman_rubin(chain):
    """
    Compute the Gelman-Rubin MCMC chain convergence statistic

    Assumes the input chain is in the emcee format.

    Copied from: http://joergdietrich.github.io/emcee-convergence.html
    """
    ssq = np.var(chain, axis=0, ddof=1)
    W = np.mean(ssq, axis=0)
    thetab = np.mean(chain, axis=1)
    thetabb = np.mean(thetab, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m - 1.) * np.sum((thetabb - thetab)**2, axis=0)
    var_theta = (n - 1.) / n * W + 1. / n * B
    R = np.sqrt(var_theta / W)
    return R


class RoasterInspector(object):
    """
    Compute summary statistics and plots for Roaster outputs
    """
    def __init__(self, args):
        print "<RoasterInspector> Loading segment {:d}".format(args.segment_number)
        self.args = args

        self.config = yaml.load(open(args.roaster_config))

        self._load_roaster_file(args)
        self._load_roaster_input_data()

        outdir = os.path.dirname(args.infile)[0]
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        return None

    def _load_roaster_file(self, args):
        f = h5py.File(args.infile, 'r')
        g = f['Samples/seg{:d}'.format(args.segment_number)]
        self.infile = args.infile
        self.roaster_infile = g.attrs['infile']
        self.segment_number = g.attrs['segment_number']

        self.epoch_num = self.config["data"]["epoch_num"]
        if self.epoch_num < 0:
                self.epoch_num = None

        self.paramnames = g['post'].attrs['paramnames']
        if len(self.paramnames.shape) > 1:
            self.paramnames = np.array(self.paramnames).ravel()

        self.nparams = len(self.paramnames)
        self.data = g['post'][...]
        self.logprob = g['logprobs'][...]
        f.close()
        return None

    def __str__(self):
        return ("<RoasterInspector>\n" + "Input file: %s" % self.args.infile)

    def _outfile_head(self):
        return os.path.splitext(self.infile)[0]

    def summary(self):
        print self.__str__()
        # print "Parameter names:", self.paramnames
        print "Roaster input file: ", self.roaster_infile
        print "Segment number:", self.segment_number
        # print "Segment number: {:d}".format(self.segment_number)
        print "data: ", self.data.shape
        print "galaxy model: {}".format(self.config["model"]["galaxy_model_type"])
        # print "filter subset: ", self.filters_to_load
        print "paramnames:", self.paramnames.shape, "\n", self.paramnames
        # print self.data
        # print self.logprob
        return None

    def report(self):
        print "\n"
        for i, p in enumerate(self.paramnames):
            print("%s = %4.3g +/- %4.3g" % (p, 
                np.mean(self.data[-self.args.keeplast:, :, i]),
                np.std(self.data[:, :, i])))
        print "\n"
        n = self.data.shape[2]
        rhat = gelman_rubin(self.data[:,:,0:(n-1)])
        print rhat
        print "Gelman-Rubin statistic: {:4.3f}".format(rhat[0])
        print "\n"

    def _get_opt_params(self):
        ndx = np.argmax(self.logprob[-self.args.keeplast:,...])
        opt_params = np.vstack(self.data[-self.args.keeplast:,...])[ndx,...]
        opt_params = opt_params[0:self.nparams]
        print "optimal parameters:", opt_params
        return opt_params

    def _load_roaster_input_data(self):
        self.roaster = Roaster.Roaster(config=self.config)

        self.roaster.Load(self.config["infiles"]["infile_1"],
                          segment=self.args.segment_number,
                          epoch_num=self.epoch_num,
                          use_PSFModel=self.config["model"]["use_psf_model"])
        self.roaster.initialize_param_values(self.config["init"]["init_param_file"])

        self.params_ref = self.roaster.get_params()

        print "Length of Roaster pixel data list: {:d}".format(
            len(self.roaster.pixel_data))
        return None

    def save_param_cov(self):
        n = len(self.paramnames)
        cov = np.cov(np.vstack(self.data[-self.args.keeplast:,:, 0:n]).transpose())
        print "cov:", cov
        outfile = (self._outfile_head() + '_param_cov.txt')
        np.savetxt(outfile, cov)
        return None

    def plot(self):
        n = len(self.paramnames)

        # Triangle plot
        # try:
        fig = corner.corner(np.vstack(self.data[-self.args.keeplast:,:, 0:n]),
                              labels=self.paramnames, truths=self.params_ref)
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
            print "noise variance: {:12.10g}".format(noise_var)
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

    def plot_GR_statistic(self):
        """
        Plot the Gelman-Rubin statistic versus chain step to monitor convergence

        Copied from: http://joergdietrich.github.io/emcee-convergence.html
        """
        fig = plt.figure(figsize=(8.9, 5.5))
        xmin = 100

        n = self.data.shape[2]
        chain = self.data[:, :, 0:(n-1)]
        chain_length = chain.shape[0]
        step_sampling = np.arange(xmin, chain_length, 5)
        rhat = np.array([gelman_rubin(chain[0:steps, :, :])  for steps in step_sampling])
        plt.plot(step_sampling, rhat, linewidth=2)
            
        ax = plt.gca()
        xmax = ax.get_xlim()[1]
        plt.hlines(1.1, xmin, xmax, linestyles="--")
        plt.ylabel("$\hat R$")
        plt.xlabel("chain length")
        # plt.ylim(1, np.max((2., rhat)))
        # legend = plt.legend(loc='best')
        outfile = (self._outfile_head() + '_gr_statistic.png')
        print "Saving {}".format(outfile)
        fig.savefig(outfile, bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("infile",
                        help="input HDF5 file with samples from Roaster")

    parser.add_argument("roaster_config",
                        help="Name of Roaster config file")

    parser.add_argument("--segment_number", type=int, default=0,
                        help="Index of the segment to load. Override any " + 
                             "value in the supplied Roaster config file.")

    parser.add_argument("--keeplast", type=int, default=0,
                        help="Keep last N samples.")

    args = parser.parse_args()

    inspector = RoasterInspector(args)
    inspector.summary()
    inspector.report()
    # inspector.save_param_cov()
    inspector.plot()
    inspector.plot_data_and_model()
    inspector.plot_GR_statistic()
    return 0


if __name__ == '__main__':
    sys.exit(main())