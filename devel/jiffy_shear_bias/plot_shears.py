#!/usr/bin/env python
# encoding: utf-8
"""
@file plot_shears.py

Created on 2016-12-28
"""
import argparse
import sys
import os.path
import numpy as np
from scipy.stats import linregress
import h5py
from astropy.io import fits
import matplotlib.pyplot as plt
import emcee
import corner

plt.style.use('ggplot')

k_n_fields = 50
k_truth_topdir = "control/ground/constant/"
k_stooker_topdir = "midsnr/reaper/jif/"
k_thresher_topdir = "thresher/cpp/"


def get_stooker_field_shears(field_num, return_samples=False):
    """
    Get the mean and std. dev. marginal shears from Stooker output
    """
    infile = os.path.join(k_stooker_topdir, 
        "{0:0>3}".format(field_num), "reaper_{0:0>3}.h5".format(field_num))

    f = h5py.File(infile, 'r')
    if return_samples:
        res = f['gals/samples'][:, 0:2]
    else:
        means = f['gals/means'][...]
        std_devs = f['gals/std_devs'][...]
        res = (means[0:2], std_devs[0:2])
    f.close()
    return res


def get_true_field_shears(field_num):
    """
    Get the true shears for a CGC field
    """
    infile = os.path.join(k_truth_topdir, 
                          "epoch_catalog-{0:0>3}-0.fits".format(field_num))
    hdulist = fits.open(infile)
    tbdata = hdulist[1].data
    g = [tbdata.field('g1')[0], tbdata.field('g2')[0]]
    return g

def load_all_shears():
    g_mean = []
    g_sd = []
    g_truth = []
    for field_num in xrange(k_n_fields):
        try: 
            gm, gs = get_stooker_field_shears(field_num=field_num)
            # gm, gs = get_thresher_field_shears(field_num=field_num)
            gt = get_true_field_shears(field_num=field_num)
            g_mean.append(gm)
            g_sd.append(gs)
            g_truth.append(gt)
        except IOError:
            pass
    return np.array(g_mean), np.array(g_sd), np.array(g_truth)

# ==============================================================================
# ==============================================================================
def main():
    """
    @brief      Main function: Plot inferred shear residuals vs true shears
    
    @return     Status indicator
    """
    parser = argparse.ArgumentParser(
        description="Plot inferred shear residuals vs true shears")

    parser.add_argument("--print_summary", action='store_true',
        help="Print summary statistics after an MCMC chain has been run.")

    parser.add_argument("--use_mc_posterior", action='store_true')

    parser.add_argument("--nsamples", default=250, type=int,
                        help="Number of samples for each emcee walker (Default: 250)")
    parser.add_argument("--nwalkers", default=8, type=int,
                        help="Number of emcee walkers (Default: 64)")
    parser.add_argument("--nburn", default=50, type=int,
                        help="Number of burn-in steps (Default: 50)")
    parser.add_argument("--nthreads", default=1, type=int,
                        help="Number of threads to use (Default: 8)") 

    parser.add_argument("--quiet", action="store_true")   

    args = parser.parse_args()

    if args.print_summary:
        print_marginal_constraints()
        return 0

    if args.use_mc_posterior:
        infer_mc_posterior(args)
    else:
        g_mean, g_sd, g_truth = load_all_shears()
        print g_mean.shape
        print g_sd.shape
        print g_truth.shape

        print "delta (mean - true) shears:", g_mean - g_truth

        plt.style.use('ggplot')
        fig = plt.figure(figsize=(10, 8))

        for i in xrange(2):
            x = g_truth[:,i]
            y = g_mean[:,i] - g_truth[:,i]
            ax = plt.subplot(2, 1, i+1)
            plt.axhline(y=0, color='grey')
            plt.ylabel(r"$\Delta g_{:d}$".format(i+1))
            slope, intercept, r_value, p_value, std_err = linregress(x,y)
            print slope, intercept
            # ax.text(-0.05, 0.0025,
            #          r"$m = {:3.2f} +/- {:3.2f}$, $c = {:3.2e}$".format(slope, std_err, intercept),
            #          fontsize=18)
            plt.title(r"$m = {:3.2e} +/- {:3.2e}$, $c = {:3.2e}$".format(slope, std_err, intercept))
            xl = np.linspace(np.min(x), np.max(x), 50)
            plt.plot(xl, slope*xl + intercept, '--')
            # ax.plot(x, y, '.')
            ax.errorbar(x, y, yerr=g_sd[:,i], fmt='.', markersize=10)
            plt.xlim(-0.1, 0.1)
        
        plt.xlabel(r"True $g$")
        plt.tight_layout()
        plotfile = "shear_plot.png"
        plt.savefig(plotfile, bbox_inches='tight')

    return 0

if __name__ == "__main__":
    sys.exit(main())