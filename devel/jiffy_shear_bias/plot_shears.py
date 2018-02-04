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

k_n_fields = 199
#k_truth_topdir = "small_shapenoise/control/ground/constant/"
#k_stooker_topdir = "small_shapenoise/reaper/jif/"
#k_thresher_topdir = "small_shapenoise/thresher/cpp/"

k_truth_topdir = "/Volumes/PromisePegasus/JIF/cgc2/control/ground/constant"
k_stooker_topdir = "/Volumes/PromisePegasus/JIF/cgc2/reaper/jif"
k_thresher_topdir = "/Volumes/PromisePegasus/JIF/cgc2/thresher/CPP"


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

def get_thresher_field_shears(field_num, nburn=100, thin=2,
                              return_samples=False):
    """
    Get the mean and std. dev. marginal shears from Thresher outputs
    """
    
    infile = os.path.join(k_thresher_topdir, 
        "thresher_{0:0>3}_galdist0.h5".format(field_num))
    print "infile: ", infile    

    try:
        f = h5py.File(infile, 'r')
        shear = f['shear/shear'][nburn::thin, ...]
        f.close()
    except IOError:
        shear = np.zeros((1, 3), dtype=np.float64)

    if return_samples:
        return shear[:, 0:2]
    else:
        g_mean = np.mean(shear[:,0:2], axis=0)
        g_sd = np.sqrt(np.var(shear[:,0:2], axis=0))
        return g_mean, g_sd

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
    g_mean_st = []
    g_sd_st = []
    g_mean_th = []
    g_sd_th = []
    g_truth = []
    # Columns: (1) Stooker mean,
    #          (2) Stooker std. dev., 
    #          (3) Thresher mean,
    #          (4) Thresher std. dev., 
    #          (5) truth
    shears = np.zeros((k_n_fields, 2, 5))
    for field_num in xrange(k_n_fields):
        try: 
            gm_st, gs_st = get_stooker_field_shears(field_num=field_num)
            gm_th, gs_th = get_thresher_field_shears(field_num=field_num)
            gt = get_true_field_shears(field_num=field_num)
            shears[field_num, :, 0] = gm_st
            shears[field_num, :, 1] = gs_st
            shears[field_num, :, 2] = gm_th
            shears[field_num, :, 3] = gs_th
            shears[field_num, :, 4] = gt
        except IOError:
            pass
    return shears

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
        shears = load_all_shears()
        print shears.shape

        print "delta (mean - true) shears:", shears[:, :, 0] - shears[:, :, 4]

        plt.style.use('ggplot')
        fig = plt.figure(figsize=(10, 8))

        for i in xrange(2):
            x = shears[:, i, 4]
            y = shears[:, i, 0] - shears[:, i, 4]

            yth = shears[:, i, 2] - shears[:, i, 4]

            ax = plt.subplot(2, 1, i+1)
            plt.axhline(y=0, color='grey')
            plt.ylabel(r"$\Delta g_{:d}$".format(i+1))

            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            st_bias = r"$m = {:3.2g} \pm {:3.2g}$, $c = {:3.2g}$".format(
                slope, std_err, intercept)

            xl = np.linspace(np.min(x), np.max(x), 50)
            plt.plot(xl, slope*xl + intercept, '--', color='black')

            # Redo the regression for Thresher shears
            slope, intercept, r_value, p_value, std_err = linregress(x, yth)
            th_bias = r"$m = {:3.2g} \pm {:3.2g}$, $c = {:3.2g}$".format(
                slope, std_err, intercept)
            plt.plot(xl, slope*xl + intercept, '--', color='red')

            print slope, intercept
            plt.title("Stooker: " + st_bias + " | Thresher: " + th_bias)

            ax.errorbar(x, y, yerr=shears[:, i, 1], fmt='.', markersize=10,
                        label="Stooker", color='black', alpha=0.5)

            ax.errorbar(x, yth, yerr=shears[:, i, 3], fmt='.', markersize=10,
                        label="Thresher", color='red', alpha=0.5)

            plt.xlim(-0.1, 0.1)
            if i == 0:
                plt.legend()
        
        plt.xlabel(r"True $g$")
        plt.tight_layout()
        plotfile = "shear_plot.png"
        plt.savefig(plotfile, bbox_inches='tight')

    return 0

if __name__ == "__main__":
    sys.exit(main())
