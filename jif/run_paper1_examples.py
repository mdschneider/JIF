#!/usr/bin/env python
# encoding: utf-8
"""
run_paper1_examples.py

Run the example scenarios for JIF paper 1.

All examples contain an isolated bulge+disk galaxy model as seen from ground & space.

TODO: Modify image filenames for different example numbers.

Examples
--------
    1. r-band / r-band; simulated; known SEDs.
    2. r-band / F178-band; simulated; known SEDs.
    3. r-band / r-band; simulated; unknown SEDs.
    4. r-band / F178-band; simulated; unknown SEDs.
    5. r-band / r-band; observed; known redshift (model SEDs).
    6. r-band / K-band; observed; known redshift (model SEDs).
"""

import argparse
import sys
import os.path
# import numpy as np

import galsim_galaxy
import Roaster
import gpp_space_ground as gpp_plot

import logging


# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
#logging.basicConfig(filename='logs/run_paper1_examples.log',
#                     level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


### Names of the filters indexed by example numbers
k_filters_ground = ['r', 'r', 'r', 'r', 'r', 'r']
k_filters_space = ['r', 'y', 'r', 'y', 'r', 'y'] ### TODO: Update bands for space 
### Roaster data format specifications indexed by example numbers
k_data_formats = ['test_galsim_galaxy', 'test_galsim_galaxy', 'test_galsim_galaxy', 
                  'test_galsim_galaxy', 
                  'observed', 'observed']
### Redshift of the galaxy for each example - TODO: add redshifts for observed galaxy examples 4,5
k_redshifts = [1., 1., 1., 1., 0., 0.]


class RoasterArgs(object):
    """
    Arguments expected by the Roaster script
    """
    def __init__(self, example_num, file_lab="", epoch=None):
        self.outfile = '../output/roasting/example{:d}/roaster_out{}'.format(example_num, file_lab)
        self.epoch = epoch
        self.seed = 6199256
        self.nsamples = 200
        self.nwalkers = 64
        self.nburn = 50
        self.nthreads = 1
        self.quiet = False
        

class GPPArgs(object):
    
    """
    Arguments expected by gpp_space_ground script
    """
    def __init__(self, example_num, file_lab=""):
        ### Head of the output file(s) from Roaster
        self.outhead = '../output/roasting/example{:d}/roaster_out{}'.format(example_num, file_lab)
        ### Name of the parameter to plot
        self.param_name = 'e_disk'
        ### True value of the parameter
        self.truth = 0.3
        ### Drop nburn samples from the start of each Roaster chain
        self.nburn = 100


def main():
    parser = argparse.ArgumentParser(
        description='Run the example scenarios for JIF paper 1.')
    
    parser.add_argument('--example_num', type=int, default=1,
                        help='Index of the example to run.')

    args = parser.parse_args()

    ### Array index corresponding to the example number
    ex_ndx = args.example_num - 1

    ### Label to attach to simulated image files and output plots
    file_lab = '_paper1_ex{:d}'.format(args.example_num)

    outdir = "../output/roasting/example{:d}".format(args.example_num)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    logging.debug('*** Started run for JIF paper 1, example {:d}'.format(args.example_num))

    logging.debug('Generating simulated images')
    galsim_galaxy.make_test_images(
        filter_name_ground=k_filters_ground[ex_ndx],
        filter_name_space=k_filters_space[ex_ndx],
        file_lab=file_lab)

    ### Run Roaster for epochs 0, 1 individually and then combined (epoch == None)
    epochs = [0, 1, None]
    for epoch in epochs:
        logging.debug('Setting up Roaster')
        ### Reset the module-level pixel data list in Roaster.
        ### Otherwise, this list gets appended to under subsequent steps in the 'epoch' loop.
        Roaster.pixel_data = [] 
        roaster = Roaster.Roaster(data_format=k_data_formats[ex_ndx],
            lnprior_omega=Roaster.DefaultPriorBulgeDisk(z_mean=k_redshifts[ex_ndx]),
            galaxy_model_type="BulgeDisk",
            epoch=epoch)
        roaster.Load("../TestData/test_image_data" + file_lab + ".h5")
        logging.debug('Running Roaster MCMC')
        roaster_args = RoasterArgs(args.example_num, file_lab=file_lab, epoch=epoch)
        Roaster.do_sampling(roaster_args, roaster)

    ### TODO: Add DM wrapper here

    logging.debug('Making plots of posterior constraints for example {:d}'.format(args.example_num))
    gpp_args = GPPArgs(args.example_num, file_lab=file_lab)
    gpp = gpp_plot.GalaxyPosteriorPlot(gpp_args)
    gpp.plot()

    logging.debug('*** Finished run for JIF paper 1, example {:d}'.format(args.example_num))
    return 0


if __name__ == "__main__":
    sys.exit(main())
