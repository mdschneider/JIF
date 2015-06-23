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
import RoasterInspector
import gpp_space_ground as gpp_plot

import logging


# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
#logging.basicConfig(filename='logs/run_paper1_examples.log',
#                     level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


# ### Names of the filters indexed by example numbers
# k_filters_ground = ['r', 'r', 'r', 'r', 'r', 'r']
# k_filters_space = ['r', 'y', 'r', 'y', 'r', 'y'] ### TODO: Update bands for space 
# ### Roaster data format specifications indexed by example numbers
# k_data_formats = ['test_galsim_galaxy', 'test_galsim_galaxy', 'test_galsim_galaxy', 
#                   'test_galsim_galaxy', 
#                   'observed', 'observed']
# ### Redshift of the galaxy for each example - TODO: add redshifts for observed galaxy examples 4,5
# k_redshifts = [1., 1., 1., 1., 0., 0.]

### Indices of the galsim_galaxy Bulge+Disk model without SED parameter sampling.
### [(nu, hlr, e, beta)_bulge, (nu, hlr, e, beta)_disk, flux_bulge, flux_disk]
k_param_indices_no_SED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14]
k_spergel_subset = [1, 2, 3, 4, 5]

### Dictionary of settings for different examples
examples = {
    1: {'filter_ground': 'r',
        'filter_space':  'r', 
        'data_format':   'test_galsim_galaxy',
        'redshift':      1.0,
        'param_indices': k_param_indices_no_SED
        },
    2: {'filter_ground': 'r',
        'filter_space':  'y', 
        'data_format':   'test_galsim_galaxy',
        'redshift':      1.0,
        'param_indices': k_param_indices_no_SED
        },
    3: {'filter_ground': 'r',
        'filter_space':  'r', 
        'data_format':   'test_galsim_galaxy',
        'redshift':      1.0,
        'param_indices': None ### Sample all parameters, including SED types
        },
    4: {'filter_ground': 'r',
        'filter_space':  'y', 
        'data_format':   'test_galsim_galaxy',
        'redshift':      1.0,
        'param_indices': None ### Sample all parameters, including SED types
        }                
}

class RoasterArgs(object):
    """
    Arguments expected by the Roaster script
    """
    def __init__(self, example_num, file_lab="", epoch=None):
        self.outfile = '../output/roasting/example{:d}/roaster_out{}'.format(example_num, file_lab)
        self.epoch = epoch
        self.seed = 6199256
        self.nsamples = 1000
        self.nwalkers = 64
        self.nburn = 100
        self.nthreads = 1
        self.quiet = True


class RoasterInspectorArgs(object):
    def __init__(self, infile, outprefix):
        self.infile = infile
        self.keeplast = 500
        self.outprefix = outprefix
        self.truths = None
        

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
        self.nburn = 1 #200


def main():
    parser = argparse.ArgumentParser(
        description='Run the example scenarios for JIF paper 1.')
    
    parser.add_argument('--example_num', type=int, default=1,
                        help='Index of the example to run (1 -- 6; 0 means run all).')

    parser.add_argument('--galaxy_model_type', type=str, default="Spergel",
                        help='Type of galaxy model to simulation [Default: "Spergel"]')

    args = parser.parse_args()

    if args.example_num < 0 or args.example_num > 6:
        raise ValueError("example_num must be in the range [0, 6]")

    if args.example_num == 0:
        example_nums = range(1, 5)
    else:
        example_nums = [args.example_num]

    for ex_num in example_nums:
        logging.debug('\n-------------\nStarting run for JIF paper 1, example {:d}'.format(ex_num))
        ### Array index corresponding to the example number
        ex_ndx = ex_num - 1

        ### Label to attach to simulated image files and output plots
        file_lab = '_paper1_ex{:d}'.format(ex_num)

        outdir = "../output/roasting/example{:d}".format(ex_num)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        logging.debug('Generating simulated images')
        galsim_galaxy.make_test_images(
            filter_name_ground=examples[ex_num]['filter_ground'],
            filter_name_space=examples[ex_num]['filter_space'],
            file_lab=file_lab,
            galaxy_model=args.galaxy_model_type)

        ### Run Roaster for epochs 0, 1 individually and then combined (epoch == None)
        epochs = [0, 1, None]
        for epoch in epochs:
            logging.debug('===== Setting up Roaster')
            ### Reset the module-level pixel data list in Roaster.
            ### Otherwise, this list gets appended to under subsequent steps in the 'epoch' loop.
            Roaster.pixel_data = [] 
            roaster = Roaster.Roaster(data_format=examples[ex_num]['data_format'],
                lnprior_omega=Roaster.DefaultPriorBulgeDisk(z_mean=examples[ex_num]['redshift']),
                galaxy_model_type=args.galaxy_model_type,
                epoch=epoch,
                param_subset=[1, 2, 3, 4, 5], #examples[ex_num]['param_indices'],
                debug=False)
            roaster.Load("../TestData/test_image_data" + file_lab + ".h5")
            logging.debug('Running Roaster MCMC')
            roaster_args = RoasterArgs(ex_num, file_lab=file_lab, epoch=epoch)
            Roaster.do_sampling(roaster_args, roaster)

            # if epoch is not None:
            #     epoch_lab = '_epoch{:d}'.format(epoch)
            #     epoch_subdir = 'epoch{:d}'.format(epoch)
            # else:
            #     epoch_lab = ''
            #     epoch_subdir = ''
            # args_inspector = RoasterInspectorArgs(
            #     infile=os.path.join(outdir, 'roaster_out_paper1_ex{:d}{}.h5'.format(ex_num, epoch_lab)),
            #     outprefix=os.path.join(outdir, epoch_subdir))
            # inspector = RoasterInspector.RoasterInspector(args_inspector)
            # inspector.plot()

        ### TODO: Add DM wrapper here

        logging.debug('Making plots of posterior constraints for example {:d}'.format(ex_num))
        gpp_args = GPPArgs(ex_num, file_lab=file_lab)
        gpp = gpp_plot.GalaxyPosteriorPlot(gpp_args)
        gpp.plot()

        logging.debug('*** Finished run for JIF paper 1, example {:d}'.format(ex_num))
    return 0


if __name__ == "__main__":
    sys.exit(main())
