#!/usr/bin/env python
# encoding: utf-8
"""
sheller_great3.py

Created by Michael Schneider on 2015-10-16
"""

import argparse
import sys
import os.path
import numpy as np
import copy
#import pandas as pd
#import matplotlib.pyplot as plt

from astropy.io import fits
# import jif.segments as segments
import jif.galsim_galaxy as gg
import footprints

import logging

# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
#logging.basicConfig(filename='logs/sheller_great3.log',
#                     level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


### Number of pixels per galaxy postage stamp, per dimension
k_g3_ngrid = {"ground": 48, "space": 96}
### Pixel scales in arcseconds
k_g3_pixel_scales = {"ground": 0.2, "space_single_epoch": 0.05,
                     "space_multi_epoch": 0.1}
### Guess what values were used to simulate optics PSFs
k_g3_primary_diameters = {"ground": 8.2, "space": 2.4}
### Guess that GREAT3 used LSST 'r' band to render images
k_filter_name = 'r'
k_filter_central_wavelengths = {'r':620.}

## Original GREAT3 types:
# k_input_cat_type = [('x', '>i8'), ('y', '>i8'), ('ID', '>i8')]
## 2016 GREAT3 simulation scripts:
k_input_cat_type = [('obj_num', '>i8'), ('x', '<f8'), ('y', '<f8'), 
                    ('dx', '<f8'), ('dy', '<f8'), ('psf_e1', '<f8'), 
                    ('psf_e2', '<f8'), ('psf_fwhm', '<f8'), ('g1', '<f8'), 
                    ('g2', '<f8'), ('gal_e1', '<f8'), ('gal_e2', '<f8')]


def get_background_and_noise_var(data, clip_n_sigma=3, clip_converg_tol=0.1,
    verbose=False):
    """
    Determine the image background level.

    clip_n_sigma = Number of standard deviations used to define outliers to
        the assumed Gaussian random noise background.
    convergence_tol = the fractional tolerance that must be met before
        iterative sigma clipping proceedure is terminated.

    This is currently largely based on the SExtractor method, which is in
    turn based on the Da Costa (1992) method. Currently the entire image is
    used in the background estimation proceedure but you could imaging a
    gridded version of the following which could account for background
    variation across the image.
    TODO: Create a background image instead of just a background value.
    """
    # Inatilize some of the iterative parameters for the while statement.
    sigma_frac_change = clip_converg_tol + 1
    i = 0
    #
    x = np.copy(data.ravel())
    # Calculate the median and standard deviation of the initial image
    x_median_old = np.median(x)
    x_std_old = np.std(x)
    # Iteratively sigma clip the pixel distribution.
    while sigma_frac_change > clip_converg_tol:
        # Mask pixel values
        mask_outliers = np.logical_and(x >= x_median_old -
                                       clip_n_sigma*x_std_old,
                                       x <= x_median_old +
                                       clip_n_sigma*x_std_old)
        # Clip the data.
        x = x[mask_outliers]
        x_std_new = np.std(x)
        # Check percent difference between latest and previous standard
        # deviation values.
        sigma_frac_change = np.abs(x_std_new-x_std_old)/((x_std_new+x_std_old)/2.)
        if verbose:
            print 'Masked {0} outlier values from this iteration.'.format(
                np.sum(~mask_outliers))
            print 'Current fractional sigma change between clipping iterations = {0:0.2f}'.format(sigma_frac_change)
        # Replace old values with estimates from this iteration.
        x_std_old = x_std_new.copy()
        x_median_old = np.median(x)
        # Make sure that we don't have a run away while statement.
        i += 1
        if i > 100:
            print 'Background variance failed to converge after 100 sigma clipping iterations, exiting.'
            sys.exit()
    # Calculate the clipped image median.
    x_clip_median = np.median(x)
    # Calculate the clipped image mean.
    x_clip_mean = np.mean(x)
    # Estimate the clipped image mode (SExtractor's version of Da Costa 1992).
    # This is the estimate of the image background level.
    background = float(2.5 * x_clip_median - 1.5 * x_clip_mean)
    # Calculate the standard deviation of the pixel distribution
    noise_var = float(np.var(x))
    return background, noise_var


def create_segments(subfield_index=0, experiment="control",
    observation_type="ground", shear_type="constant",
    data_path="./", catfile_head="galaxy_catalog",
    n_gals=10000, verbose=False):
    """
    Load pixel data and PSFs for all epochs for a given galaxy
    """
    if subfield_index < 0 or subfield_index > 199:
        raise ValueError("subfield_index must be in range [0, 199]")

    ### Reconstruct the GREAT3 image input file path and name
    # indir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
    #     "../data/great3", experiment, observation_type, shear_type)
    indir = os.path.join(data_path, experiment, observation_type, shear_type)

    ### Collect input image filenames for all epochs.
    ### FIXME: Get correct 'experiment' names here
    if experiment in ["control", "real_galaxy", "variable_PSF"]:
        nepochs = 1
    else:
        nepochs = 6
    infiles = []
    starfiles = []
    for epoch_index in xrange(nepochs):
        infiles.append(os.path.join(indir,
            "image-{:03d}-{:d}.fits".format(subfield_index, epoch_index)))
        starfiles.append(os.path.join(indir,
            "starfield_image-{:03d}-{:d}.fits".format(subfield_index, epoch_index)))
    if verbose:
        print "input files:", infiles

    ### Load the galaxy catalog for this subfield
    f = fits.open(os.path.join(indir,
        catfile_head + "-{:03d}-{:d}.fits".format(subfield_index, epoch_index)))
    # print np.asarray(f[1].data).shape
    # assert len(f[1].data[0]) == len(k_input_cat_type), "Wrong length for catalog types"    
    # gal_cat = np.core.records.array(np.asarray(f[1].data), 
    #                                 dtype=k_input_cat_type)
    # gal_cat = np.rec.array(np.asarray(f[1].data), dtype=k_input_cat_type)
    gal_cat = copy.copy(f[1].data)
    f.close()

    ### Specify the output filename for the Segments
    segdir = os.path.join(indir, "segments")
    if not os.path.exists(segdir):
        os.makedirs(segdir)
    seg_filename = os.path.join(segdir, "seg_{:03d}.h5".format(subfield_index))
    if verbose:
        print "seg_filename:", seg_filename

    ## Set some common metadata required by the Segment file structure
    ## The telescope name must match a model in the JIF telescopes.py module. 
    # telescope_name = "GREAT3_{}".format(observation_type)
    telescope_name = {"ground": "LSST", "space": "WFIRST"}[observation_type]
    if verbose:
        print "telescope_name:", telescope_name
    filter_name = k_filter_name
    dummy_mask = 1.0
    dummy_background = 0.0

    ### Create and fill the elements of the segment file for all galaxies
    ### in the current sub-field. Different sub-fields go in different segment
    ### files (no particular reason for this - just seems convenient).
    seg = footprints.Footprints(seg_filename)

    ## Get the noise and background for the entire image, before iterating 
    ## over galaxy stamps. This is *much* faster than separate noise estimates
    ## for each stamp 
    noise_vars = []
    backgrounds = []
    for ifile, infile in enumerate(infiles): # Iterate over epochs
        f = fits.open(infile)
        bkgrnd, noise_var = get_background_and_noise_var(f[0].data)
        noise_vars.append(noise_var)
        backgrounds.append(bkgrnd)
        # print "empirical nosie variance: {:5.4g}".format(np.var(f[0].data))
        f.close()

    ### There are 1e4 galaxies in one GREAT3 image file.
    ### Save all images to the segment file, but with distinct 'segment_index'
    ### values.
    n_gals_image_file = n_gals
    ngals_per_dim = 100 ### The image is a 100 x 100 grid of galaxies
    for igal in xrange(n_gals_image_file):
        if verbose and np.mod(igal, 2) == 0:
            print "Galaxy {:d} / {:d}".format(igal+1, n_gals_image_file)
        ### Specify input image grid ranges for this segment
        ng = k_g3_ngrid[observation_type]
        i, j = np.unravel_index(igal, (ngals_per_dim, ngals_per_dim))
        ### These lines define the order in which we sort the postage stamps
        ### into segment indices:
        xmin = j * ng
        xmax = (j+1) * ng
        ymin = i * ng
        ymax = (i+1) * ng

        images = []
        psfs = []
        for ifile, infile in enumerate(infiles): # Iterate over epochs, 
                                                 # select same galaxy
            f = fits.open(infile)
            images.append(np.asarray(f[0].data[ymin:ymax, xmin:xmax],
                dtype=np.float64))
            f.close()

            s = fits.open(starfiles[ifile])
            ### Select just the perfectly centered star image for the PSF model.
            ### There are 8 other postage stamps (for constant PSF branches)
            ### that have offset star locations with respect to the pixel grid.
            ### It's not clear we need these for JIF modeling until we're
            ### marginalizing the PSF model.
            psfs.append(np.asarray(s[0].data[0:ng, 0:ng], dtype=np.float64))
            s.close()

        # print "noise_vars:", noise_vars
        seg.save_images(images, noise_vars, [dummy_mask], backgrounds,
            segment_index=igal, telescope=telescope_name)
        seg.save_psf_images(psfs, segment_index=igal, telescope=telescope_name,
            filter_name=filter_name, model_names=None)
        # seg.save_source_catalog(np.reshape(gal_cat[igal], (1,)),
        #     segment_index=igal)
        seg.save_source_catalog(np.asarray(gal_cat[igal]),
            segment_index=igal)

    ### It's not strictly necessary to instantiate a GalSimGalaxyModel object
    ### here, but it's useful to do the parsing of existing bandpass files to
    ### copy filter curves into the segment file.
    # gg_obj = gg.GalSimGalaxyModel(telescope_name=telescope_name,
    #     filter_names=list(filter_name), filter_wavelength_scale=1.0)
    # gg.save_bandpasses_to_segment(seg, gg_obj, k_filter_name, telescope_name,
    #     scale={"LSST": 1.0, "WFIRST": 1.e3}[telescope_name])

    seg.save_tel_metadata(telescope=telescope_name,
                          primary_diam=k_g3_primary_diameters[observation_type],
                          pixel_scale_arcsec=k_g3_pixel_scales[observation_type],
                          atmosphere=(observation_type == "ground"))
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Parse GREAT3 data file and save as galaxy segment files.')

    parser.add_argument('--subfield_index', type=int, default=0,
                        help="GREAT3 data set subfield index (0-199) " +
                             "[Default: 0]")

    parser.add_argument('--experiment', type=str, default="control",
                        help="GREAT3 'experiment' name. [Default: control]")

    parser.add_argument('--observation_type', type=str, default="ground",
                        help="GREAT3 'observation' type (ground/space) " +
                             "[Default: ground]")

    parser.add_argument('--shear_type', type=str, default="constant",
                        help="GREAT shear type [Default: constant]")

    parser.add_argument('--data_path', type=str, default="./",
                        help="Path to the directory with input images " +
                             "[Default: ./]")

    parser.add_argument('--catfile_head', type=str, default="galaxy_catalog",
                        help="File name head for the input galaxy catalog " +
                             "[Default: galaxy_catalog]")

    parser.add_argument('--n_gals', type=int, default=10,
                        help="How many galaxies to process from a sub-field? " +
                             "[Default: 10]")

    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose messaging")

    args = parser.parse_args()
    logging.debug('Creating footprints file for subfield {:d}'.format(
        args.subfield_index))


    create_segments(subfield_index=args.subfield_index,
                    experiment=args.experiment,
                    observation_type=args.observation_type,
                    shear_type=args.shear_type,
                    data_path=args.data_path,
                    catfile_head=args.catfile_head,
                    n_gals=args.n_gals,
                    verbose=args.verbose)

    logging.debug('Finished creating footprints for subfield {:d}'.format(
        args.subfield_index))
    return 0


if __name__ == "__main__":
    sys.exit(main())
