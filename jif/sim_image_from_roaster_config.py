#!/usr/bin/env python
# encoding: utf-8
"""
sim_image_from_roaster_config.py

Simulate a stamp using the Roaster config inputs to set the parameters.

Use the Roaster method to render the model image - that way this simulation
should exactly match the models in Roaster during sampling.
"""
import argparse
import os
import sys
import copy
import numpy as np

import jif
import footprints

import logging


# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
#logging.basicConfig(filename='logs/Roaster.log',
#                     level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


def init_galaxy_models(args, roaster):
    """
    Set the member variables that are needed from the Load() method
    """
    nimages = 1 ### Render only 1 stamp in 1 filter
    roaster.num_epochs = nimages
    tel_names = [args.telescope]
    filter_name = args.filters[0]
    #
    tel_dict = jif.telescopes.k_telescopes[args.telescope]
    pixel_scales = [tel_dict["pixel_scale"]]
    primary_diams = [tel_dict["primary_diam_meters"]]
    atmospheres = [tel_dict["atmosphere"]]
    lam_over_diam = (tel_dict["filter_central_wavelengths"][filter_name] * 1e-9 /
                     tel_dict["primary_diam_meters"]) * 180*3600/np.pi

    psfs = [jif.PSFModel(telescope=args.telescope, achromatic=args.achromatic,
                         lam_over_diam=lam_over_diam, 
                         gsparams=None)]

    roaster._init_galaxy_models(nimages, tel_names, pixel_scales, primary_diams,
                                atmospheres, psfs)

    ### Set grid size to a constant angular extent
    stamp_size_arcsec = 20.0
    roaster.nx = [int(np.floor(stamp_size_arcsec / p)) for p in pixel_scales]
    roaster.ny = copy.copy(roaster.nx)
    return roaster


def save_model_image(args, roaster):
    model_image = roaster._get_model_image(iepochs=0, add_noise=False)

    if args.telescope.lower() == "lsst":
        noise_model = jif.telescopes.lsst_noise(args.seed)
    elif args.telescope.lower() == "wfirst":
        noise_model = jif.telescopes.wfirst_noise(args.seed)
    else:
        raise KeyError("Unsupported telescope name for noise model")
    model_image.addNoise(noise_model)

    tel_dict = jif.telescopes.k_telescopes[args.telescope]

    ### ========================================================================
    ### Save a FITS image using the GalSim 'write' method for Image() class
    ### ========================================================================
    outfile = args.outfile + "_model_image.fits"
    model_image.write(outfile)
    logging.debug('Wrote model image to %r', outfile)

    ### ========================================================================
    ### Save a PNG plot using matplotlib
    ### ========================================================================
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(6, 6))
    ### Model
    plt.subplot(1, 1, 1)
    plt.imshow(model_image.array, interpolation='none',
               cmap=plt.cm.pink)
    plt.colorbar()
    # plt.title("Model")

    outfile = args.outfile + "_model_image.png"
    plt.savefig(outfile, bbox_inches='tight')

    ### ========================================================================
    ### Save a PNG plot of the PSF using matplotlib
    ### ========================================================================
    psf_image = roaster.src_models[0][0].get_psf_image(
        filter_name=args.filters[0],
        ngrid=64, gain=1.0, add_noise=False)
    print "PSF HLR: {:12.10g}".format(psf_image.calculateHLR())

    outfile = args.outfile + "_psf_image.png"
    fig=plt.figure(figsize=(6,6))
    plt.imshow(psf_image.array, interpolation='none', cmap=plt.cm.pink)
    plt.colorbar()
    plt.savefig(outfile, bbox_inches='tight')

    ### ========================================================================
    ### Save a Footprints file
    ### ========================================================================
    dummy_mask = 1.0
    dummy_background = 1.0
    dummy_catalog = np.zeros((args.num_sources, 1))

    outfile = args.outfile + "_model_image.h5"
    f = footprints.Footprints(outfile)
    f.save_source_catalog(dummy_catalog, segment_index=0)
    f.save_images([model_image.array], [noise_model.getVariance()], 
                  [dummy_mask], [dummy_background],
                  segment_index=0,
                  telescope=args.telescope,
                  filter_name=args.filters[0])
    f.save_tel_metadata(telescope=args.telescope,
                        primary_diam=tel_dict["primary_diam_meters"],
                        pixel_scale_arcsec=tel_dict["pixel_scale"],
                        atmosphere=tel_dict["atmosphere"])
    f.save_psf_images([psf_image.array],
        segment_index=0,
        telescope=args.telescope,
        filter_name=args.filters[0])
    return None


def main(**kwargs):
    logging.info('Reading from configuration file {}'.format(kwargs['config_file']))
    args = jif.RoasterModule.ConfigFileParser(kwargs['config_file'])

    roaster = jif.Roaster(debug=args.debug,
                          lnprior_omega=jif.RoasterModule.EmptyPrior(),
                          lnprior_Pi=jif.RoasterModule.EmptyPrior(),
                          galaxy_model_type=args.galaxy_model_type,
                          telescope=args.telescope,
                          filters_to_load=args.filters,
                          achromatic_galaxy=args.achromatic)
    roaster.num_sources = args.num_sources
    roaster.filter_names = args.filters
    roaster.filters = jif.telescopes.load_filter_files(telescope_name=args.telescope)

    roaster = init_galaxy_models(args, roaster)

    if args.init_param_file is not None:
        roaster.initialize_param_values(args.init_param_file)

    save_model_image(args, roaster)
    return 0


def run():
    parser = argparse.ArgumentParser(
        description='Draw interim samples of source model parameters via MCMC.')

    parser.add_argument('config_file', type=str, #default="../config/roaster_defaults.cfg",
                        help="Name of a configuration file listing inputs." +
                             "If specified, ignore other command line flags." +
                             "(Default: None)")

    args = parser.parse_args()
    d = vars(args)

    main(**d)
    return 0


if __name__ == "__main__":
    sys.exit(run())
