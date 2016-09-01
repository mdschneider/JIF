#!/usr/bin/env python
# encoding: utf-8
"""
sim_image_from_roaster_config.py

Simulate a stamp using the Roaster config inputs to set the parameters.

Use the Roaster method to render the model image - that way this simulation
should exactly match the models in Roaster during sampling.
"""
import os
import h5py
import jif

import logging


# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
#logging.basicConfig(filename='logs/Roaster.log',
#                     level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


def save_model_image(args, roaster):


def main():
    parser = argparse.ArgumentParser(
        description='Draw interim samples of source model parameters via MCMC.')

    parser.add_argument('--config_file', type=str, default=None,
                        help="Name of a configuration file listing inputs." +
                             "If specified, ignore other command line flags." +
                             "(Default: None)")

    args = parser.parse_args()

	logging.info('Reading from configuration file {}'.format(args.config_file))
	args = jif.Roaster.ConfigFileParser(args.config_file)

	roaster = Roaster(debug=args.debug,
					  lnprior_omega=Roaster.EmptyPrior(),
					  lnprior_Pi=Roaster.EmptyPrior(),
					  galaxy_model_type=args.galaxy_model_type,
					  telescope=args.telescope,
					  filters_to_load=args.filters,
					  achromatic_galaxy=args.achromatic)

	# -----------------------------------------------------
	# Set the member variables that are needed from the 
	# Load() method.
	# -----------------------------------------------------
	nimages = 1 ### Render only 1 stamp in 1 filter
	tel_names = [args.telescope]
	tel_dict = jif.telescopes.k_telescopes[args.telescope]
	pixel_scales = [tel_dict["pixel_scale"]]
	primary_diams = [tel_dict["primary_diam_meters"]]
	atmospheres = [tel_dict["atmosphere"]]
	psfs = []

	roaster._init_galaxy_models(nimages, tel_names, pixel_scales, primary_diams,
								atmospheres, psfs)

    if args.init_param_file is not None:
        roaster.initialize_param_values(args.init_param_file)

    save_model_image(args, roaster)
	return 0


if __name__ == "__main__":
	sys.exti(main())