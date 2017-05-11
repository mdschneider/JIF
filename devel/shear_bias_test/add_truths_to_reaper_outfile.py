#!/usr/bin/env python
# encoding: utf-8
"""
@file add_truths_to_reaper_outfile.py

Append the true parameter values to the outfile from Thresher for use in the 
ThreshingInspector

Created on 2017-0420
"""

import argparse
import os
import numpy as np
import h5py
import configparser
import plot_shears


def main():
	parser = argparse.ArgumentParser(
        description="Add true parameter values to Thresher outfile")

	parser.add_argument("--field_num", type=int, default=0,
						help="Index of the simulation field to edit")

	parser.add_argument("--prior_type", type=int, default=0,
                        help="Type of hyperprior used in Thresher (0 or 1)")

	args = parser.parse_args()

	true_shears = plot_shears.get_true_field_shears(args.field_num)
	true_kappa = 0.0

	config = configparser.ConfigParser()
	config.read("thresher_params.ini")
	true_sigma_e_sq = float(config['const_ellip_dist']['sigma_e_sq'])

	truths = [true_shears[0], true_shears[1], true_kappa]
	paramnames = ['g1', 'g2', 'kappa']
	if args.prior_type == 0:
		truths.append(true_sigma_e_sq)
		paramnames.append('sigma_e_sq')

	thresher_file = os.path.join(plot_shears.k_thresher_topdir,
		"thresher_{0:0>3}_galdist{1:d}.h5".format(args.field_num,
			args.prior_type))

	f = h5py.File(thresher_file, 'a')
	if "truths" in f:
		del f["truths"]
	truths_dat = f.create_dataset("truths", data=truths)
	truths_dat.attrs['paramnames'] = paramnames
	f.close()	

if __name__ == "__main__":
	main()