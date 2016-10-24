#!/usr/bin/env python
# encoding: utf-8
"""
run_prob_shear_transform.py

Run the tests of galaxy model probability distribution transformation under
shear.

Created by Michael Schneider on 2016-10-22
"""

import argparse
import sys
import os.path
import ConfigParser
import h5py
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kde import kdensity

import jif

import logging


# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
#logging.basicConfig(filename='logs/run_prob_shear_transform.log',
#                     level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


def make_configs(cfg_name, cfg_params_name, image_file, roaster_file, shear, 
                 nsamples=100):
    make_param_config(cfg_params_name, shear)
    make_roaster_config(cfg_name, cfg_params_name, image_file, roaster_file, 
                        nsamples)
    return None


def make_param_config(cfg_name, shear):
    logging.info("Making {}".format(cfg_name))
    e_int = [0.2, 0.0]

    e_int_c = complex(e_int[0], e_int[1])
    g = complex(shear[0], shear[1])
    e_sh = apply_shear(e_int_c, g)

    e = np.abs(e_sh)
    beta = np.arctan2(e_sh.imag, e_sh.real) / 2.

    cfgfile = open(cfg_name, 'w')

    Config = ConfigParser.ConfigParser()

    Config.add_section('parameters')
    Config.set('parameters', 'redshift', 1.0)
    Config.set('parameters', 'nu', 0.3)
    Config.set('parameters', 'hlr', 1.2)
    Config.set('parameters', 'e', e)
    Config.set('parameters', 'beta', beta)
    Config.set('parameters', 'mag_sed1', 25)
    Config.set('parameters', 'dx', 0.0)
    Config.set('parameters', 'dy', 0.0)

    Config.write(cfgfile)
    cfgfile.close()


def make_roaster_config(cfg_name, cfg_params_name, image_file, roaster_file, 
                        nsamples=100):
    logging.info("Making {}".format(cfg_name))
    cfgfile = open(cfg_name, 'w')

    Config = ConfigParser.ConfigParser()

    Config.add_section('infiles')
    Config.set('infiles', 'infile_1', image_file)

    Config.add_section('metadata')
    Config.set('metadata', 'outfile', roaster_file)

    Config.add_section('data')
    Config.set('data', 'data_format', 'jif_segment')
    Config.set('data', 'segment_number', 0)
    Config.set('data', 'telescope', 'LSST')
    Config.set('data', 'filters', 'r')
    Config.set('data', 'epoch_num', -1)

    Config.add_section('model')
    Config.set('model', 'galaxy_model_type', 'Spergel')
    # Config.set('model', 'model_params', 'nu hlr e beta mag_sed1 dx dy')
    Config.set('model', 'model_params', 'e beta')
    Config.set('model', 'num_sources', 1)
    Config.set('model', 'achromatic', True)

    Config.add_section('init')
    Config.set('init', 'init_param_file', cfg_params_name)
    Config.set('init', 'seed', 324876155)

    Config.add_section('run')
    Config.set('run', 'quiet', True)
    Config.set('run', 'debug', False)
    Config.set('run', 'output_model', False)

    Config.add_section('sampling')
    Config.set('sampling', 'sampler', 'emcee')
    Config.set('sampling', 'nsamples', nsamples)
    Config.set('sampling', 'nwalkers', 32)
    Config.set('sampling', 'nburn', 200)
    Config.set('sampling', 'nthreads', 1)

    Config.write(cfgfile)
    cfgfile.close()
    return None


def run_roaster(cfg_name):
    cfg = jif.RoasterModule.ConfigFileParser(cfg_name)
    rstr, rstr_args = jif.RoasterModule.InitRoaster(cfg)
    jif.do_roaster_sampling(rstr_args, rstr, sampler=cfg.sampler)
    return None


def load_roaster_samples(roaster_file, segment_number=0):
    f = h5py.File(roaster_file + '_seg0_LSST.h5', 'r')
    g = f['Samples/seg{:d}'.format(segment_number)]
    dat = g['post'][...]
    lnp = g['logprobs'][...]
    paramnames = g['post'].attrs['paramnames']
    print "Input parameter names:", paramnames
    f.close()
    return dat, lnp


def apply_shear(e, g):
    return (e + g) #/ (1.0 + g.conjugate() * e)
    # return e


def unshear(e, g):
    return (e - g) #/ (1.0 - g.conjugate() * e)
    # return e


def unshear_roaster_samples(dat, shear, e_ndx=0, beta_ndx=1):
    nsamples, nwalkers, nparams = dat.shape
    e = np.empty((nsamples, nwalkers), dtype=complex)
    e.real = dat[:, :, e_ndx] * np.cos(2. * dat[:, :, beta_ndx])
    e.imag = dat[:, :, e_ndx] * np.sin(2. * dat[:, :, beta_ndx])
    g = complex(shear[0], shear[1])
    e_int = unshear(e, g)
    e_int_mag = np.abs(e_int)
    e_int_beta = np.arctan2(e_int.imag, e_int.real) / 2.
    dat[:, :, e_ndx] = e_int_mag
    dat[:, :, beta_ndx] = e_int_beta
    return dat


def make_pdf_plots(dat_nosh, dat_wsh, e_ndx=0, beta_ndx=1):
    fig = plt.figure(figsize=(6, 2 * 6 / 1.618))

    print dat_nosh.shape
    print dat_wsh.shape

    ## |e|
    dens_nosh, x_nosh, bw = kdensity(dat_nosh[:, :, e_ndx], retgrid=True)
    dens_wsh, x_wsh, bw = kdensity(dat_wsh[:, :, e_ndx], retgrid=True)
    ax = plt.subplot(2, 1, 1)
    ax.plot(x_nosh, dens_nosh, label="no shear")
    ax.plot(x_wsh, dens_wsh, label="with shear")
    ax.legend()
    ax.set_xlabel(r"$|e|$")

    ## beta
    dens_nosh, x_nosh, bw = kdensity(dat_nosh[:, :, beta_ndx], retgrid=True)
    dens_wsh, x_wsh, bw = kdensity(dat_wsh[:, :, beta_ndx], retgrid=True)
    ax = plt.subplot(2, 1, 2)
    ax.plot(x_nosh, dens_nosh)
    ax.plot(x_wsh, dens_wsh)
    ax.set_xlabel(r"$\beta$")

    fig.tight_layout()
    fig.savefig("output/density_comparison.png", bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(
        description='Run probability shear transform tests.')

    parser.add_argument('--nsamples', type=int, default=100,
        help="Number of MCMC samples (after burnin) [Default: 100]")

    args = parser.parse_args()

    logging.debug('Shear probability transform test started')

    shear = [0.0, 0.0]

    topdir = 'output'

    roaster_cfg_name_noshear = os.path.join(topdir, 'roaster_config_noshear.cfg')
    params_cfg_name_noshear = os.path.join(topdir, 'roaster_params_noshear.cfg')
    image_file_noshear = os.path.join(topdir, 'roaster_noshear_model_image.h5')
    roaster_file_noshear = os.path.join(topdir, 'roaster_noshear')

    roaster_cfg_name_wshear = os.path.join(topdir, 'roaster_config_wshear.cfg')
    params_cfg_name_wshear = os.path.join(topdir, 'roaster_params_wshear.cfg')
    image_file_wshear = os.path.join(topdir, 'roaster_wshear_model_image.h5')
    roaster_file_wshear = os.path.join(topdir, 'roaster_wshear')

    ## 0. Generate the test image (without shear)
    make_configs(roaster_cfg_name_noshear, params_cfg_name_noshear,
                 image_file_noshear, roaster_file_noshear,
                 shear=[0., 0.], nsamples=args.nsamples)
    jif.sim_image_from_roaster_config.main(config_file=roaster_cfg_name_noshear)

    ## 1. Run Roaster on the (unsheared) image
    run_roaster(roaster_cfg_name_noshear)

    ## 2. Shear the image
    make_configs(roaster_cfg_name_wshear, params_cfg_name_wshear,
                 image_file_wshear, roaster_file_wshear,
                 shear=shear, nsamples=args.nsamples)
    jif.sim_image_from_roaster_config.main(config_file=roaster_cfg_name_wshear)

    ## 3. Run Roaster on the sheared image
    run_roaster(roaster_cfg_name_wshear)

    ## 4. Unshear the Roaster output parameter samples from step 3.
    dat_wsh, lnp_wsh = load_roaster_samples(roaster_file_wshear)
    dat_wsh = unshear_roaster_samples(dat_wsh, shear)

    ## 5. Compare galaxy parameter marginal densities from steps 1 & 4
    dat_nosh, lnp_nosh = load_roaster_samples(roaster_file_noshear)
    make_pdf_plots(dat_nosh, dat_wsh)

    logging.debug('Shear probability transform test finished')
    return 0


if __name__ == "__main__":
    sys.exit(main())