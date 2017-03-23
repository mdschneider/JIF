#!/usr/bin/env python
# encoding: utf-8
"""
run_debug_roaster_bias.py

Run the tests of Roaster posterior biases.

Created by Michael Schneider on 2017-02-11
"""

import argparse
import sys
import os.path
import copy
import ConfigParser
import h5py
import numpy as np
import matplotlib.pyplot as plt

import galsim
import jif

import logging


# Print log messages to screen:
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
#logging.basicConfig(filename='logs/run_prob_shear_transform.log',
#                     level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')


def make_configs(cfg_name, cfg_params_name, image_file, roaster_file, params,
                 active_parameters,
                 galaxy_model="Spergel", nsamples=100):
    make_param_config(cfg_params_name, params)
    make_roaster_config(cfg_name, cfg_params_name, image_file, roaster_file,
                        active_parameters=active_parameters, 
                        galaxy_model=galaxy_model, nsamples=nsamples)
    return None


def make_param_config(cfg_name, params):
    logging.info("Making {}".format(cfg_name))

    cfgfile = open(cfg_name, 'w')

    Config = ConfigParser.ConfigParser()

    Config.add_section('parameters')
    Config.set('parameters', 'redshift', 1.0)
    Config.set('parameters', 'nu', params[0])
    # Config.set('parameters', 'nu', -0.6)
    Config.set('parameters', 'hlr', params[1])
    Config.set('parameters', 'e1', params[2])
    Config.set('parameters', 'e2', params[3])
    Config.set('parameters', 'mag_sed1', params[4])
    Config.set('parameters', 'dx', 0.0)
    Config.set('parameters', 'dy', 0.0)

    Config.write(cfgfile)
    cfgfile.close()


def make_roaster_config(cfg_name, cfg_params_name, image_file, roaster_file,
                        active_parameters='nu hlr e1 e2 mag_sed1 dx dy',
                        galaxy_model="Spergel", nsamples=100):
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
    Config.set('model', 'galaxy_model_type', galaxy_model)
    Config.set('model', 'model_params', active_parameters)
    # Config.set('model', 'model_params', 'e beta')
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


def get_noise_realization(gg, nx=80, ny=80, seed=590025467011):
    """
    @brief      Get a realization of the noise in each pixel of the model image
    @param      gg    an instance of GalSimGalaxyModel
    @return     The noise realization.
    """
    # im = gg.get_image()
    # nx, ny = im.array.shape
    noise = jif.telescopes.lsst_noise(random_seed=seed)
    im = galsim.Image(nx, ny, scale=gg.pixel_scale_arcsec, init_value=0.)
    im.addNoise(noise)
    return im.array, noise.getVariance()


def run_roaster(cfg_name):
    cfg = jif.RoasterModule.ConfigFileParser(cfg_name)
    rstr, rstr_args = jif.RoasterModule.InitRoaster(cfg)
    resid_rms = jif.RoasterModule.save_model_image(rstr_args, rstr)
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
    return (e + g) / (1.0 + g.conjugate() * e)


def unshear(e, g):
    return (e - g) / (1.0 - g.conjugate() * e)


def shear_params(p, g):
    e1 = p[0] * np.cos(2*p[1])
    e2 = p[0] * np.sin(2*p[1])

    e_int_c = complex(e1, e2)
    g_c = complex(g[0], g[1])
    e_sh = apply_shear(e_int_c, g_c)

    e = np.abs(e_sh)
    beta = np.mod(np.arctan2(e_sh.imag, e_sh.real) / 2., np.pi)
    return [e, beta]


def unshear_roaster_samples(dat, shear, e_ndx=0, beta_ndx=1):
    nsamples, nwalkers, nparams = dat.shape
    e = np.empty((nsamples, nwalkers), dtype=complex)
    e.real = dat[:, :, e_ndx] * np.cos(2. * dat[:, :, beta_ndx])
    e.imag = dat[:, :, e_ndx] * np.sin(2. * dat[:, :, beta_ndx])
    g = complex(shear[0], shear[1])
    e_int = unshear(e, g)
    e_int_mag = np.abs(e_int)
    e_int_beta = np.mod(np.arctan2(e_int.imag, e_int.real) / 2., np.pi)
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


class InspectorArgs(object):
    def __init__(self, infile, roaster_config):
        self.infile = infile + '_seg0_LSST.h5'
        self.roaster_config = roaster_config,
        self.segment_number = 0
        self.keeplast = 0


def run_meyers_test(nu, nx=64, ny=64, noise_var=3e-7, flux=1.0):
    """
    Duplicate the test by J. Meyers here:

    https://github.com/jmeyers314/silver-waffle/blob/master/Spergel%20Probability.ipynb
    """
    HLR = 1.0
    PSF = galsim.Kolmogorov(fwhm=0.6)
    scale = 0.2
    noise = galsim.GaussianNoise(sigma=np.sqrt(noise_var))

    def make_data(nu_val):
        gal = galsim.Spergel(nu_val, half_light_radius=HLR)
        gal = gal.withFlux(flux)
        obj = galsim.Convolve(PSF, gal)
        img = obj.drawImage(nx=nx, ny=ny, scale=scale)
        img.addNoise(noise)
        return img

    img = make_data(nu_val=0.5)

    def lnlike1(nu1):
        gal = galsim.Spergel(nu1, half_light_radius=HLR)
        obj = galsim.Convolve(PSF, gal)
        model = obj.drawImage(nx=nx, ny=ny, scale=scale)
        return -0.5*np.sum((model.array-img.array)**2/noise_var)

    def lnlikelihood(x):
        return [lnlike1(nu1) for nu1 in x]

    return lnlikelihood(nu)


def main():
    parser = argparse.ArgumentParser(
        description='Run Roaster posterior bias tests.')

    parser.add_argument('--nsamples', type=int, default=500,
        help="Number of MCMC samples (after burnin) [Default: 500]")

    parser.add_argument('--gal_model', type=str, default='Spergel',
        help="Galaxy model name for the simulated data")

    parser.add_argument('--seed', type=int, default=1325986508796,
        help="Seed for PRNG")

    args = parser.parse_args()

    logging.debug('Roaster bias test started')

    shear = [0.0, 0.0]

    ##
    ## Roaster parameters (for fitting the fake data)
    ##
    e_ndx = 2
    beta_ndx = 3    

    ##
    ## GalSimGalaxyModel parameters (for generating the fake data)
    ##
    topdir = 'output'

    if args.gal_model == "Spergel":
        active_parameters = ['nu', 'hlr', 'e1', 'e2', 'mag_sed1']
        params = [0.5, 1.0, 0.0, 0.0, 27.5]
        print "params:", params
    elif args.gal_model == "BulgeDisk":
        active_parameters = ['e1_bulge', 'e2_bulge', 'e1_disk', 'e2_disk']
        params = [0.1, 0.0, 0.3, 0.5236]
        print "params:", params
    else:
        raise KeyError("Unsupported galaxy model")

    ## 0. Generate the test image 
    image_file = os.path.join(topdir, 'roaster_model_image.h5')        

    ## 1. Run Roaster on the image with only 'nu' as an active parameter
    roaster_cfg_name = os.path.join(topdir, 'ap1', 'roaster_config.cfg')
    params_cfg_name = os.path.join(topdir, 'ap1', 'roaster_params.cfg')
    roaster_file = os.path.join(topdir, 'ap1', 'roaster')
    if not os.path.exists(os.path.join(topdir, 'ap1')):
        os.mkdir(os.path.join(topdir, 'ap1'))
    ## 
    make_configs(roaster_cfg_name, params_cfg_name,
                 image_file, roaster_file,
                 params=params,
                 active_parameters='nu',
                 galaxy_model=args.gal_model,
                 nsamples=args.nsamples)


    seed = args.seed
    fig = plt.figure(figsize=(8, 8/1.618))
    nu = np.linspace(0.2, 0.8, 160)
    nu_ml = 0.0
    nu_ml_jm = 0.0
    niter = 50
    for i_noise in xrange(niter):
        gg = jif.GalSimGalaxyModel(galaxy_model=args.gal_model,
            telescope_model="LSST",
            psf_model="Model",
            active_parameters=active_parameters)

        noise_realization, noise_var = get_noise_realization(gg, nx=80, ny=80, seed=seed)
        nx, ny = noise_realization.shape

        gg.set_params(params)
        out_image = galsim.Image(nx, ny, init_value=0.)
        gg.save_footprint(image_file, out_image=out_image, noise=noise_realization)

        ## Plot univariate Roaster posterior as a function of 'nu'
        cfg = jif.RoasterModule.ConfigFileParser(roaster_cfg_name)
        rstr, rstr_args = jif.RoasterModule.InitRoaster(cfg)
        # jif.RoasterModule.save_model_image(rstr_args, rstr)

        lnp = np.array([rstr([nu_i]) for nu_i in nu])
        plt.plot(nu, np.exp(lnp - np.max(lnp)), color="#348ABD", alpha=0.2)

        nu_ml += nu[np.argmax(lnp)]

        # flux = jif.parameters.flux_from_AB_mag(gg.params[0].mag_sed1)
        noise_var = 6e-8
        flux = 1.0
        lnp_jm = run_meyers_test(nu, nx=nx, ny=ny, noise_var=noise_var, flux=flux)
        plt.plot(nu, np.exp(lnp_jm - np.max(lnp_jm)), color="#7A68A6", alpha=0.2)
        nu_ml_jm += nu[np.argmax(lnp_jm)]

        seed += 1
    nu_ml /= niter
    nu_ml_jm /= niter
    plt.axvline(0.5, color='grey')
    plt.axvline(nu_ml, color='#348ABD', linestyle='dashed')
    plt.axvline(nu_ml_jm, color='#7A68A6', linestyle='dashed')
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"Pr($\nu$)")
    plt.savefig("nu_posterior.png")


    # ## 
    # run_roaster(roaster_cfg_name)

    # iargs = InspectorArgs(roaster_file, roaster_cfg_name)
    # inspector = jif.RoasterInspector(iargs)
    # inspector.plot()
    # inspector.plot_data_and_model()

    # ## 2. Run Roaster on the image with all parameters active
    # roaster_cfg_name = os.path.join(topdir, 'ap5', 'roaster_config.cfg')
    # params_cfg_name = os.path.join(topdir, 'ap5', 'roaster_params.cfg')
    # roaster_file = os.path.join(topdir, 'ap5', 'roaster')  
    # if not os.path.exists(os.path.join(topdir, 'ap5')):
    #     os.mkdir(os.path.join(topdir, 'ap5'))     
    # ## 
    # make_configs(roaster_cfg_name, params_cfg_name,
    #              image_file, roaster_file,
    #              params=params,
    #              active_parameters='nu hlr e1 e2 mag_sed1',
    #              galaxy_model=args.gal_model,
    #              nsamples=args.nsamples)

    # run_roaster(roaster_cfg_name)

    # iargs = InspectorArgs(roaster_file, roaster_cfg_name)
    # inspector = jif.RoasterInspector(iargs)
    # inspector.plot()
    # inspector.plot_data_and_model()

    logging.debug('Roaster bias test finished')
    return 0


if __name__ == "__main__":
    sys.exit(main())
