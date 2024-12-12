#!/usr/bin/env python
# encoding: utf-8
#
# Copyright (c) 2017, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory. Written by
# Michael D. Schneider schneider42@llnl.gov.
# LLNL-CODE-742321. All rights reserved.
#
# This file is part of JIF. For details, see https://github.com/mdschneider/JIF
#
# Please also read this link – Our Notice and GNU Lesser General Public License
# https://github.com/mdschneider/JIF/blob/master/LICENSE
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the terms and conditions of the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
'''
@file jiffy roaster.py

Draw posterior samples of image source model parameters
'''
import numpy as np
from tqdm import tqdm
import yaml
import argparse
import configparser
from multiprocessing import Pool
import h5py

import os
import sys

import galsim
from galsim.errors import GalSimFFTSizeError, GalSimError

import emcee

import jiffy
from . import priors, detections
import footprints

class Roaster(object):
    '''
    Likelihood model for footprint pixel data given a parametric source model

    Only single epoch images are allowed.
    '''
    def __init__(self, args):
        with open(args.config_file, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        self.num_sources = self.config['model']['num_sources']

        actv_params = self.config['model']['model_params'].split(' ')
        model_kwargs = dict({'active_parameters': actv_params})
        self.n_params = len(actv_params)

        model_class_name = self.config['model']['model_class']
        if 'model_module' in self.config['model']:
            model_module = __import__(self.config['model']['model_module'])
        else:
            model_module = __import__('jiffy.galsim_galaxy')
        self.src_models = [getattr(model_module, model_class_name)(self.config, **model_kwargs)
                           for i in range(self.num_sources)]

        if 'init' in self.config:
            if 'seed' in self.config['init']:
                np.random.seed(self.config['init']['seed'])
            if 'param_values' in self.config['init']:
                for param_name, param_value in self.config['init']['param_values'].items():
                    self.set_param_by_name(param_name, param_value)
        # This is decided at the beginning of roasting. True by default until then.
        self.good_initial_params = True

        self.init_prior(args)
        self.init_detection_correction(args)

        # Initialize objects describing the pixel data in a footprint
        self.wcs_matrix = None
        self.bounds_arr = None
        self.ngrid_x = 64
        self.ngrid_y = 64
        self.noise_var = 3e-10
        self.var_slope = None
        self.var_intercept = None
        self.scale = 0.2
        self.gain = 1.0
        self.data = None
        self.mask = None
        self.bkg = None
        self.lnnorm = self._set_like_lnnorm()
        self.load_and_import_data()

    def init_prior(self, args=None):
        # Parse config
        prior_form = None
        prior_module = None
        for arg_name in self.config['model']:
            if arg_name[:6] == 'prior_':
                if arg_name[6:] == 'form':
                    prior_form = self.config['model'][arg_name]
                if arg_name[6:] == 'module':
                    prior_module = self.config['model'][arg_name]
        prior_kwargs = dict()
        if 'prior' in self.config:
            for arg_name in self.config['prior']:
                prior_kwargs[arg_name] = self.config['prior'][arg_name]
        
        # Initialize the prior
        self.prior = priors.initialize_prior(prior_form, prior_module, args, **prior_kwargs)

    def init_detection_correction(self, args=None):
        # Parse the config
        detection_correction_form = None
        detection_correction_module = None
        for arg_name in self.config['model']:
            if arg_name[:21] == 'detection_correction_':
                if arg_name[21:] == 'form':
                    detection_correction_form = self.config['model'][arg_name]
                if arg_name[21:] == 'module':
                    detection_correction_module = self.config['model'][arg_name]
        detection_correction_kwargs = dict()
        if 'detection_correction' in self.config:
            for arg_name in self.config['detection_correction']:
                detection_correction_kwargs[arg_name] = self.config['detection_correction'][arg_name]
        
        self.detection_correction = detections.initialize_detection_correction(
            detection_correction_form, detection_correction_module, args, **detection_correction_kwargs)        

    def get_params(self):
        '''
        Make a flat array of active model parameters for all sources

        For use in MCMC sampling
        '''
        return np.array([m.get_params() for m in self.src_models]).ravel()

    def set_params(self, params):
        '''
        Set the active parameters for all sources from a flattened array
        '''
        valid_params = True
        for isrc in range(self.num_sources):
            imin = isrc * self.n_params
            imax = (isrc + 1) * self.n_params
            p_set = params[imin:imax]
            valid_params *= self.src_models[isrc].set_params(p_set)
        return valid_params

    def set_param_by_name(self, paramname, value):
        '''
        Set a galaxy or PSF model parameter by name

        Can pass a single value that will be set for all source models, or a
        list of length num_sources with unique values for each source.
        '''
        if hasattr(value, '__len__'):
            if len(value) == self.num_sources:
                for isrc, v in enumerate(value):
                    if np.issubdtype(type(v), np.floating):
                        self.src_models[isrc].set_param_by_name(paramname, v)
                    else:
                        raise ValueError('If passing iterable, each entry must be a number')
            else:
                raise ValueError('If passing iterable, must have length num_sources')
        elif np.issubdtype(type(value), np.floating):
            for isrc in range(self.num_sources):
                self.src_models[isrc].set_param_by_name(paramname, value)
        else:
            raise ValueError('Unsupported type for input value')
        return None

    def make_data(self, noise=None):
        '''
        Make fake data from the current stored galaxy model

        @param noise Specify custom noise model. Use GaussianNoise if not provided.
        '''
        image = self._get_model_image()
        if noise is None:
            if self.var_slope is not None:
                var_image = image * self.var_slope
                if self.var_intercept is not None:
                    var_image = var_image + self.var_intercept
                noise = galsim.VariableGaussianNoise(rng=None, var_image=var_image)
            elif np.issubdtype(type(self.noise_var), np.floating):
                noise = galsim.GaussianNoise(sigma=np.sqrt(self.noise_var))
            elif issubclass(type(self.noise_var), np.ndarray):
                noise = galsim.VariableGaussianNoise(rng=None, var_image=self.noise_var)
        image.addNoise(noise)
        self.data = image.array
        return image

    def draw(self):
        '''
        Draw simulated noiseless data from the likelihood function
        '''
        return self.make_data()

    def import_data(self, pix_dat_array, wcs_matrix, bounds_arr, noise_var,
                    var_slope, var_intercept, mask=1, bkg=0, scale=0.2, gain=1.0):
        '''
        Import the pixel data and noise variance for a footprint
        '''
        if pix_dat_array is not None:
            self.ngrid_y, self.ngrid_x = pix_dat_array.shape
        self.data = pix_dat_array
        self.wcs_matrix = wcs_matrix
        self.bounds_arr = bounds_arr
        self.noise_var = noise_var
        self.var_slope = var_slope
        self.var_intercept = var_intercept
        self.mask = mask
        self.bkg = bkg
        self.scale = scale
        self.gain = gain
        self.lnnorm = self._set_like_lnnorm()

    def load_and_import_data(self):
        dat, noise_var, var_slope, var_intercept = None, None, None, None
        mask, bkg, scale, gain = None, None, None, None
        wcs_matrix, bounds_arr = None, None
        
        def _load_array(item):
            if isinstance(item, str):
                item = np.load(item)
            return item
        if 'footprint' in self.config:
            fp = self.config['footprint']
            dat = _load_array(fp['image']) if 'image' in fp else None
            wcs_matrix = _load_array(fp['wcs_matrix']) if 'wcs_matrix' in fp else None
            bounds_arr = _load_array(fp['bounds']) if 'bounds' in fp else None
            noise_var = _load_array(fp['variance']) if 'variance' in fp else None
            var_slope = fp['var_slope'] if 'var_slope' in fp else None
            var_intercept = fp['var_intercept'] if 'var_intercept' in fp else None
            mask = _load_array(fp['mask']) if 'mask' in fp else None
            scale = _load_array(fp['scale']) if 'scale' in fp else None
            gain = _load_array(fp['gain']) if 'gain' in fp else None
            bkg = _load_array(fp['background']) if 'background' in fp else None
        
        if 'io' in self.config and 'infile' in self.config['io']:
            dat, noise_var, mask, bkg, scale, gain = footprints.load_image(self.config['io']['infile'],
                segment=args.footprint_number, filter_name=self.config['io']['filter'])
        
        self.import_data(dat, wcs_matrix, bounds_arr, noise_var, var_slope, var_intercept,
                         mask=mask, bkg=bkg, scale=scale, gain=gain)

    def initialize_param_values(self, param_file_name):
        '''
        Initialize model parameter values from config file
        '''
        param_config = configparser.RawConfigParser()
        param_config.read(param_file_name)

        params = param_config.items('parameters')
        for paramname, val in params:
            vals = str.split(val, ' ')
            if len(vals) > 1: ### Assume multiple sources
                fval = [float(v) for v in vals[0:self.num_sources]]
            else:
                fval = float(val)
            self.set_param_by_name(paramname, fval)
        return None

    def _get_model_image(self):
        # Set up a blank template image
        if self.wcs_matrix is not None:
            wcs = galsim.JacobianWCS(self.wcs_matrix[0, 0], self.wcs_matrix[0, 1],
                                     self.wcs_matrix[1, 0], self.wcs_matrix[1, 1])
            if self.bounds_arr is not None:
                bounds = galsim.BoundsI(*self.bounds_arr)
                model_image = galsim.ImageF(self.ngrid_x, self.ngrid_y,
                                            wcs=wcs, bounds=bounds, init_value=0.)
            else:
                model_image = galsim.ImageF(self.ngrid_x, self.ngrid_y,
                                            wcs=wcs, init_value=0.)
        elif self.bounds_arr is not None:
            bounds = galsim.BoundsI(*self.bounds_arr)
            model_image = galsim.ImageF(self.ngrid_x, self.ngrid_y,
                                        scale=self.scale, bounds=bounds, init_value=0.)
        else:
            model_image = galsim.ImageF(self.ngrid_x, self.ngrid_y,
                                        scale=self.scale, init_value=0.)
        
        # Try to draw all the sources on the template image
        for isrc in range(self.num_sources):
            if model_image is None: # Can happen if previous source could not render
                # Give up on rendering, as this parameter combination's likelihood cannot be rigorously evaluated
                break
            else:
                model_image = self.src_models[isrc].get_image(image=model_image,
                                                              gain=self.gain)
        
        return model_image

    def lnprior(self):
        '''
        Evaluate the log-prior of the model parameters
        '''
        try:
            res = self.prior(self.src_models)
        except:
            # Assign 0 probability to parameter combinations that produce an unhandled exception in prior evaluation
            print('Unhandled exception encountered in prior evaluation.')
            print('Assigning 0 prior probability to this parameter combination.')
            return -np.inf
        if not np.isfinite(res):
            print('Computed prior has the pathological value', res)
            print('Assigning 0 prior probability to this parameter combination.')
            return -np.inf
        
        return res

    def _set_like_lnnorm(self, model_image=None):
        if self.var_slope is not None:
            if model_image is None:
                model_image = self.data
            if model_image is None:
                return None
            var_image = model_image * self.var_slope
            if self.var_intercept is not None:
                var_image = var_image + self.var_intercept
            
            # This is -inf at locations where noise_var == 0
            logden = -0.5 * np.log(2 * np.pi * var_image)
            
            valid_pixels = var_image > 0
            if self.mask is not None:
                valid_pixels &= self.mask.astype(bool)
            if np.sum(valid_pixels) == 0:
                return 0
            
            lnnorm = np.sum(logden[valid_pixels])
            return float(lnnorm)
        
        # This is -inf at locations where noise_var == 0
        logden = -0.5 * np.log(2 * np.pi * self.noise_var)

        if issubclass(type(self.noise_var), np.ndarray) and self.noise_var.size > 1:
            # Using a per-pixel variance plane.
            # Treat pixels with nonpositive variance as masked.
            valid_pixels = self.noise_var > 0
            if self.mask is not None:
                valid_pixels &= self.mask.astype(bool)
            if np.sum(valid_pixels) == 0:
                return 0
            lnnorm = np.sum(logden[valid_pixels])
        
        else:
            # Using a constant variance over the entire image
            # Treat nonpositive variance as a mask for the entire image
            if self.noise_var <= 0:
                return 0
            elif self.mask is None:
                npix = self.ngrid_x * self.ngrid_y
            elif np.issubdtype(type(self.mask), np.number):
                npix = self.ngrid_x * self.ngrid_y * mask
            else:
                npix = np.sum(self.mask)
            if npix == 0:
                return 0
            lnnorm = npix * logden

        return float(lnnorm)

    def lnlike(self):
        '''
        Evaluate the log-likelihood of the pixel data in a footprint
        '''
        res = -np.inf

        try:
            model_image = self._get_model_image()
        except:
            # Assign 0 probability to parameter combinations that produce an unhandled exception in image rendering
            return -np.inf
        
        if model_image is not None:
            # Compute log-likelihood assuming independent Gaussian-distributed noise in each pixel
            delta = model_image.array - self.data

            variance = self.noise_var
            lnnorm = self.lnnorm
            if self.var_slope is not None:
                lnnorm = self._set_like_lnnorm(model_image)
                variance = self.var_slope * model_image.array
                if self.var_intercept is not None:
                    variance = variance + self.var_intercept
            
            elif issubclass(type(variance), np.ndarray) and variance.size > 1:
                # Using a per-pixel variance plane
                # Treat zero variance pixels as masked
                valid_pixels = variance > 0
                if self.mask is not None:
                    valid_pixels &= self.mask.astype(bool)
                if np.sum(valid_pixels) == 0:
                    return 0
                sum_chi_sq = np.sum(delta[valid_pixels]**2 / variance[valid_pixels])
            
            else:
                # Using a constant variance over the entire image
                # Treat nonpositive variance as a mask for the entire image
                if variance <= 0:
                    return 0
                if self.mask is None:
                    sum_chi_sq = np.sum(delta**2) / variance
                else:
                    if np.sum(self.mask) == 0:
                        return 0
                    sum_chi_sq = np.sum(delta[self.mask.astype(bool)]**2) / variance
            
            res = -0.5 * sum_chi_sq + lnnorm
            if not np.isfinite(res):
                return -np.inf
        
        if self.detection_correction:
            # Scale up the likelihood to account for the fact that we're only
            # looking at data examples that pass a detection algorithm
            try:
                detection_correction = self.detection_correction(
                    model_image.array, self.src_models, variance, self.mask)
            except:
                # Assign 0 probability to parameter combinations that produce an unhandled exception in detection correction evaluation
                return -np.inf
            if not np.isfinite(detection_correction):
                return -np.inf
            res += detection_correction
        
        res = float(res)
        return res

    def __call__(self, params):
        # Assign 0 probability to invalid parameter combinations
        lnp = -np.inf

        # Set the Roaster params to the given values
        valid_params = self.set_params(params)
        if valid_params:
            # Compute the log-likelihood and log-prior,
            # using the newly-set Roaster params.
            lnp = self.lnlike()
            lnp += self.lnprior()

        return lnp


def init_roaster(args):
    '''
    Instantiate Roaster object and initialize parameters
    '''
    rstr = Roaster(args)

    if 'init' in rstr.config and 'init_param_file' in rstr.config['init']:
        rstr.initialize_param_values(rstr.config['init']['init_param_file'])

    return rstr

def run_sampler(args, sampler, p0, nsamples, rstr):
    burned_in_state = p0
    nburn = rstr.config['sampling']['nburn']
    if nburn > 0:
        if args.verbose:
            print('Burning in')
        burned_in_state = sampler.run_mcmc(p0, nburn, progress=args.show_progress_bar)
        sampler.reset()
    if args.verbose:
        print('Sampling')
    final_state = sampler.run_mcmc(burned_in_state, nsamples, progress=args.show_progress_bar)
    pps = sampler.get_chain()
    lnps = sampler.get_log_prob()
    return pps, lnps

def do_sampling(args, rstr, return_samples=False, write_results=True, moves=None):
    '''
    Execute MCMC sampling for posterior model inference
    '''
    omega_interim = rstr.get_params()
    if not np.isfinite(rstr(omega_interim)):
        rstr.good_initial_params = False
        if args.verbose:
            print('Bad initial chain parameters.')

    nvars = len(omega_interim)
    nsamples = rstr.config['sampling']['nsamples']
    nwalkers = rstr.config['sampling']['nwalkers']

    p0 = emcee.utils.sample_ball(omega_interim, 
                                 np.ones_like(omega_interim) * 0.01, nwalkers)

    if args.unparallelize:
        sampler = emcee.EnsembleSampler(nwalkers, nvars, rstr, moves=moves)
        pps, lnps = run_sampler(args, sampler, p0, nsamples, rstr)
    else:
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, nvars, rstr, moves=moves, pool=pool)
            pps, lnps = run_sampler(args, sampler, p0, nsamples, rstr)
    
    if args.cluster_walkers:
        pps, lnps = cluster_walkers(pps, lnps,
            thresh_multiplier=args.cluster_walkers_thresh)

    if write_results:
        write_to_h5(args, pps, lnps, rstr)
    if return_samples:
        return pps, lnps
    else:
        return None

def cluster_walkers(pps, lnps, thresh_multiplier=1):
    '''
    Down-select emcee walkers to those with the largest mean posteriors

    Follows the algorithm of Hou, Goodman, Hogg et al. (2012)
    '''
    # print("Clustering emcee walkers with threshold multiplier {:3.2f}".format(
    #       thresh_multiplier))
    pps = np.array(pps)
    lnps = np.array(lnps)
    ### lnps.shape == (Nsteps, Nwalkers) => lk.shape == (Nwalkers,)
    lk = -np.mean(np.array(lnps), axis=0)
    nwalkers = len(lk)
    ndx = np.argsort(lk)
    lks = lk[ndx]
    d = np.diff(lks)
    thresh = np.cumsum(d) / np.arange(1, nwalkers)
    selection = d > (thresh_multiplier * thresh)
    if np.any(selection):
        nkeep = np.argmax(selection)
    else:
        nkeep = nwalkers
    # print("pps, lnps:", pps.shape, lnps.shape)
    pps = pps[:, ndx[0:nkeep], :]
    lnps = lnps[:, ndx[0:nkeep]]
    # print("New pps, lnps:", pps.shape, lnps.shape)
    return pps, lnps

def write_to_h5(args, pps, lnps, rstr):
    '''
    Save an HDF5 file with posterior samples from Roaster
    '''
    outfile = rstr.config['io']['roaster_outfile'] + '_seg{:d}.h5'.format(args.footprint_number)
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    hfile = h5py.File(outfile, 'w')

    ### Store outputs in an HDF5 (sub-)group so we don't always
    ### need a separate HDF5 file for every segment.
    group_name = 'Samples/footprint{:d}'.format(args.footprint_number)
    grp = hfile.create_group(group_name)

    paramnames = rstr.config['model']['model_params'].split()
    if rstr.num_sources > 1:
        paramnames = [p + '_src{:d}'.format(isrc) for isrc in range(rstr.num_sources)
                      for p in paramnames]

    ## Write the MCMC samples and log probabilities
    if 'post' in grp:
        del grp['post']
    post = grp.create_dataset('post',
                              data=np.transpose(np.dstack(pps), [2, 0, 1]))
    # pnames = np.array(rstr.src_models[0][0].paramnames)
    post.attrs['paramnames'] = paramnames
    if 'logprobs' in grp:
        del grp['logprobs']
    _ = grp.create_dataset('logprobs', data=np.vstack(lnps))
    hfile.close()
    return None


def initialize_arg_parser():
    parser = argparse.ArgumentParser(
        description='Draw interim samples of source model parameters via MCMC.')

    parser.add_argument('--config_file', type=str,
                        default='../config/jiffy.yaml',
                        help='Name of a configuration file listing inputs.')

    parser.add_argument('--footprint_number', type=int, default=0,
                        help='The footprint number to load from input.')

    parser.add_argument('--unparallelize', action='store_true',
                        help='Disable parallelizing during sampling.' +
                        ' Usually need to do this if running multiple separate fits in parallel.')

    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose messaging.')
    parser.add_argument('--show_progress_bar', action='store_true',
                        help='Show progress bar.')
       
    parser.add_argument('--cluster_walkers', action='store_true',
                        help='Throw away outlier walkers.')
    parser.add_argument('--cluster_walkers_thresh', type=float, default=4,
                        help='Threshold multiplier for throwing away walkers.')

    return parser


def main():
    parser = initialize_arg_parser()
    args = parser.parse_args()

    rstr = Roaster(args)
    do_sampling(args, rstr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
