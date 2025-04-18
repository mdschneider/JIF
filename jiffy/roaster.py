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
from . import priors, likelihoods, detections


class Roaster(object):
    '''
    Likelihood model for footprint pixel data given a parametric source model

    Only single epoch images are allowed.
    '''
    def __init__(self, args):
        # Load config
        self.config = None
        # Whenever a config object is supplied directly, use that.
        if 'config' in vars(args):
            self.config = args.config
        # If no config object is supplied directly,
        # read a config file from a given location.
        if self.config is None and 'config_file' in vars(args):
            with open(args.config_file, 'r') as config_file:
                self.config = yaml.safe_load(config_file)
        assert self.config is not None, 'No config given for new Roaster object.'

        self.init_model()
        # This is decided at the beginning of roasting. True by default until then.
        self.good_initial_params = True

        self.init_prior(args)
        self.init_likelihood(args)
        self.init_detection_correction(args)

        # Numerical objects describing the pixel data in a footprint
        self.data = None
        self.ngrid_x = None
        self.ngrid_y = None
        self.mask = None
        self.variance = None
        self.wcs_matrix = None
        self.bounds = None
        self.bias = 0.0
        self.scale = 0.2
        self.gain = 1.0
        self.photo_calib = 1.0 # inst flux / nJy
        # Load the values of these objects from supplied data
        self.load_and_import_data()


    def init_model(self):
        # Load basic model characteristics
        assert 'model' in self.config, 'config is missing "model" section.'
        for necessary_attribute in ['num_sources', 'model_params', 'model_class']:
            assert necessary_attribute in self.config['model'], \
                f'"model" section of config is missing {necessary_attribute}.'

        self.num_sources = self.config['model']['num_sources']

        actv_params = self.config['model']['model_params'].split(' ')
        model_kwargs = dict({'active_parameters': actv_params})
        self.n_params = len(actv_params)

        model_class_name = self.config['model']['model_class']
        if 'model_module' in self.config['model']:
            model_module = __import__(self.config['model']['model_module'])
        else:
            model_module = __import__('jiffy.galsim_galaxy')
        self.src_models = [getattr(model_module, model_class_name)(
                            self.config, **model_kwargs)
                            for i in range(self.num_sources)]

        # Load values that control how the MCMC chain evolves
        if 'init' in self.config:
            # Set seed for random number generation
            if 'seed' in self.config['init']:
                np.random.seed(self.config['init']['seed'])

            # Load initial parameter values for the MCMC chain
            if 'param_values' in self.config['init']:
                assert 'init_param_file' not in self.config['init'], \
                    'Supply only one of param_values or init_param_file in config.'
                for param_name, param_value in self.config['init']['param_values'].items():
                    self.set_param_by_name(param_name, param_value)
            elif 'init_param_file' in self.config['init']:
                self.initialize_param_values(self.config['init']['init_param_file'])


    # Parse config
    def parse_init_config(self, label):
        form = None
        module = None
        for arg_name in self.config['model']:
            P = len(label) + 1
            if arg_name[:P] == f'{label}_':
                if arg_name[P:] == 'form':
                    form = self.config['model'][arg_name]
                elif arg_name[P:] == 'module':
                    module = self.config['model'][arg_name]

        kwargs = dict()
        if label in self.config:
            for arg_name in self.config[label]:
                kwargs[arg_name] = self.config[label][arg_name]

        return form, module, kwargs


    # Construct the function used to compute the log-prior
    def init_prior(self, args=None):
        form, module, kwargs = self.parse_init_config('prior')
        
        self.prior = priors.initialize_prior(
            form, module, args, **kwargs)


    # Construct the function used to compute the log-likelihood
    def init_likelihood(self, args=None):
        form, module, kwargs = self.parse_init_config('likelihood')

        self.likelihood = likelihoods.initialize_likelihood(
            form, module, args, **kwargs)


    # Construct the function used to compute detection corrections
    # to the log-likelihood
    def init_detection_correction(self, args=None):
        form, module, kwargs = self.parse_init_config('detection_correction')

        self.detection_correction = detections.initialize_detection_correction(
            form, module, args, **kwargs)        


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


    def import_data(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if 'data' in kwargs and kwargs['data'] is not None:
            self.ngrid_y, self.ngrid_x = kwargs['data'].shape


    def load_and_import_data(self):
        kwargs = {'data': None,
            'variance': None, 'mask': None,
            'wcs_matrix': None, 'bounds': None,
            'bias': 0.0, 'scale': 0.2, 'gain': 1.0, 'photo_calib': 1.0}
        
        if 'footprint' in self.config:
            fp = self.config['footprint']
            for key in kwargs:
                if key in fp:
                    if isinstance(fp[key], str):
                        kwargs[key] = np.load(fp[key])
                    else:
                        kwargs[key] = fp[key]
        
        self.import_data(**kwargs)


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


    def get_model_image(self):
        # Set up a blank image template with the correct bounds and wcs,
        # onto which the model image will be drawn.
        if self.wcs_matrix is not None:
            wcs = galsim.JacobianWCS(self.wcs_matrix[0, 0], self.wcs_matrix[0, 1],
                                     self.wcs_matrix[1, 0], self.wcs_matrix[1, 1])
            if self.bounds is not None:
                bounds = galsim.BoundsI(*self.bounds)
                model_image = galsim.ImageF(wcs=wcs, bounds=bounds, init_value=self.bias)
            else:
                model_image = galsim.ImageF(self.ngrid_x, self.ngrid_y,
                                            wcs=wcs, init_value=self.bias)
        elif self.bounds is not None:
            bounds = galsim.BoundsI(*self.bounds)
            model_image = galsim.ImageF(scale=self.scale, bounds=bounds, init_value=self.bias)
        else:
            model_image = galsim.ImageF(self.ngrid_x, self.ngrid_y,
                                        scale=self.scale, init_value=self.bias)
        
        # Try to draw all the sources on the template image
        for isrc in range(self.num_sources):
            model_image = self.src_models[isrc].get_image(
                template_image=model_image, gain=self.gain, photo_calib=self.photo_calib)
            if model_image is None:
                return None
        
        return model_image


    def logprior(self):
        '''
        Evaluate the log-prior of the model parameters
        '''
        try:
            res = self.prior(self.src_models)
        except:
            return np.nan
        
        return res


    def loglike(self):
        '''
        Evaluate the log-likelihood of the pixel data in a footprint,
        for some assumed image model with assumed model parameter values.
        '''
        # Render a model image
        try:
            model_image = self.get_model_image()
        except Exception as exception:
            return np.nan
        if model_image is None:
            return np.nan

        # Compute the log-likelihood, using the rendered model image
        try:
            loglike = self.likelihood(model_image, self)
        except:
            return np.nan
        
        if self.detection_correction:
            # Scale up the likelihood to account for the fact that we're only
            # looking at data examples that pass a detection algorithm
            try:
                detection_correction = self.detection_correction(
                    model_image, self)
            except:
                return np.nan

            loglike += detection_correction
        
        return loglike


    # Compute the log of the numerator of the posterior
    def logpost(self, params):
        # Set the Roaster params to the given values,
        # and check if they form a valid combination for this model.
        valid_params = self.set_params(params)
        # We consider "invalid" parameter combinations at this stage
        # to be out of the support of the prior.
        if not valid_params:
            return -np.inf

        # Compute the log-prior and log-likelihood,
        # using the newly-set Roaster params.
        logprior = self.logprior()
        if not np.isfinite(logprior):
            # We consider parameters with non-finite log-prior values
            # to be out of the support of the prior.
            # Terminate early without computing the likelihood,
            # which would waste time and potentially raise errors.
            return logprior

        loglike = self.loglike()

        logpost = logprior + loglike

        return logpost


    def __call__(self, params):
        '''
        Ensure the chain produces outputs for all inputs.
        '''
        try:
            logpost = self.logpost(params)
            logpost = float(logpost)
        except:
            logpost = np.nan

        return logpost


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
            print('Bad (non-finite) initial chain parameters.')

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
