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
"""
jiffy galsim_galaxy.py

Wrapper for simple GalSim galaxy models to use in MCMC.
"""
import os
import numpy as np
import galsim
from jiffy import galsim_psf


# Used in validate_params()
PARAM_BOUNDS = {
    # Sersic n range allowed by galsim, minus a small safety margin
    'n': [0.31, 6.19],
    # Spergel nu range allowed by galsim, minus a small safety margin
    'nu': [-0.84, 3.99],
    # hlr must be positive for a real source
    # Overly large hlr values can cause major rendering slowdowns
    'hlr': [0.00001, 6.0],
    'e1': [-0.99, 0.99],
    'e2': [-0.99, 0.99],
    # Flux must be positive for a real source
    'flux': [0.0001, np.inf],
    'dx': [-np.inf, np.inf],
    'dy': [-np.inf, np.inf]
}
PARAM_CONSTRAINTS = (
    lambda params: params['e1'][0]**2 + params['e2'][0]**2 < 1,
)


# All light profiles must have at least the following functions defined
class LightProfile(object):
    def __init__(self):
        self.draw_method = None

    def init_params(self):
        raise NotImplementedError('Model not defined.')

    def light_profile(self, params, gsparams=None):
        raise NotImplementedError('Model not defined.')

class CircularProfile(LightProfile):
    def init_params(self):
        param_defaults = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        param_dtypes = [('hlr', '<f8'),
                        ('e1', '<f8'),
                        ('e2', '<f8'),
                        ('flux', '<f8'),
                        ('dx', '<f8'),
                        ('dy', '<f8')]

        return param_defaults, param_dtypes

    def circular_profile(self, params, gsparams=None):
        raise NotImplementedError('Model not defined.')

    def light_profile(self, params, gsparams=None):
        gal = self.circular_profile(params, gsparams)
        gal = gal.shear(galsim.Shear(g1=params.e1[0],
                                     g2=params.e2[0]))
        gal = gal.shift(params.dx[0], params.dy[0])

        return gal

class Exponential(CircularProfile):
    def circular_profile(self, params, gsparams=None):
        return galsim.Exponential(half_light_radius=params.hlr[0],
                                flux=params.flux[0],
                                gsparams=gsparams)    

class DeVaucouleurs(CircularProfile):
    def circular_profile(self, params, gsparams=None):
        return galsim.DeVaucouleurs(half_light_radius=params.hlr[0],
                                flux=params.flux[0],
                                gsparams=gsparams)

class Sersic(CircularProfile):
    def init_params(self):
        param_defaults, param_dtypes = super().init_params()
        param_defaults = (1.0,) + param_defaults
        param_dtypes = [('n', '<f8')] + param_dtypes

        return param_defaults, param_dtypes

    def circular_profile(self, params, gsparams=None):
        return galsim.Sersic(params.n[0],
                                half_light_radius=params.hlr[0],
                                flux=params.flux[0],
                                gsparams=gsparams)

class Spergel(CircularProfile):
    def init_params(self):
        param_defaults, param_dtypes = super().init_params()
        param_defaults = (0.5,) + param_defaults
        param_dtypes = [('nu', '<f8')] + param_dtypes

        return param_defaults, param_dtypes

    def circular_profile(self, params, gsparams=None):
        return galsim.Spergel(params.nu[0],
                                half_light_radius=params.hlr[0],
                                flux=params.flux[0],
                                gsparams=gsparams)

model_type_by_name = {'Spergel': Spergel,
                        'Sersic': Sersic,
                        'Exponential': Exponential,
                        'Disk': Exponential,
                        'DeVaucouleurs': DeVaucouleurs,
                        'Bulge': DeVaucouleurs}

class GalsimGalaxyModel(object):
    '''
    Parametric galaxy model from GalSim for image forward modeling

    The galaxy model is fixed as a 'Spergel' profile
    unless otherwise specified in the config
    '''
    def __init__(self, config,
                 active_parameters=['e1', 'e2'], **kwargs):
        self.draw_method = 'auto'
        model_type_name = 'Spergel'
        if 'model_type' in config['model']:
            model_type_name = config['model']['model_type']
        self.model_type = model_type_by_name[model_type_name]()
        # Override the 'auto' draw method if specified by this model type.
        # This will in turn be overridden later if the PSF specifies a draw method.
        if self.model_type.draw_method is not None:
            self.draw_method = self.model_type.draw_method

        self.active_parameters = active_parameters
        self.n_params = len(self.active_parameters)

        # Check if we're sampling in any PSF model parameters
        self.sample_psf = False
        self.actv_params_psf = []
        if np.any(['psf' in p for p in self.active_parameters]):
            self.sample_psf = True
            self.actv_params_psf = [p for p in self.active_parameters
                                    if 'psf' in p]
        self.actv_params_gal = [p for p in self.active_parameters
                                if 'psf' not in p]

        # Initialize parameters array
        param_defaults, param_dtypes = self.model_type.init_params()
        self.params = np.array([param_defaults], dtype=param_dtypes)
        self.params = self.params.view(np.recarray)

        # Initialize psf model
        self.init_psf(config, **kwargs)

        self.gsparams = galsim.GSParams(
            maximum_fft_size = 8192
        )

    def init_psf(self, config, **kwargs):
        # Initialize the PSF model that will be convolved with the galaxy
        psf_model_class_name = config['model']['psf_class']
        self.psf_model = getattr(galsim_psf, psf_model_class_name)(
            config, active_parameters=self.actv_params_psf, **kwargs)
        # Override the draw method if necessary for this PSF type.
        # Important when using PSFs from the observed image of a star,
        # since these have already been convolved by the pixel,
        # so don't want to do that again.
        if self.psf_model.draw_method is not None:
            self.draw_method = self.psf_model.draw_method

        # Store fixed PSF now unless we're sampling in the PSF model parameters
        self.static_psf = None
        if not self.sample_psf:
            self.static_psf = self.psf_model.get_model()

    def get_params(self):
        p = np.array([pv for pv in self.params[self.actv_params_gal][0]])
        if self.sample_psf:
            p = np.append(p, self.psf_model.get_params())
        return p

    def set_params(self, params):
        assert len(params) >= self.n_params
        for ip, pname in enumerate(self.actv_params_gal):
            self.params[pname][0] = params[ip]
        valid_params = self.validate_params()
        if self.sample_psf:
            valid_params *= self.psf_model.set_params(
                params[len(self.actv_params_gal):])
        return valid_params

    def get_param_by_name(self, paramname):
        '''
        Get a single parameter value using the parameter name as a key.

        Can access "active" or "inactive" parameters.
        '''
        if 'psf' in paramname:
            p = self.psf_model.get_param_by_name(paramname)
        else:
            p = self.params[paramname][0]
        return p

    def set_param_by_name(self, paramname, value):
        '''
        Set a single parameter value using the parameter name as a key.

        Can set "active" or "inactive" parameters. So, this routine gives a
        way to set fixed or fiducial values of model parameters that are not
        used in the MCMC sampling in Roaster.

        @param paramname    The name of the galaxy or PSF model parameter to set
        @param value        The value to assign to the model parameter
        '''
        if 'psf' in paramname:
            self.psf_model.set_param_by_name(paramname, value)
        else:
            self.params[paramname][0] = value
        return None

    def validate_params(self):
        '''
        Check that all model parameters are within allowed ranges

        @returns a boolean indicating the validity of the current parameters
        '''
        # Run a series of validity checks
        # If any of them fail, immediately return False
        def _inbounds(param, bounds):
            return param >= bounds[0] and param <= bounds[1]
        for pname, _ in self.params.dtype.descr:
            if not np.isfinite(self.params[pname][0]):
                return False
            if not _inbounds(self.params[pname][0], PARAM_BOUNDS[pname]):
                return False
        for constraint_satisfied in PARAM_CONSTRAINTS:
            if not constraint_satisfied(self.params):
                return False
        
        # All checks passed
        return True

    def get_psf(self):
        if self.sample_psf:
            return self.psf_model.get_model()
        else:
            return self.static_psf

    def get_image(self, ngrid_x=16, ngrid_y=16, scale=0.2, image=None, gain=1.0,
                  real_galaxy_catalog=None):
        '''
        Render a GalSim Image() object from the internal model
        
        Parameters
        ----------
        ngrid_x : int, optional
            Description
        ngrid_y : int, optional
            Description
        scale : float, optional
            Description
        image : None, optional
            Description
        gain : float, optional
            Description
        real_gals : bool, optional
            Render using a GalSim "RealGalaxy" rather than Spergel profile.
            Useful for testing model bias. (Default: False)
        
        Returns
        -------
        TYPE
            Description
        '''
        if real_galaxy_catalog is not None:
            # "Real" galaxies have intrinsic sizes and ellipticities, so
            # do not add any more here.
            rgndx = np.random.randint(low=0, high=real_galaxy_catalog.nobjects)
            print(f'*** Using GalSim RealGalaxy {rgndx}***')
            gal = galsim.RealGalaxy(real_galaxy_catalog,
                                    index=rgndx,
                                    flux=self.params.flux[0],
                                    gsparams=self.gsparams)
            gal = gal.shift(self.params.dx[0], self.params.dy[0])
        else:
            gal = self.model_type.light_profile(self.params,
                                                self.gsparams)
        obj = galsim.Convolve(self.get_psf(), gal)
        
        N = obj.getGoodImageSize(scale)
        if N > 2048:
            model = None
        else:
            try:
                if image is not None:
                    model = obj.drawImage(image=image, gain=gain,
                                          add_to_image=True,
                                          method=self.draw_method)
                else:
                    model = obj.drawImage(nx=ngrid_x, ny=ngrid_y, scale=scale,
                                          gain=gain, method=self.draw_method)
            except galsim.GalSimFFTSizeError:
                print("Trying to make an image that's too big.")
                print('Model params:')
                print(self.get_params())
                model = None
        return model


if __name__ == '__main__':
    '''
    Make a default test footprint file
    '''
    import footprints

    gg = GalsimGalaxyModel()
    img = gg.get_image(64, 64)
    noise_var = 1.e-8
    noise = galsim.GaussianNoise(sigma=np.sqrt(noise_var))
    img.addNoise(noise)

    dummy_mask = 1.0
    dummy_background = 0.0

    fname = '../data/TestData/jiffy_gg_image'

    galsim.fits.write(img, fname + '.fits')

    ftpnt = footprints.Footprints(fname + '.h5')

    ftpnt.save_images([img.array], [noise_var], [dummy_mask], [dummy_background],
                    segment_index=0, telescope='LSST', filter_name='r')
    ftpnt.save_tel_metadata()
