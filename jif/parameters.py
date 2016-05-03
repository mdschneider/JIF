#!/usr/bin/env python
# encoding: utf-8
"""
parameters.py

Image model parameters as numpy ndarrays
"""
import numpy as np

### These SEDs do not go to long enough wavelengths for WFIRST bands
# k_SED_names = ['CWW_E_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext', 'CWW_Im_ext']
### From the Brown et al. (2014) atlas:
k_SED_names = ['NGC_0695_spec', 'NGC_4125_spec', 'NGC_4552_spec', 'CGCG_049-057_spec']

### Minimum brightness a magnitude parameter can take
k_mag_param_minval = 99.


k_spergel_paramnames = ['nu', 'hlr', 'e', 'beta']

### Numpy composite object types for the model parameters for galaxy images under different
### modeling assumptions.
k_galparams_type_sersic = [('redshift', '<f8'), ('n', '<f8'), ('hlr', '<f8'),
                           ('e', '<f8'), ('beta', '<f8')]
k_galparams_type_sersic += [('mag_sed{:d}'.format(i+1), '<f8')
                            for i in xrange(len(k_SED_names))]
k_galparams_type_sersic += [('dx', '<f8'), ('dy', '<f8')]


k_galparams_type_spergel = [('redshift', '<f8')] + [(p, '<f8')
                            for p in k_spergel_paramnames]
k_galparams_type_spergel += [('mag_sed{:d}'.format(i+1), '<f8')
                             for i in xrange(len(k_SED_names))]
k_galparams_type_spergel += [('dx', '<f8'), ('dy', '<f8')]


k_galparams_type_bulgedisk = [('redshift', '<f8')]
k_galparams_type_bulgedisk += [('{}_bulge'.format(p), '<f8')
                               for p in k_spergel_paramnames]
k_galparams_type_bulgedisk += [('{}_disk'.format(p), '<f8')
                               for p in k_spergel_paramnames]
k_galparams_type_bulgedisk += [('mag_sed{:d}_bulge'.format(i+1), '<f8')
    for i in xrange(len(k_SED_names))]
k_galparams_type_bulgedisk += [('mag_sed{:d}_disk'.format(i+1), '<f8')
    for i in xrange(len(k_SED_names))]
k_galparams_type_bulgedisk += [('dx_bulge', '<f8'), ('dy_bulge', '<f8')]
k_galparams_type_bulgedisk += [('dx_disk', '<f8'), ('dy_disk', '<f8')]


k_galsim_psf_types = [('psf_fwhm', '<f8'), ('psf_e', '<f8'), ('psf_beta', '<f8'),
                      ('psf_lnflux', '<f8')]
k_galsim_psf_defaults = [(0.6, 0.01, 0.4, 0.0)]



k_galparams_types = {
    "Sersic": k_galparams_type_sersic,
    "Spergel": k_galparams_type_spergel,
    "BulgeDisk": k_galparams_type_bulgedisk,
    "star": k_galsim_psf_types
}

### The galaxy models are initialized with these values:
k_galparams_defaults = {
    "Sersic": [(1., 3.4, 1.0, 0.1, np.pi/4, 22., k_mag_param_minval,
        k_mag_param_minval, k_mag_param_minval, 0., 0.)],
    "Spergel": [(1.,        # redshift
                 0.3,       # nu
                 1.0,       # hlr
                 0.1,       # e
                 np.pi/6,   # beta
                 20.,      # mag_sed1
                 k_mag_param_minval,   # mag_sed2
                 k_mag_param_minval,   # mag_sed3
                 k_mag_param_minval,   # mag_sed4
                 0.,        # dx
                 0.)],      # dy
    "BulgeDisk": [(1.,
        0.5, 0.6, 0.05, 0.0,
        -0.6, 1.8, 0.3, np.pi/4,
        22., k_mag_param_minval, k_mag_param_minval, k_mag_param_minval,
        k_mag_param_minval, 22., k_mag_param_minval, k_mag_param_minval,
        0., 0., 0., 0.)],
    "star": k_galsim_psf_defaults
}

def select_psf_paramnames(model_paramnames):
    """
    Given a list of galaxy and PSF model parameter names, select just the PSF
    model parameter names.

    Assumes PSF parameters contain the string 'psf'.
    """
    return [p for p in model_paramnames if 'psf' in p]


def select_galaxy_paramnames(model_paramnames):
    """
    Given a list of galaxy and PSF model parameter names, select just the galaxy
    model parameter names.

    Assumes PSF parameters contain the string 'psf', while galaxy parameter
    names do not.
    """
    return [p for p in model_paramnames if 'psf' not in p]


def replace_psf_parameters(model_params, model_params_new_psf, active_parameters):
    """
    Put the PSF parameter values from 'model_params_new_psf' into 'model_params'.

    Used in calculating the Multiple Importance Sampling (MIS) weights when marginalizing PSFs
    in different observation epochs of a single galaxy.

    @param model_params          Array of model parameter values
    @param model_params_new_psf  Array of model parameter values
    """
    # psf_paramnames = select_psf_paramnames(active_parameters)
    psf_param_ndx = ['psf' in p for p in active_parameters]
    p_out = model_params
    p_out[psf_param_ndx] = model_params_new_psf[psf_param_ndx]
    return p_out

