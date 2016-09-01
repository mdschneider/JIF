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

### Star SEDs for chromatic PSF simulations (see psf_model.py).
### Testing kludge: use the galaxy SEDs
k_star_SED_names = ['NGC_0695_spec', 'NGC_4125_spec', 'NGC_4552_spec', 'CGCG_049-057_spec']

### Minimum brightness a magnitude parameter can take
k_mag_param_minval = 99.


k_spergel_paramnames = ['nu', 'hlr', 'e', 'beta']

### Numpy composite object types for the model parameters for galaxy images under different
### modeling assumptions.
k_galparams_type_sersic = [('redshift', '<f8'),
                           ('n', '<f8'),
                           ('hlr', '<f8'),
                           ('e', '<f8'),
                           ('beta', '<f8')]
k_galparams_type_sersic += [('mag_sed{:d}'.format(i+1), '<f8')
                            for i in xrange(len(k_SED_names))]
k_galparams_type_sersic += [('dx', '<f8'), ('dy', '<f8')] ### In sky coordinates
# ---------------------------------------------------------------
k_galparams_type_spergel = [('redshift', '<f8')] + [(p, '<f8')
                            for p in k_spergel_paramnames]
k_galparams_type_spergel += [('mag_sed{:d}'.format(i+1), '<f8')
                             for i in xrange(len(k_SED_names))]
k_galparams_type_spergel += [('dx', '<f8'), ('dy', '<f8')] ### In sky coordinates
# ---------------------------------------------------------------
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
# ---------------------------------------------------------------
k_galsim_psf_types = [('psf_fwhm', '<f8'),
                      ('psf_e', '<f8'),
                      ('psf_beta', '<f8'),
                      ('psf_mag', '<f8'),
                      ('psf_dx', '<f8'),
                      ('psf_dy', '<f8'),
                      ('psf_aber_defocus', '<f8'),
                      ('psf_aber_astig1', '<f8'),
                      ('psf_aber_astig2', '<f8'),
                      ('psf_aber_coma1', '<f8'),
                      ('psf_aber_coma2', '<f8'),
                      ('psf_aber_trefoil1', '<f8'),
                      ('psf_aber_trefoil2', '<f8'),
                      ('psf_aber_spher', '<f8')]
k_galsim_psf_defaults = [(0.8,    # fwhm
                          0.01,   # e
                          0.4,    # beta
                          20.0,   # mag
                          0.0,    # dx (pixels)
                          0.0,    # dy (pixels)
                          0.,     # defocus
                          0.,     # astig1
                          0.,     # astig2
                          0.,     # coma1
                          0.,     # coma2
                          0.,     # trefoil1
                          0.,     # trefoil2
                          0.      # spher
                          )]  


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
    # -----------------------------------------------------
    "Spergel": [(1.,                    # redshift
                 0.3,                   # nu
                 1.0,                   # hlr
                 0.1,                   # e
                 np.pi/6,               # beta
                 26.,                   # mag_sed1
                 k_mag_param_minval,    # mag_sed2
                 k_mag_param_minval,    # mag_sed3
                 k_mag_param_minval,    # mag_sed4
                 0.,                    # dx
                 0.)],                  # dy
    # -----------------------------------------------------
    "BulgeDisk": [(1.,                  # redshift
                   0.5,                 # nu_bulge
                   0.6,                 # hlr_bulge
                   0.05,                # e_bulge
                   0.0,                 # beta_bulge
                   -0.6,                # nu_disk
                   1.8,                 # hlr_disk
                   0.3,                 # e_disk
                   np.pi/4,             # beta_disk
                   22.,                 # mag_sed1_bulge
                   k_mag_param_minval,  # mag_sed2_bulge
                   k_mag_param_minval,  # mag_sed3_bulge
                   k_mag_param_minval,  # mag_sed4_bulge
                   k_mag_param_minval,  # mag_sed1_disk
                   22.,                 # mag_sed2_disk
                   k_mag_param_minval,  # mag_sed3_disk
                   k_mag_param_minval,  # mag_sed4_disk
                   0.,                  # dx_bulge
                   0.,                  # dy_bulge
                   0.,                  # dx_disk
                   0.)],                # dy_disk
    # -----------------------------------------------------
    "star": k_galsim_psf_defaults
}

### Lower, upper bounds on each possible parameter
k_param_bounds = {
  "redshift": (0.0, 6.0),
  "nu": (-0.8, 0.8),
  "hlr": (0.01, 10.0),
  "e": (0.0001, 0.9), ### Max |e| < 1 to avoid numerical instabilities
  "beta": (0.0, np.pi),
  "mag_sed1": (10.0, k_mag_param_minval),
  "mag_sed2": (10.0, k_mag_param_minval),
  "mag_sed3": (10.0, k_mag_param_minval),
  "mag_sed4": (10.0, k_mag_param_minval),
  "dx": (-10.0, 10.0),
  "dy": (-10.0, 10.0)
}

### Nominal expected variance of each parameter (for sampling)
k_param_vars = {
  "redshift": 1.0,
  "nu": 0.05**2,
  "hlr": 0.015**2,
  "e": 0.04**2,
  "beta": 0.07**2,
  "mag_sed1": 0.04**2,
  "mag_sed2": 0.04**2,
  "mag_sed3": 0.04**2,
  "mag_sed4": 0.04**2,
  "dx": 0.008**2,
  "dy": 0.008**2
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


def flux_from_AB_mag(mag, exposure_time_s=15, gain=2.1, 
                     bandpass_over_wavelength=2.,
                     pixel_scale_arcsec=0.2):
    """
    Convert an AB apparent magnitude to a flux
    """
    h = 6.62606957e-27 # Planck's constant in erg seconds
    # flux_AB = 3.63e-20 # ergs / s / Hz / cm^2
    mag_AB = 48.6# - 84. ### kludgey offset here to make fluxes look okay in GalSim units
    ### This flux is in units of erg / s / Hz
    flux = 10. ** (-(mag + mag_AB) / 2.5)
    ### Convert the flux to ADU / pixel
    ### See galsim sed.py
    flux *= gain * exposure_time_s * bandpass_over_wavelength / h
    flux *= 4*np.pi/(pixel_scale_arcsec / (180*60*60))
    return flux


def wrap_ellipticity_phase(phase):
    """
    Map a phase in radians to [0, pi) to model ellipticity orientation.
    """
    return (phase % np.pi)


def get_bounds(paramnames):
  """
  Get (min,max) bounds for the listed parameters

  Return a list of tuples to match the 'bounds' argument of scipy.optimize.minimze:
  http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
  """
  bounds = [k_param_bounds[p] for p in paramnames]
  return bounds
