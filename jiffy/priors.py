import numpy as np
import pickle
from jiffy.galsim_galaxy import PARAM_BOUNDS
from functools import partial


class EmptyPrior(object):
    '''
    Prior form for the image model parameters

    This prior form is flat in all parameters (for any given parameterization).
    '''

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return 0.0


# ---------------------------------------------------------------------------------------
# Prior distributions for interim sampling of galaxy model parameters
# ---------------------------------------------------------------------------------------
# Prior for isolated (one true object) footprints detected in DC2 tract 3830
# Conditioned on the object being a specified type (bulge or disk)
# Add log_type_frac to log-prior/posterior to jointly model the type probability
# Updated for DP0.2 analysis, including optional PSF parameters
class IsolatedFootprintPrior_FixedNu_DC2(object):
    def __init__(self, args=None, **kwargs):
        # Number to add to flux_inst so that log(flux_inst + flux_inst_offset) is always well-defined
        self.flux_inst_offset = 5
        
        galtype = kwargs['type'] # 'bulge' or 'disk'
        prior_params = {
            'gm_filename': {'bulge': 'gmfile_bulge.pkl',
                             'disk': 'gmfile_disk.pkl'},
            'logprob_e_angle': {'bulge': partial(self.log_sine_prob, phase=3.9, scale=0.037774, level=0.159155),
                                'disk': lambda a : -np.log(2 * np.pi)},
            'logprob_dr_angle': {'bulge': partial(self.log_sine_prob, phase=np.pi/2, scale=-0.0198944, level=0.159155),
                                 'disk': lambda a : -np.log(2 * np.pi)},
            # For MAP fit initialization
            'mean_hlrFlux': {'bulge': np.array([-2.90110191,  0.2653783 ]),
                              'disk': np.array([-2.211403  ,  0.83366477])},
            'inv_cov_hlrFlux': {'bulge': np.array([[ 5.5390664 , -2.35809197],
                                                   [-2.35809197,  2.42428197]]),
                                 'disk': np.array([[ 3.78450651, -1.16082712],
                                                   [-1.16082712,  2.02404995]])},
            # For reference:
            'nu': {'bulge': -0.708, 'disk': 0.5},
            'n': {'bulge': 4, 'disk': 1},
            'type_frac': {'bulge': 0.39758631383088694,
                           'disk': 0.6024136861691131}
        }

        # Mean and inverse covariance matrix of log-hlr (in log-pixels)
        # and log-flux (in log-inst flux),
        # for use by MAP fitter
        self.mean_hlrFlux = prior_params['mean_hlrFlux'][galtype]
        self.inv_cov_hlrFlux = prior_params['inv_cov_hlrFlux'][galtype]

        # Correlated log-hlr, log-flux, dr, and e distribution
        # from 4D Bayesian Gaussian mixture model fit
        gm_filename = prior_params['gm_filename'][galtype]
        with open(gm_filename, mode='rb') as gm_file:
            self.gm = pickle.load(gm_file)

        # Uniform e angle distribution
        self.logprob_e_angle = prior_params['logprob_e_angle'][galtype]
        # Uniform centroid angle distribution
        self.logprob_dr_angle = prior_params['logprob_dr_angle'][galtype]        

        type_frac = prior_params['type_frac'][galtype]
        self.log_type_frac = np.log(type_frac)

        self.psf_fwhm_mean = 0.7630464189398478
        self.psf_fwhm_var = 3.4294244645920723e-05
        self.psf_e1_mean = 0.0016812232284244407
        self.psf_e1_var = 5.101391332227311e-07
        self.psf_e2_mean = -0.00015326566654888057
        self.psf_e2_var = 5.809799912978508e-07

    def log_sine_prob(t, phase, scale, level):
        return np.log(scale * np.sin(t + phase) + level)

    def log_1dgaussian(x, mean, var):
        return -0.5 * np.log(var * 2 * np.pi) - (x - mean)**2 / (2 * var)
        
    def evaluate(self, hlr, e1, e2, flux, dx, dy, psf_fwhm, psf_e1, psf_e2):
        # 4D Bayesian Gaussian mixture model for log-hlr, log-flux, dr, e
        e = np.sqrt(e1**2 + e2**2)
        dr = np.sqrt(dx**2 + dy**2)
        features = np.array([np.log(hlr), np.log(flux + self.flux_inst_offset), dr, e])
        lnprior_4features = self.gm.score_samples([features])

        # Prior for ellipticity angle
        e_angle = np.angle(e1 + e2*1j)
        lnprior_e_angle = self.logprob_e_angle(e_angle)
        # Prior for position angle
        dr_angle = np.arctan2(dy, dx)
        lnprior_dr_angle = self.logprob_dr_angle(dr_angle)

        # Prior for PSF parameters
        lnprior_psf_fwhm, lnprior_psf_e1, lnprior_psf_e2 = 0, 0, 0
        if psf_fwhm is not None:
            lnprior_psf_fwhm = self.log_1dgaussian(psf_fwhm, self.psf_fwhm_mean, self.psf_fwhm_var)
        if psf_e1 is not None:
            lnprior_psf_e1 = self.log_1dgaussian(psf_e1, self.psf_e1_mean, self.psf_e1_var)
        if psf_e2 is not None:
            lnprior_psf_e2 = self.log_1dgaussian(psf_e2, self.psf_e2_mean, self.psf_e2_var)
        lnprior_psf = lnprior_psf_fwhm + lnprior_psf_e1 + lnprior_psf_e2

        # Combined prior
        lnprior = lnprior_4features + lnprior_e_angle + lnprior_dr_angle + lnprior_psf
        return lnprior

    def __call__(self, src_models):
        param_names = ['hlr', 'e1', 'e2', 'flux', 'dx', 'dy']
        param_names_psf = ['psf_fwhm', 'psf_e1', 'psf_e2']

        # Fill kwargs with the model parameter values
        kwargs = dict()
        for param_name in param_names:
            kwargs[param_name] = src_models[0].params[param_name][0]
        for param_name in param_names_psf:
            if param_name in src_models[0].actv_params_psf:
                kwargs[param_name] = src_models[0].psf_model.params[param_name][0]

        # Compute and return the log-prior with these model parameter values
        return self.evaluate(**kwargs)


# Allow functions to easily be looked up by name,
# and define a few useful aliases.
priors = {None: EmptyPrior,
          'Empty': EmptyPrior,
          'EmptyPrior': EmptyPrior,
          'IsolatedFootprintPrior_FixedNu': IsolatedFootprintPrior_FixedNu_DC2,
          'IsolatedFootprintPrior_FixedNu_DC2': IsolatedFootprintPrior_FixedNu_DC2
         }


'''
prior_form (str): Name of a prior, to look up in a priors dict like the one above.
prior_module (str): If specified, look up prior_form in the specified module rather than here.
args: Parsed command-line args from Roaster initialization
kwargs (dict): Keyword arguments specified in a config file
'''
def initialize_prior(prior_form=None, prior_module=None, args=None, **kwargs):
    if prior_module is None:
        # prior_form should be one of the names of priors in this file
        prior = priors[prior_form]
    else:
        prior_module = __import__(prior_module)
        # prior_form should be the name of a class in prior_module
        prior = getattr(prior_module, prior_form)

    # Initialize an instance of the prior with the given keyword arguments
    return prior(args, **kwargs)
