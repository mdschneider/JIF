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
        galtype = kwargs['type'] # 'bulge' or 'disk'
        prior_params = {
            'gm_filename': {'bulge': 'gmfile_bulge.pkl',
                             'disk': 'gmfile_disk.pkl'},
            'logprob_e_angle': {'bulge': partial(self.log_sine_prob, phase=3.9, scale=0.037774, level=0.159155),
                                'disk': lambda a : -np.log(2 * np.pi)},
            'logprob_dr_angle': {'bulge': partial(self.log_sine_prob, phase=np.pi/2, scale=-0.0198944, level=0.159155),
                                 'disk': lambda a : -np.log(2 * np.pi)},
            # For MAP fit initialization
            'mean_hlrFlux': {'bulge': np.array([-2.90110191,  4.41054969]),
                              'disk': np.array([-2.211403  ,  4.95104578])},
            'cov_hlrFlux': {'bulge': np.array([[0.30813249, 0.28674538],
                                               [0.28674538, 0.6420062 ]]),
                                 'disk': np.array([[0.32064104, 0.1782948 ],
                                                   [0.1782948 , 0.56214681]])},
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
        self.cov_hlrFlux = prior_params['inv_cov_hlrFlux'][galtype]

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

        self.psf_fwhm_mean = 0.763591484431654 # arcsec
        self.psf_fwhm_std = 0.00582026355508698 # arcsec

        self.psf_e_means_0 = [0.00172415, 0.00211098, 0.00146727, 0.00118756]
        self.psf_e_means_1 = [-0.00022855, -0.00011055, -0.0007743, 0.00047993]
        self.psf_e_stds_0 = [0.00025075, 0.00046715, 0.00037621, 0.00054171]
        self.psf_e_stds_1 = [0.00026504, 0.00052926, 0.00062448, 0.00058792]
        self.psf_e_weights = [0.41875398, 0.41875476, 0.19374868, 0.19374944]
        self.psf_e_n_components = len(self.psf_e_weights)
        self.psf_e_sum_weights = np.sum(psf_e_weights)
        self.psf_e_means = []
        self.psf_e_covs = []
        for idx in self.psf_e_n_components:
            self.psf_e_means.append([self.psf_e_means_0[idx], self.psf_e_means_1[idx]])
            std0, std1 = self.psf_e_stds_0[idx], self.psf_e_stds_1[idx]
            corr = self.psf_e_correlation
            cov = [[std0**2, corr * std0 * std1],
                   [corr * std0 * std1, std1**2]]
            self.psf_e_covs.append(cov)

    def log_sine_prob(self, t, phase, scale, level):
        return np.log(scale * np.sin(t + phase) + level)

    def logprob_psf_fwhm(self, psf_fwhm):
        return norm.logpdf(psf_fwhm, loc=self.psf_fwhm_mean, scale=psf_fwhm_std)

    def logprob_psf_e1e2(self, psf_e1, psf_e2):
        pdf = 0
        for idx in range(self.psf_e_n_components):
            weight = self.psf_e_weights[idx]
            mean = self.psf_e_means[idx]
            cov = self.psf_e_covs[idx]
            pdf += weight * multivariate_normal.pdf(
                [psf_e1, psf_e2], mean=mean, cov=cov) / self.psf_e_sum_weights

        return np.log(pdf)
        
    def evaluate(self, hlr, e1, e2, flux, dx, dy, psf_fwhm, psf_e1, psf_e2):
        e = np.sqrt(e1**2 + e2**2)
        e_angle = np.angle(e1 + e2*1j)
        dr = np.sqrt(dx**2 + dy**2)
        dr_angle = np.arctan2(dy, dx)
        features = np.array([np.log(hlr), np.log(flux), dr, e])

        # If anything is nan, return nan.
        # Otherwise, if anything is infinite, say that it falls outside the prior's support.
        # For anything explicitly outside the prior's support, return -inf (i.e., log(0)).
        all_gal_args = [hlr, e1, e2, flux, dx, dy]
        all_psf_args = [psf_fwhm, psf_e1, psf_e2]
        if np.any(np.isnan(all_gal_args)):
            return np.nan
        for param in all_psf_args:
            if param is not None:
                if np.any(np.isnan(param)):
                    return np.nan
                if not np.all(np.isfinite(param)):
                    return -np.inf
        if (psf_e1 is None and psf_e2 is not None) or (psf_e1 is not None and psf_e2 is None):
            return np.nan
        if not np.all(np.isfinite(all_gal_args)) or not np.all(np.isfinite(features)):
            return -np.inf
        if np.any(e >= 1):
            return -np.inf

        # 4D Bayesian Gaussian mixture model for log-hlr, log-flux, dr, e
        lnprior_4features = self.gm.score_samples([features])

        # Prior for ellipticity angle
        lnprior_e_angle = self.logprob_e_angle(e_angle)
        # Prior for position angle
        lnprior_dr_angle = self.logprob_dr_angle(dr_angle)

        # Prior for PSF parameters
        lnprior_psf_fwhm, lnprior_psf_e1e2 = 0, 0
        if psf_fwhm is not None:
            lnprior_psf_fwhm = self.logprob_psf_fwhm(psf_fwhm)
        if psf_e1 is not None and psf_e2 is not None:
            lnprior_psf_e1e2 = self.logprob_psf_e1e2(psf_e1, psf_e2)
        lnprior_psf = lnprior_psf_fwhm + lnprior_psf_e1e2

        # Combined prior
        lnprior = lnprior_4features + lnprior_e_angle + lnprior_dr_angle + lnprior_psf

        all_args = all_gal_args + all_psf_args
        if np.all([np.issubdtype(type(x), np.number) for x in all_args]):
            lnprior = float(lnprior)

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
