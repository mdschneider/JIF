import numpy as np
import pickle
from jiffy.galsim_galaxy import PARAM_BOUNDS


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
class IsolatedFootprintPrior_FixedNu_DC2(object):
    def __init__(self, args=None, **kwargs):
        galtype = kwargs['type'] # 'bulge' or 'disk'
        prior_params = {
            'gm_filename': {'bulge': 'gmfile_negNu.pkl',
                             'disk': 'gmfile_posNu.pkl'},
            'mean_hlrFlux': {'bulge': np.array([-1.02586301,  0.56355062]),
                              'disk': np.array([-0.61084441,  0.86118777])},
            'inv_cov_hlrFlux': {'bulge': np.array([[ 4.04738463, -1.8642854 ],
                                                   [-1.8642854 ,  2.10040375]]),
                                 'disk': np.array([[ 3.76129273, -1.17928324],
                                                   [-1.17928324,  2.08487754]])},
            # For reference:
            'nu': {'bulge': -0.708, 'disk': 0.5},
            'n': {'bulge': 4, 'disk': 1},
            'type_frac': {'bulge': 0.5282292589586854,
                         'disk': 0.47177074104131456}
        }

        self.scale = 0.2 # arcsec per pixel

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
        self.lognorm_e_angle = -np.log(2 * np.pi)
        # Uniform centroid angle distribution
        self.lognorm_dr_angle = -np.log(2 * np.pi)

        type_frac = prior_params['type_frac'][galtype]
        self.log_type_frac = np.log(type_frac)

    def evaluate(self, hlr, e1, e2, flux, dx, dy):
        # These specific prior functions correspond to pixel distances,
        # but the MCMC parameters are in arcsec
        hlr, dx, dy = hlr / self.scale, dx / self.scale, dy / self.scale

        # 4D Bayesian Gaussian mixture model for log-hlr, log-flux, dr, e
        e = np.sqrt(e1**2 + e2**2)
        dr = np.sqrt(dx**2 + dy**2)
        features = np.array([np.log(hlr), np.log(flux), dr, e])
        lnprior_4features = self.gm.score_samples([features])

        # Flat prior for ellipticity angle
        lnprior_e_angle = self.lognorm_e_angle
        # Flat prior for position angle
        lnprior_dr_angle = self.lognorm_dr_angle

        lnprior = lnprior_4features + lnprior_e_angle + lnprior_dr_angle
        return lnprior

    def __call__(self, src_models):
        param_names = ['hlr', 'e1', 'e2', 'flux', 'dx', 'dy']
        hlr, e1, e2, flux, dx, dy = [src_models[0].params[param_name][0]
            for param_name in param_names]

        return self.evaluate(hlr, e1, e2, flux, dx, dy)

# Isolated (one true object) footprints detected in DC2 tract 3830
class IsolatedFootprintPrior_VariableNu_DC2(object):
    def __init__(self, args=None, hlrFlux_gm_filename='hlrflux_gmfile.pkl',
        e_gm_filename='e_gmfile.pkl', dr_gm_filename='dr_gmfile.pkl'):
        self.scale = 0.2 # arcsec per pixel

        # Mean and inverse covariance matrix of log-hlr (in log-pixels)
        # and log-flux (in log-inst flux)
        self.mean_hlrFlux = np.array([-0.83006938,  0.70396712])
        self.inv_cov_hlrFlux = np.array([[ 3.56387564, -1.54370072],
                                         [-1.54370072,  2.05263992]])
        # self.lognorm_hlrFlux = -np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(self.inv_cov_hlrFlux))

        with open(hlrFlux_gm_filename, mode='rb') as hlrFlux_gm_file:
            self.hlrFlux_gm = pickle.load(hlrFlux_gm_file)

        with open(e_gm_filename, mode='rb') as e_gm_file:
            self.e_gm = pickle.load(e_gm_file)
        self.lognorm_e_angle = -np.log(2 * np.pi)

        with open(dr_gm_filename, mode='rb') as dr_gm_file:
            self.dr_gm = pickle.load(dr_gm_file)
        self.lognorm_dr_angle = -np.log(2 * np.pi)

        self.lognorm_nu = -np.log(PARAM_BOUNDS['nu'][1] - PARAM_BOUNDS['nu'][0])

    def evaluate(self, nu, hlr, e1, e2, flux, dx, dy):
        # These specific prior functions correspond to pixel distances,
        # but the MCMC parameters are in arcsec
        hlr, dx, dy = hlr / self.scale, dx / self.scale, dy / self.scale

        # # 2D Gaussian prior for log-hlr, log-flux
        # hlrFlux_dev = np.log(np.array([hlr, flux])) - self.mean_hlrFlux
        # lnprior_hlrFlux = self.lognorm_hlrFlux
        # lnprior_hlrFlux -= 0.5 * np.dot(hlrFlux_dev, np.matmul(self.inv_cov_hlrFlux, hlrFlux_dev))

        # Bayesian Gaussian mixture model for log-hlr, log-flux
        log_hlrFlux = np.log(np.array([hlr, flux]))
        lnprior_hlrFlux = self.hlrFlux_gm.score_samples([log_hlrFlux])

        # Bayesian Gaussian mixture model for ellipticity
        e = np.sqrt(e1**2 + e2**2)
        lnprior_e = self.e_gm.score_samples([[e]])[0]
        # Flat prior for ellipticity angle
        lnprior_e_angle = self.lognorm_e_angle

        # Bayesian Gaussian mixture model for position
        dr = np.sqrt(dx**2 + dy**2)
        lnprior_dr = self.dr_gm.score_samples([[dr]])[0]
        # Flat prior for position angle
        lnprior_dr_angle = self.lognorm_dr_angle

        # Flat prior for nu
        lnprior_nu = self.lognorm_nu

        lnprior = lnprior_hlrFlux + lnprior_e + lnprior_e_angle + lnprior_dr + lnprior_dr_angle + lnprior_nu
        return lnprior

    def __call__(self, src_models):
        param_names = ['nu', 'hlr', 'e1', 'e2', 'flux', 'dx', 'dy']
        nu, hlr, e1, e2, flux, dx, dy = [src_models[0].params[param_name][0]
            for param_name in param_names]

        return self.evaluate(nu, hlr, e1, e2, flux, dx, dy)


class DefaultPriorSpergel(object):
    """
    A default prior for a single-component Spergel galaxy
    """
    def __init__(self, args=None):
        ### Gaussian mixture in 'nu' parameter
        # self.nu_mean_1 = -0.6 ### ~ de Vacouleur profile
        # self.nu_mean_2 = 0.5 ### ~ exponential profile
        # self.nu_var_1 = 0.05
        # self.nu_var_2 = 0.01
        self.nu_mean_1 = 0.0
        self.nu_mean_2 = 0.0
        self.nu_var_1 = 0.1
        self.nu_var_2 = 0.1
        ### Gamma distribution keeping half-light radius from becoming
        ### much larger than 1 arcsecond or too close to zero.
        self.hlr_shape = 2.
        self.hlr_scale = 0.15
        ### Gaussian distribution in log flux
        self.mag_mean = 40.0
        self.mag_var = 7.0
        ### Beta distribution in ellipticity magnitude
        self.e_beta_a = 1.5
        self.e_beta_b = 5.0
        ### Gaussian priors in centroid parameters
        self.pos_var = 0.5

    def _lnprior_nu(self, nu):
        d1 = (nu - self.nu_mean_1)
        d2 = (nu - self.nu_mean_2)
        return -0.5 * (d1*d1/self.nu_var_1 + d2*d2/self.nu_var_2)
        #return 0.0

    def _lnprior_hlr(self, hlr):
        if hlr < 0:
            return -(np.inf)
        else:
            return (self.hlr_shape-1.)*np.log(hlr) - (hlr / self.hlr_scale)

    def _lnprior_mag(self, mag):
        delta = mag - self.mag_mean
        return -0.5 * delta * delta / self.mag_var

    def _lnprior_flux(self, flux):
        return -0.5 * (flux - 1.0)**2 / 0.01

    def evaluate(nu, hlr, e1, e2, flux, dx, dy):
        lnp = 0.0
        lnp += self._lnprior_nu(nu)
        ### Half-light radius
        lnp += self._lnprior_hlr(hlr)
        ### Flux
        lnp += self._lnprior_flux(flux)
        #lnp += self._lnprior_mag(omega[0].mag_sed1)
        #lnp += self._lnprior_mag(omega[0].mag_sed2)
        #lnp += self._lnprior_mag(omega[0].mag_sed3)
        #lnp += self._lnprior_mag(omega[0].mag_sed4)
        ### Ellipticity magnitude
        e = np.sqrt(e1**2 + e2**2)
        if e > 1:
            return -(np.inf)
        else:
            lnp += (self.e_beta_a-1.)*np.log(e) + (self.e_beta_b-1.)*np.log(1.-e)
        ### Centroid (x,y) perturbations
        lnp += -0.5 * (dx*dx + dy*dy) / self.pos_var
        return lnp

    def __call__(self, src_models):
        param_names = ['nu', 'hlr', 'e1', 'e2', 'flux', 'dx', 'dy']
        nu, hlr, e1, e2, flux, dx, dy = [src_models[0].params[param_name][0]
            for param_name in param_names]

        return self.evaluate(nu, hlr, e1, e2, flux, dx, dy)


# Allow functions to easily be looked up by name,
# and define a few useful aliases.
priors = {None: EmptyPrior,
          'Empty': EmptyPrior,
          'EmptyPrior': EmptyPrior,
          'IsolatedFootprintPrior_FixedNu': IsolatedFootprintPrior_FixedNu_DC2,
          'IsolatedFootprintPrior_FixedNu_DC2': IsolatedFootprintPrior_FixedNu_DC2,
          'IsolatedFootprintPrior_VariableNu': IsolatedFootprintPrior_VariableNu_DC2,
          'IsolatedFootprintPrior_VariableNu_DC2': IsolatedFootprintPrior_VariableNu_DC2}

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
