import numpy as np
from jiffy.galsim_galaxy import K_PARAM_BOUNDS

priors = {None: EmptyPrior,
          'Empty': EmptyPrior,
          'EmptyPrior': EmptyPrior,
          'IsolatedFootprintPrior': IsolatedFootprintPrior}

def initialize_prior(prior_form=None, prior_module=None, **kwargs):
    if prior_module is None:
        # prior_form should be one of the names of priors in this file
        prior = priors[prior_form]
    else:
        prior_module = __import__(prior_module)
        # prior_form should be the name of a class in prior_module
        prior = getattr(prior_module, prior_form)

    # Initialize an instance of the prior with the given keyword arguments
    return prior(**kwargs)


# ---------------------------------------------------------------------------------------
# Prior distributions for interim sampling of galaxy model parameters
# ---------------------------------------------------------------------------------------
class IsolatedFootprintPrior(object):
    def __init__(self):
        self.scale = 0.2 # arcsec per pixel

        # Mean and inverse covariance matrix of log-hlr (in log-pixels)
        # and log-flux (in log-inst flux)
        self.mean_hlrflux = np.array([-0.83008735,  0.70397003])
        self.inv_cov_hlrflux = np.array([[ 3.56382608, -1.54375759],
                                         [-1.54375759,  2.05263523]])
        self.lognorm_hlrflux = 0.5*np.log(np.linalg.det(2*np.pi * self.inv_cov_hlrflux))

        self.mean_e = 0
        # Laplace distribution parameter for e1,e2
        self.laplace_width_e = 0.19345975321402278
        self.lognorm_e = -np.log(2*self.laplace_width_e)

        self.mean_dxdy = 0
        # Variance of dx,dy, in pixels
        self.gauss_var_dxdy = 0.5769578977280152
        self.lognorm_dxdy = -0.5*np.log(2*np.pi * self.gauss_var_dxdy)

        self.lognorm_nu = -np.log(K_PARAM_BOUNDS['nu'][1] - K_PARAM_BOUNDS['nu'][0])

    def __call__(self, params):
        nu, hlr, e1, e2, flux, dx, dy = tuple(params)
        # The prior parameters correspond to pixel distances, not arcsec
        hlr, dx, dy = hlr/self.scale, dx/self.scale, dy/self.scale

        # 2D Gaussian prior for log-hlr, log-flux
        hlrflux = np.log(np.array([hlr, flux])) - self.mean_hlrflux
        lnprior_hlrflux = self.lognorm_hlrflux - 0.5*np.dot(hlrflux, np.matmul(self.inv_cov_hlrflux, hlrflux))

        # Laplace priors for e1, e2 with the same width
        lnprior_e1 = self.lognorm_e - np.abs(e1 - self.mean_e) / self.laplace_width_e
        lnprior_e2 = self.lognorm_e - np.abs(e2 - self.mean_e) / self.laplace_width_e

        # 1D Gaussian priors for dx, dy with the same width
        lnprior_dx = self.lognorm_dxdy - 0.5*(dx - self.mean_dxdy)**2 / self.gauss_var_dxdy
        lnprior_dy = self.lognorm_dxdy - 0.5*(dy - self.mean_dxdy)**2 / self.gauss_var_dxdy

        # Flat prior for nu
        lnprior_nu = self.lognorm_nu

        lnprior = lnprior_hlrflux + lnprior_e1 + lnprior_e2 + lnprior_dx + lnprior_dy + lnprior_nu
        return lnprior

class DefaultPriorSpergel(object):
    """
    A default prior for a single-component Spergel galaxy
    """
    def __init__(self):
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

    def __call__(self, omega):
        lnp = 0.0
        ### 'nu' parameter - peaked at exponential and de Vacouleur profile values
        omega = np.rec.array([tuple(omega)], 
                             dtype=[('e1', '<f8'),
                                    ('e2', '<f8'),
                                    ('hlr', '<f8'),
                                    ('flux', '<f8'),
                                    ('nu', '<f8'),
                                    ('dx', '<f8'),
                                    ('dy', '<f8')])
        # print(omega)
        # print(omega[0])
        lnp += self._lnprior_nu(omega[0].nu)
        ### Half-light radius
        lnp += self._lnprior_hlr(omega[0].hlr)
        ### Flux
        lnp += self._lnprior_flux(omega[0].flux)
        #lnp += self._lnprior_mag(omega[0].mag_sed1)
        #lnp += self._lnprior_mag(omega[0].mag_sed2)
        #lnp += self._lnprior_mag(omega[0].mag_sed3)
        #lnp += self._lnprior_mag(omega[0].mag_sed4)
        ### Ellipticity magnitude
        e = np.hypot(omega[0].e1, omega[0].e2)
        if e > 1:
            return -(np.inf)
        else:
            lnp += (self.e_beta_a-1.)*np.log(e) + (self.e_beta_b-1.)*np.log(1.-e)
        ### Centroid (x,y) perturbations
        dx = omega[0].dx
        dy = omega[0].dy
        lnp += -0.5 * (dx*dx + dy*dy) / self.pos_var
        return lnp
