import numpy as np

# ---------------------------------------------------------------------------------------
# Prior distributions for interim sampling of galaxy model parameters
# ---------------------------------------------------------------------------------------
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
