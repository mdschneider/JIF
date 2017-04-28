#!/usr/bin/env python
# encoding: utf-8
"""
jiffy galsim_galaxy.py

Wrapper for simple GalSim galaxy models to use in MCMC.
"""
import numpy as np
import galsim

class GalsimGalaxyModel(object):
    """
    Parametric galaxy model from GalSim for image forward modeling
    """
    def __init__(self,
                 active_parameters=['e1', 'e2']):
        self.active_parameters = active_parameters
        self.n_params = len(self.active_parameters)

        self.params = np.array([(0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)],
                               dtype=[('nu', '<f8'),
                                      ('hlr', '<f8'),
                                      ('e1', '<f8'),
                                      ('e2', '<f8'),
                                      ('flux', '<f8'),
                                      ('dx', '<f8'),
                                      ('dy', '<f8')])
        self.params = self.params.view(np.recarray)

        # TESTING: hard-code some stuff for now
        self.psf = galsim.Kolmogorov(fwhm=0.6)
        return None

    def set_params(self, params):
        assert len(params) >= self.n_params
        for ip, pname in enumerate(self.active_parameters):
            self.params[pname][0] = params[ip]
        return None

    def get_image(self, ngrid_x, ngrid_y, scale=0.2, gain=1.0):
        """
        Render a GalSim Image() object from the internal model
        """
        gal = galsim.Spergel(self.params[0].nu,
                             half_light_radius=self.params[0].hlr,
                             flux=self.params[0].flux)
        gal = gal.shear(galsim.Shear(g1=self.params[0].e1,
                                     g2=self.params[0].e2))
        # mu = 1. / (1. - (self.params[0].e1**2 + self.params[0].e2**2))
        # gal = gal.lens(g1=self.params[0].e1, g2=self.params[0].e2, mu=mu)
        gal = gal.shift(self.params[0].dx, self.params[0].dy)
        obj = galsim.Convolve(self.psf, gal)
        model = obj.drawImage(nx=ngrid_x, ny=ngrid_y, scale=scale,
                              gain=gain)
        return model
