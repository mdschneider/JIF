#!/usr/bin/env python
# encoding: utf-8
"""
jiffy roaster.py

Draw posterior samples of image source model parameters given the
likelihood functxion of an image footprint
"""
import numpy as np
import galsim
import galsim_galaxy

class Roaster(object):
    """
    Likelihood model for footprint pixel data given a parametric source model
    """

    def __init__(self):
        self.src_models = [galsim_galaxy.GalsimGalaxyModel()]

        self.ngrid_x = 64
        self.ngrid_y = 64
        self.noise_var = 3e-10
        self.data = None

    def _get_model_image(self):
        return self.src_models[0].get_image(self.ngrid_x, self.ngrid_y)

    def make_data(self):
        """
        Make fake data from the current stored galaxy model
        """
        image = self._get_model_image()
        noise = galsim.GaussianNoise(sigma=np.sqrt(self.noise_var))
        image.addNoise(noise)
        self.data = image.array
        return None

    def lnlike(self, params):
        """
        Evaluate the log-likelihood of the pixel data in a footprint
        """
        self.src_models[0].set_params(params)
        model = self.src_models[0].get_image(self.ngrid_x, self.ngrid_y)
        delta = (model.array - self.data)**2
        lnnorm = (- 0.5 * self.ngrid_x * self.ngrid_y *
                  np.sqrt(self.noise_var * 2 * np.pi))
        return -0.5*np.sum(delta / self.noise_var) + lnnorm

    def __call__(self, params):
        return self.lnlike(params)
