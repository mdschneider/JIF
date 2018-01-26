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
# the terms of the GNU General Public License (as published by the Free Software
# Foundation) version 2.1 dated February 1999. 
# 
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the terms and conditions of the GNU General
# Public License for more details. 
# 
# You should have received a copy of the GNU Lesser General Public License along
# with this program; if not, write to the Free Software Foundation, Inc., 59 
# Temple Place, Suite 330, Boston, MA 02111-1307 USA 
"""
jiffy galsim_galaxy.py

Wrapper for simple GalSim galaxy models to use in MCMC.
"""
import numpy as np
import galsim


K_PARAM_BOUNDS = {
    "nu": [-0.8, 0.8],
    "hlr": [0.01, 10.0],
    "e1": [-0.7, 0.7],
    "e2": [-0.7, 0.7],
    "flux": [0.001, 1000.0],
    "dx": [-10.0, 10.0],
    "dy": [-10.0, 10.0]
}


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

    def get_params(self):
      return self.params[self.active_parameters].view('<f8').copy()

    def set_params(self, params):
        assert len(params) >= self.n_params
        for ip, pname in enumerate(self.active_parameters):
            self.params[pname][0] = params[ip]
        valid_params = self.validate_params()
        return valid_params

    def validate_params(self):
        """
        Check that all model parameters are within allowed ranges

        @returns a boolean indicating teh validity of the current parameters
        """
        def _inbounds(param, bounds):
            return param >= bounds[0] and param <= bounds[1]

        valid_params = 1
        for pname, _ in self.params.dtype.descr:
            if not _inbounds(self.params[pname][0], K_PARAM_BOUNDS[pname]):
                valid_params *= 0
        return bool(valid_params)

    def get_image(self, ngrid_x, ngrid_y, scale=0.2, gain=1.0):
        """
        Render a GalSim Image() object from the internal model
        """
        gal = galsim.Spergel(self.params.nu[0],
                             half_light_radius=self.params.hlr[0],
                             flux=self.params.flux[0])
        gal = gal.shear(galsim.Shear(g1=self.params.e1[0],
                                     g2=self.params.e2[0]))
        # mu = 1. / (1. - (self.params.e1[0]**2 + self.params.e2[0]**2))
        # gal = gal.lens(g1=self.params.e1[0], g2=self.params.e2[0], mu=mu)
        gal = gal.shift(self.params.dx[0], self.params.dy[0])
        obj = galsim.Convolve(self.psf, gal)
        try:
            model = obj.drawImage(nx=ngrid_x, ny=ngrid_y, scale=scale,
                                  gain=gain)
        except RuntimeError:
            print "Trying to make an image that's too big."
            print self.get_params()
            model = None
        return model


if __name__ == '__main__':
    """
    Make a default test footprint file
    """
    import footprints

    gg = GalsimGalaxyModel()
    img = gg.get_image(64, 64)
    noise_var = 1.e-8
    noise = galsim.GaussianNoise(sigma=np.sqrt(noise_var))
    img.addNoise(noise)

    dummy_mask = 1.0
    dummy_background = 0.0

    ftpnt = footprints.Footprints("../data/TestData/jiffy_gg_image.h5")

    ftpnt.save_images([img.array], [noise_var], [dummy_mask], [dummy_background],
                    segment_index=0, telescope="LSST", filter_name='r')
    ftpnt.save_tel_metadata()
