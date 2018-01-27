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
jiffy galsim_psf.py

Wrapper for simple GalSim PSF models to use in MCMC.
"""
import numpy as np
import galsim


K_PARAM_BOUNDS = {
    "psf_fwhm": [0.1, 2.0], ## in arcseconds
                           ## Note that very small PSF widths will require 
                           ## large GalSim FFTs
    "psf_e1": [-0.6, 0.6],
    "psf_e2": [-0.6, 0.6],
    "psf_flux": [0.001, 1000.],
    "psf_dx": [-10., 10.],
    "psf_dy": [-10., 10.]
}


class GalsimPSFModel(object):
    """Parametric PSF models from GalSim for image forward modeling"""
    def __init__(self, active_parameters=['psf_fwhm']):
        self.active_parameters = active_parameters
        self.n_params = len(self.active_parameters)
        self._init_params()

    def _init_params(self):
        self.params = np.array([(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)],
                               dtype=[('psf_fwhm', '<f8'),
                                      ('psf_e1', '<f8'),
                                      ('psf_e2', '<f8'),
                                      ('psf_flux', '<f8'),
                                      ('psf_dx', '<f8'),
                                      ('psf_dy', '<f8')])
        self.params = self.params.view(np.recarray)

    def get_params(self):
        """
        Return a list of active model parameter values.
        """
        if len(self.active_parameters) > 0:
            p = self.params[self.active_parameters].view('<f8').copy()
        else:
            p = []
        return p

    def set_params(self, params):
        assert len(params) >= self.n_params
        for ip, pname in enumerate(self.active_parameters):
            print "Setting {} to {5.4f}".format(pname, params[ip])
            self.params[pname][0] = params[ip]
        valid_params = self.validate_params()
        return valid_params

    def validate_params(self):
        """
        Check that all model parameters are within allowed ranges

        @returns a boolean indicating the validity of the current parameters
        """
        def _inbounds(param, bounds):
            return param >= bounds[0] and param <= bounds[1]

        valid_params = 1
        for pname, _ in self.params.dtype.descr:
            if not _inbounds(self.params[pname][0], K_PARAM_BOUNDS[pname]):
                valid_params *= 0
        return bool(valid_params)

    def get_model(self):
        """
        Get the GalSim image model

        This is the object that can be used in, e.g., GalSim convolutions
        """
        psf = galsim.Kolmogorov(fwhm=self.params.psf_fwhm[0])
        psf_shape = galsim.Shear(e1=self.params.psf_e1[0],
                                 e2=self.params.psf_e2[0])
        psf = psf.shear(psf_shape)
        psf = psf.withFlux(self.params.psf_flux[0])
        psf = psf.shift(self.params.psf_dx[0], self.params.psf_dy[0])
        return psf

    def get_image(self, ngrid_x, ngrid_y, scale=0.2, image=None, gain=1.0):
        """
        Render a GalSim Image() object from the internal model
        """        
        obj = self.get_model()
        try:
            if image is not None:
                model = obj.drawImage(image=image, gain=gain, add_to_image=True)
            else:
                model = obj.drawImage(nx=ngrid_x, ny=ngrid_y, scale=scale,
                                      gain=gain)
        except RuntimeError:
            print "Trying to make an image that's too big."
            print self.get_params()
            model = None
        return model


class GalsimPSFLSST(GalsimPSFModel):
    fov_deg = 3.5
    tel_diam_m = 8.4
    wavelength_nm = 500.
    def __init__(self, active_parameters=['psf_fwhm']):
        self.active_parameters = active_parameters
        self.n_params = len(self.active_parameters)
        self._init_params()
        self._init_optics_dz_aberrations()
        self.aper = galsim.Aperture(diam=self.tel_diam_m)

    def _init_params(self):
        self.params = np.array([(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                                 0.1)],
                               dtype=[('psf_fwhm', '<f8'),
                                      ('psf_e1', '<f8'),
                                      ('psf_e2', '<f8'),
                                      ('psf_flux', '<f8'),
                                      ('psf_dx', '<f8'),
                                      ('psf_dy', '<f8'),
                                      ('psf_a_2020', '<f8')])
        self.params = self.params.view(np.recarray)

    def _init_optics_dz_aberrations(self):
        npupil = 12
        nfield = 21
        self.aberrations = np.zeros((npupil, nfield), dtype=np.float64)
        self.aberrations[2, 2] = self.params.psf_a_2020[0]

    def get_model(self, theta=(0.*galsim.arcmin, 0.*galsim.arcmin)):
        # atmosphere PSF
        atmos = galsim.Kolmogorov(fwhm=self.params.psf_fwhm[0])
        psf_shape = galsim.Shear(g=self.params.psf_e[0],
                                 beta=self.params.psf_beta[0]*galsim.radians)
        atmos = atmos.shear(psf_shape)

        # optics PSF
        op_model = galsim.OpticalScreenField(self.aberrations,
                                             fov_radius=self.fov_deg)
        screens = galsim.PhaseScreenList(op_model)
        screens.makePSF(lam=self.wavelength_nm, aper=self.aper, theta=theta)

        psf = galsim.Convolve([atmos, optics])
        psf = psf.shift(self.params.psf_dx[0], self.params.psf_dy[0])
        return psf


def main():
    """
    Make a default test footprint file
    """
    import footprints

    gp = GalsimPSFModel()
    img = gp.get_image(16, 16)
    noise_var = 1.e-8
    noise = galsim.GaussianNoise(sigma=np.sqrt(noise_var))
    img.addNoise(noise)

    dummy_mask = 1.0
    dummy_background = 0.0

    ftpnt = footprints.Footprints("../data/TestData/jiffy_psf_image.h5")

    ftpnt.save_images([img.array], [noise_var], [dummy_mask], [dummy_background],
                      segment_index=0, telescope="LSST", filter_name='r')
    ftpnt.save_tel_metadata()


if __name__ == '__main__':
    main()
