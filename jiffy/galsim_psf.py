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
# the terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the terms and conditions of the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
"""
jiffy galsim_psf.py

Wrapper for simple GalSim PSF models to use in MCMC.
"""
import numpy as np
import warnings
import galsim
try:
    import galsim.lsst
except ImportError:
    warnings.warn("Cannot import GalSim LSST module - LSST PSF models won't work")


K_PARAM_BOUNDS = {
    "psf_fwhm": [0.02, 200.0], ## in arcseconds
                               ## Note that very small PSF widths will require 
                               ## large GalSim FFTs
    "psf_e1": [-0.6, 0.6],
    "psf_e2": [-0.6, 0.6],
    "psf_flux": [0.001, 1000.],
    "psf_dx": [-100., 100.],
    "psf_dy": [-100., 100.]
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
            # p = self.params[self.active_parameters].view('<f8').copy()
            p = np.array([pv for pv in self.params[self.active_parameters][0]])
        else:
            p = []
        return p

    def set_params(self, params):
        assert len(params) >= self.n_params
        for ip, pname in enumerate(self.active_parameters):
            # print "Setting {} to {5.4f}".format(pname, params[ip])
            self.params[pname][0] = params[ip]
        valid_params = self.validate_params()
        return valid_params

    def get_param_by_name(self, paramname):
        """
        Get a single parameter value using the parameter name as a key.

        Can access 'active' or 'inactive' parameters.
        """
        return self.params[paramname][0]

    def set_param_by_name(self, paramname, value):
        """
        Set a single parameter value using the parameter name as a key.

        Can set 'active' or 'inactive' parameters. So, this routine gives a
        way to set fixed or fiducial values of model parameters that are not
        used in the MCMC sampling in Roaster.

        @param paramname    The name of the galaxy or PSF model parameter to set
        @param value        The value to assign to the model parameter
        """
        self.params[paramname][0] = value
        return None

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
                print "bad params: ", pname, self.params[pname]
        return bool(valid_params)

    def get_model(self, theta=None):
        """
        Get the GalSim image model

        This is the object that can be used in, e.g., GalSim convolutions
        """
        # argument theta is not used here
        psf = galsim.Kolmogorov(fwhm=self.params.psf_fwhm[0])
        psf_shape = galsim.Shear(e1=self.params.psf_e1[0],
                                 e2=self.params.psf_e2[0])
        psf = psf.shear(psf_shape)
        psf = psf.withFlux(self.params.psf_flux[0])
        psf = psf.shift(self.params.psf_dx[0], self.params.psf_dy[0])
        return psf

    def get_image(self, ngrid_x=16, ngrid_y=16, scale=0.2, image=None, gain=1.0,
                  theta_x_arcmin=0., theta_y_arcmin=0., with_atmos=True):
        """
        Render a GalSim Image() object from the internal model
        """
        theta = (theta_x_arcmin*galsim.arcmin, theta_y_arcmin*galsim.arcmin)
        obj = self.get_model(theta, with_atmos=with_atmos)
        # try:
        #     if image is not None:
        #         model = obj.drawImage(image=image, gain=gain, add_to_image=True, method='fft')
        #     else:
        #         model = obj.drawImage(nx=ngrid_x, ny=ngrid_y, scale=scale,
        #                               gain=gain, method='fft')
        # except RuntimeError:
        #     print "Trying to make an image that's too big."
        #     print self.get_params()
        #     model = None
        
        if image is not None:
            model = obj.drawImage(image=image, gain=gain, add_to_image=True, method='fft')
        else:
            model = obj.drawImage(nx=ngrid_x, ny=ngrid_y, scale=scale,
                                  gain=gain, method='fft')        
        return model


class GalsimPSFLSST(GalsimPSFModel):
    fov_deg = 3.5
    tel_diam_m = 8.4
    wavelength_nm = 500.
    gsparams = galsim.GSParams(
            # folding_threshold=1.e-2, # maximum fractional flux that may be folded around edge of FFT
            # maxk_threshold=2.e-1,    # k-values less than this may be excluded off edge of FFT
            # xvalue_accuracy=1.e-2,   # approximations in real space aim to be this accurate
            # kvalue_accuracy=1.e-2,   # approximations in fourier space aim to be this accurate
            # shoot_accuracy=1.e-2,    # approximations in photon shooting aim to be this accurate
            minimum_fft_size=32,     # minimum size of ffts
            maximum_fft_size=20480)   # maximum size of ffts
    def __init__(self, active_parameters=['psf_fwhm']):
        self.active_parameters = active_parameters
        self.n_params = len(self.active_parameters)
        self._init_params()
        self._init_optics_dz_aberrations()
        self.aper = galsim.Aperture(diam=self.tel_diam_m,
                                    obscuration=0.65,
                                    lam=self.wavelength_nm,
                                    circular_pupil=True,
                                    # nstruts=4,
                                    # pupil_plane_scale=0.02,
                                    # pupil_plane_size=self.tel_diam_m*2,
                                    # pad_factor=1.0,
                                    # oversampling=1.0,
                                    gsparams=self.gsparams)

    def _init_params(self):
        # Parameter names / types
        atmos_types = [('psf_fwhm', '<f8'),
                       ('psf_e1',   '<f8'),
                       ('psf_e2',   '<f8'),
                       ('psf_flux', '<f8'),
                       ('psf_dx',   '<f8'),
                       ('psf_dy',   '<f8')]
        optics_aberr_types = [('psf_a_{:d}'.format(i), '<f8')
                              for i in range(1, 41)]
        param_types = atmos_types + optics_aberr_types

        # Parameter initial values
        atmos_vals = (0.6, 0.0, 0.0, 1.0, 0.0, 0.0)
        optics_vals = tuple(0.0 for i in range(1, 41))

        self.params = np.array([atmos_vals + optics_vals],
                               dtype=param_types)
        self.params = self.params.view(np.recarray)

    def _init_optics_dz_aberrations(self):
        # Read LSST nominal coefficients
        dat = galsim.lsst.lsst_psfs._read_aberrations()
        # npupil = int(np.max(dat[:, 0]))
        # nfield = int(np.max(dat[:, 2]))
        npupil = 67
        nfield = 28

        # Make a lookup table
        noll = np.zeros((11, 12), dtype=int)
        for j in range(67):
            n, m = galsim.phase_screens._noll_to_zern(j)
            if m >= 0:
                noll[n, m] = j
            else:
                noll[n, abs(m)+1] = j

        self.aberrations = np.zeros((npupil, nfield), dtype=np.float64)
        for i in xrange(dat.shape[0]):
            m_pupil = int(dat[i, 1])
            m_field = int(dat[i, 3])
            j_pupil = noll[int(dat[i, 0]), m_pupil]
            j_field = noll[int(dat[i, 2]), m_field]
            for mp in range(np.min((m_pupil, 2))):
                for mf in range(np.min((m_field, 2))):
                    self.aberrations[j_pupil+mp, j_field+mp] = dat[i, 4]
        return None

    def _get_phase_screens(self):
        # optics PSF
        op_model = galsim.OpticalScreenField(self.aberrations,
                                             diam=self.tel_diam_m,
                                             fov_radius=self.fov_deg*galsim.degrees)
        screens = galsim.PhaseScreenList(op_model)
        return screens

    def get_wavefront(self, theta=(0.*galsim.arcmin, 0.*galsim.arcmin)):
        screens = self._get_phase_screens()
        wf = screens.wavefront(self.aper.u, self.aper.v, t=0, theta=theta)
        wf_out = np.zeros_like(wf)
        wf_out[self.aper.illuminated] = wf[self.aper.illuminated]
        return wf_out

    def get_model(self, theta=(0.*galsim.arcmin, 0.*galsim.arcmin), with_atmos=True):
        screens = self._get_phase_screens()

        optics = screens.makePSF(lam=self.wavelength_nm, aper=self.aper, 
                                 theta=theta, pad_factor=1., ii_pad_factor=4.,
                                 gsparams=self.gsparams)

        if with_atmos:
            atmos = galsim.Kolmogorov(fwhm=self.params.psf_fwhm[0])
            psf_shape = galsim.Shear(e1=self.params.psf_e1[0],
                                     e2=self.params.psf_e2[0])
            atmos = atmos.shear(psf_shape)            
            psf = galsim.Convolve([atmos, optics])
        else:
            psf = optics
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
