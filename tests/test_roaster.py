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
import unittest
import numpy as np
import jiffy


def ln_gaus(xval, ln_amp, x_mean, sigma):
    """
    Evaluate the log of a Gaussian density function (unnormalized)
    """
    return ln_amp - 0.5 * (xval - x_mean)**2 / sigma**2


def fit_gaussian(dat_x, dat_y):
    """
    Fit a 1D Gaussian to input data
    """
    from scipy.optimize import curve_fit

    params = [-2000, 0., 0.001]

    popt, pcov = curve_fit(ln_gaus, dat_x, dat_y, p0=params)
    return popt, pcov


class TestRoaster(unittest.TestCase):
    """ Unit tests for the Roaster class in Jiffy
    """

    def test_init(self):
        """ Check that all values in the Roaster init method are unchanged
        """
        rstr = jiffy.Roaster("test.yaml")
        self.assertEqual(len(rstr.src_models), 1)
        self.assertEqual(rstr.src_models[0].__class__.__name__,
                          "GalsimGalaxyModel")
        self.assertEqual(rstr.ngrid_x, 64)
        self.assertEqual(rstr.ngrid_y, 64)
        self.assertAlmostEqual(rstr.noise_var, 3e-10)

    def test_make_data(self):
        """ Check that the Roaster make_data() method yields constant summary
        statisticss
        """
        rstr = jiffy.Roaster("test.yaml")
        rstr.make_data()
        self.assertAlmostEqual(rstr.data.var(), 8.02857e-07)

    def test_shear_bias(self):
        """
        Check that the shear bias (m, c) parameters are below threshold for
        a series of simulated images.

        This takes a bit longer, but is the key measure of our code performance
        """
        from scipy.stats import linregress

        rstr = jiffy.Roaster("test.yaml")
        rstr.make_data()

        g_true = np.linspace(-0.3, 0.3, 60)

        bias = np.zeros((len(g_true), 2), dtype=np.float64)
        for i, shear_val in enumerate(g_true):
            # if np.mod(i, 10) == 0:
            #     print "--- {:d} / {:d} ---".format(i, len(g_true))
            rstr.src_models[0].set_params([0.0, shear_val])
            rstr.make_data()
            shears = np.linspace(shear_val - 0.01, shear_val + 0.01, 100)
            lnp = np.array([rstr([0.0, g_val]) for g_val in shears])
            popt, _ = fit_gaussian(shears, lnp)
            bias[i, 0] = popt[1] - shear_val
            bias[i, 1] = popt[2]

        slope, intercept, _, _, std_err = linregress(g_true, bias[:, 0])
        self.assertLess(np.abs(slope), 1.e-3)
        self.assertLess(std_err, 3.e-3)
        self.assertLess(np.abs(intercept), 5.e-4)


if __name__ == "__main__":
    unittest.main()
