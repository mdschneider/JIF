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
import jiffy


class TestGalSimGalaxyModel(unittest.TestCase):

    def test_init_values(self):
        gg = jiffy.GalsimGalaxyModel()
        self.assertEqual(gg.static_psf.__class__.__name__, "Transformation")
        self.assertAlmostEqual(gg.params.nu[0], 0.5)
        self.assertAlmostEqual(gg.params.hlr[0], 1.0)
        self.assertAlmostEqual(gg.params.e1[0], 0.0)
        self.assertAlmostEqual(gg.params.e2[0], 0.0)
        self.assertAlmostEqual(gg.params.flux[0], 1.0)
        self.assertAlmostEqual(gg.params.dx[0], 0.0)
        self.assertAlmostEqual(gg.params.dy[0], 0.0)
        self.assertEqual(gg.sample_psf, False)
        # self.assertAlmostEqual(gg.static_psf.fwhm, 0.6)
        # self.assertAlmostEqual(gg.static_psf.flux, 1.0)
        self.assertAlmostEqual(gg.psf_model.params.psf_fwhm, 0.6)
        self.assertAlmostEqual(gg.psf_model.params.psf_flux, 1.0)
        self.assertAlmostEqual(gg.psf_model.params.psf_e1, 0.0)
        self.assertAlmostEqual(gg.psf_model.params.psf_e2, 0.0)
        self.assertAlmostEqual(gg.psf_model.params.psf_dx, 0.0)
        self.assertAlmostEqual(gg.psf_model.params.psf_dy, 0.0)

    def test_set_params(self):
        gg = jiffy.GalsimGalaxyModel()
        gg.set_params([0.1, -0.5])
        self.assertAlmostEqual(gg.params.e1[0], 0.1)
        self.assertAlmostEqual(gg.params.e2[0], -0.5)
        with self.assertRaises(AssertionError):
            gg.set_params([0.1])

    def test_default_image(self):
        gg = jiffy.GalsimGalaxyModel()
        image = gg.get_image(64, 64)
        self.assertAlmostEqual(image.array.sum(), 0.9983294)
        self.assertAlmostEqual(image.array.max(), 0.0094704079)

if __name__ == "__main__":
    unittest.main()
