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
import galsim.lsst


class TestGalSimPSFModel(unittest.TestCase):

    def test_lsst_aberrations_input(self):
        psf = jiffy.GalsimPSFLSST()
        dat = galsim.lsst.lsst_psfs._read_aberrations()

        for i in xrange(dat.shape[0]):
            jp = jiffy.galsim_psf.get_noll_index(dat[i,0], -dat[i,1])
            jf = jiffy.galsim_psf.get_noll_index(dat[i,2], -dat[i,3])
            self.assertAlmostEqual(dat[i,4], psf.aberrations[jp, jf])
            # print dat[i,1], dat[i,3], dat[i,4], psf.aberrations[jp, jf]