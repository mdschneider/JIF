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
from setuptools import setup

setup(name='jif',
      version='1.0',
      description='Joint Image Framework for galaxy modeling',
      url='https://github.com/mdschneider/JIF',
      author='Michael D. Schneider, William A. Dawson',
      author_email='schneider42@llnl.gov',
      license='GNU',
      packages=['jiffy'],
      # package_data={'jif': ['input/*.dat', 'input/*.sed']},
      # The python scripts in this package can be designated as distinct
      # command-line executables here. Be careful about potential name clashes.
      entry_points={
          'console_scripts': ['jiffy_sheller=jiffy.sheller:main',
                              'jiffy_roaster=jiffy.roaster:main',
                              'jiffy_roaster_inspector=jiffy.roaster_inspector:main',
                              'jiffy_stooker=jiffy.stooker:main'],
      },
      zip_safe=False)
