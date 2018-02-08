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
jiffy stooker.py

Collect Roaster output parameter samples across many galaxies into a thinned set
for Thresher
"""
import os
from argparse import ArgumentParser
import numpy as np
import h5py


def load_roaster_samples(roaster_outfile, igal):
    """
    Load parameter samples for one galaxy from Roaster output file

    @param roaster_outfile  Name of the Roaster output file
    @param igal             Index of the footprint to load
    """
    # print "loading ", roaster_outfile
    hfile = h5py.File(roaster_outfile, 'r')
    grp = hfile['Samples/footprint{:d}'.format(igal)]
    paramnames = grp['post'].attrs['paramnames']
    data = grp['post'][...]
    lnps = grp['logprobs'][...]
    hfile.close()
    return data, lnps, paramnames


def get_summary_stats_per_gal(samps, paramnames, verbose=False):
    e1ndx = np.where(paramnames == 'e1')[0][0]
    e2ndx = np.where(paramnames == 'e2')[0][0]
    e1mean = np.mean(samps[:, :, e1ndx].ravel())
    e2mean = np.mean(samps[:, :, e2ndx].ravel())
    e1std = np.std(samps[:, :, e1ndx].ravel())
    e2std = np.std(samps[:, :, e2ndx].ravel())
    if verbose:
        # print "e1ndx: ", e1ndx, " e2ndx:", e2ndx
        print "e1 = {:4.3g} +/- {:4.3g}, e2 = {:4.3g} +/- {:4.3g}".format(
            e1mean, e2mean, e1std, e2std)
    return e1mean, e1std, e2mean, e2std


def main():
    parser = ArgumentParser()

    parser.add_argument('infns', metavar='input-files', nargs='+',
                        help='Input files (HDF5)')

    parser.add_argument('-o', dest='outfn', help='Output filename')

    parser.add_argument('--nsamples_out', type=int, default=200,
                        help="Desired number of samples per galaxy to save in" +
                        "the output file.")

    args = parser.parse_args()

    means = np.zeros(2, dtype=np.float64)
    std_devs = np.zeros(2, dtype=np.float64)
    samples_out = []
    # For running variance:
    m = [0., 0.]
    s = [0., 0.]
    i = 0
    for infn in args.infns:
        # print 'Reading', infn
        if not os.path.exists(infn):
            print 'MISSING FILE -- SKIPPING'
            continue

        # ----- Load -----
        # Get 'seg' label from file name
        seg_lab = filter(lambda x: 'seg' in x, os.path.splitext(infn)[0].split("_"))[0] # output: 'seg[D]'
        seg_lab = seg_lab.split("g")[1]

        samps, lnps, paramnames = load_roaster_samples(infn, int(seg_lab))

        (nsteps, nwalkers, nparams) = samps.shape
        # print "nsteps: {:d}, nwalkers: {:d}, nparams: {:d}".format(nsteps,
        #                                                            nwalkers,
        #                                                            nparams)

        # ----- Summary statistics -----
        e1mean, e1std, e2mean, e2std = get_summary_stats_per_gal(samps,
                                                                 paramnames)
        means[0] += e1mean
        means[1] += e2mean
        # std_devs[0] += e1std
        # std_devs[1] += e2std

        if i == 0:
            m = [e1mean, e2mean]
        else:
            m = [m[j] + (means[j] - m[j])/float(i) for j in xrange(2)]
            s = [s[j] + (means[j] - s[j])*(means[j] - s[j]) for j in xrange(2)]
        if i > 1:
            std_devs = [np.sqrt(s / (i - 1)) for j in xrange(2)]

        # ----- Aggregate samples -----
        # flatten walkers and samples into single dimension
        samps = np.vstack(samps)
        ndx = np.random.random_integers(low=0, high=nsteps*nwalkers - 1,
                                        size=args.nsamples_out)
        samples_out.append(samps[ndx, :])

    means /= len(args.infns)
    #std_devs /= len(args.infns)

    samples_out = np.array(samples_out)

    outfile = h5py.File(args.outfn, 'w')
    grp = outfile.create_group('gals')
    grp.attrs['paramnames'] = paramnames
    grp.create_dataset('samples', data=samples_out)
    grp.create_dataset('means', data=means)
    grp.create_dataset("std_devs", data=std_devs)
    outfile.close()
    del outfile

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
