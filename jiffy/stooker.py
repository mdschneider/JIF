#!/usr/bin/env python
# encoding: utf-8
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
    hfile = h5py.File(roaster_outfile, 'r')
    grp = hfile['Samples/footprint{:d}'.format(igal)]
    paramnames = grp['post'].attrs['paramnames']
    data = grp['post'][...]
    hfile.close()
    return data, paramnames


def get_summary_stats_per_gal(samps, paramnames, verbose=True):
    e1ndx = np.where(paramnames == 'e1')[0][0]
    e2ndx = np.where(paramnames == 'e2')[0][0]
    e1mean = np.mean(samps[:, :, e1ndx].ravel())
    e2mean = np.mean(samps[:, :, e2ndx].ravel())
    e1std = np.std(samps[:, :, e1ndx].ravel())
    e2std = np.std(samps[:, :, e2ndx].ravel())
    if verbose:
        print "e1ndx: ", e1ndx, " e2ndx:", e2ndx
        print "e1 = {:4.3g} +/- {:4.3g}, e2 = {:4.3g} +/- {:4.3g}".format(
            e1mean, e2mean, e1std, e2std)
    return e1mean, e1std, e2mean, e2std


def main():
    parser = ArgumentParser()

    parser.add_argument('infns', metavar='input-files', nargs='+',
                        help='Input files (HDF5)')

    parser.add_argument('-o', dest='outfn', help='Output filename')

    args = parser.parse_args()

    means = np.zeros(2, dtype=np.float64)
    std_devs = np.zeros(2, dtype=np.float64)
    for i, infn in enumerate(args.infns):
        print 'Reading', infn
        if not os.path.exists(infn):
            print 'MISSING FILE -- SKIPPING'
            continue

        # -----
        # Get 'seg' label from file name
        seg_lab = int(filter(lambda x: 'seg' in x, infn.split("_"))[0][3])

        samps, paramnames = load_roaster_samples(infn, seg_lab)

        (nsteps, nwalkers, nparams) = samps.shape
        print "nsteps: {:d}, nwalkers: {:d}, nparams: {:d}".format(nsteps,
            nwalkers, nparams)

        # -----
        e1mean, e1std, e2mean, e2std = get_summary_stats_per_gal(samps,
                                                                 paramnames)
        means[0] += e1mean
        means[1] += e2mean
        std_devs[0] += e1std
        std_devs[1] += e2std

    means /= len(args.infns)
    std_devs /= len(args.infns)

    outfile = h5py.File(args.outfn, 'w')
    grp = outfile.create_group('gals')
    grp.attrs['paramnames'] = paramnames
    grp.create_dataset('means', data=means)
    grp.create_dataset("std_devs", data=std_devs)
    outfile.close()
    del outfile

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
