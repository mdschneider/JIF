#!/usr/bin/env python
# encoding: utf-8
"""
roaster_wrapper.py

Wrap the Jiffy roaster module to set initial parameters to known truth values
"""
import numpy as np
import argparse
import os
import numpy as np
from astropy.io import fits
import footprints
import jiffy

def apply_shear(e, g):
    return (e + g) / (1.0 + g.conjugate() * e)

def get_truths(ifield, igal, workdir):
    """
    Get the true parameter values
    """
    scale = 0.2 # arcseconds

    infile = os.path.join(workdir, "control/ground/constant",
                          "epoch_catalog-{0:0>3}-0.fits".format(ifield))
    hdulist = fits.open(infile)
    tbdata = hdulist[1].data

    nu = tbdata.field('gal_nu')[igal]
    hlr = tbdata.field('gal_hlr')[igal]
    flux = tbdata.field('gal_flux')[igal]

    g1 = tbdata.field('g1')[igal]
    g2 = tbdata.field('g2')[igal]
    # Convert offsets in the truth catalog from pixels to arcseconds 
    # (which are the units expected by the galsim shift() method)
    dx = tbdata.field('dx')[igal] * scale
    dy = tbdata.field('dy')[igal] * scale

    e1int = tbdata.field('gal_e1')[igal]
    e2int = tbdata.field('gal_e2')[igal]

    e_sh = apply_shear(e1int + 1j*e2int, g1 + 1j*g2)

    e1 = e_sh.real
    e2 = e_sh.imag

    truths = np.array([e1, e2, hlr, flux, nu, dx, dy])
    return truths

def split_list_across_tasks(n_items, n_tasks):
    if n_tasks <= n_items:
        n = int(n_items) / int(n_tasks)
        n_per_task = [n for i in xrange(n_tasks)]
        remainder = n_items - (n * n_tasks)
        for i in xrange(remainder):
            n_per_task[i] += 1
    else:
        n_per_task = [0] * n_tasks
        for i in xrange(n_items):
            n_per_task[i] += 1
    return n_per_task

class _close_pool_message(object):
    def __repr__(self):
        return "<Close pool message>"

def main():
    import mpi4py.MPI
    comm = mpi4py.MPI.COMM_WORLD
    rank = comm.Get_rank()
    ntasks = comm.Get_size()
    master = (rank == 0)
    comm.Barrier()

    print "Starting with rank {:d} and {:d} tasks".format(rank, ntasks)

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str,
                        default="jiffy_cgc1.yaml",
                        help="Name of a configuration file listing inputs." +
                        "If specified, ignore other command line flags.")

    parser.add_argument("--n_fields", type=int, default=200)

    parser.add_argument("--n_gals", type=int, default=10000)

    parser.add_argument("--workdir", type=str, default="./")

    parser.add_argument("--gal_start", type=int, default=0,
                        help="Optional starting offset in the galaxy list")

    parser.add_argument("--field_start", type=int, default=0,
                        help="Optional starting offset in the field list")

    args = parser.parse_args()

    ntasks = 2

    # Loop over fields and galaxies within a field
    n_per_task = split_list_across_tasks(args.n_gals * args.n_fields - args.gal_start, ntasks)
    print "n_per_task:", n_per_task
    displacements = np.insert(np.cumsum(n_per_task), 0, 0)
    print "displacements:", displacements
    i_start = displacements[rank]
    i_stop = displacements[rank + 1]

    for i in xrange(i_start, i_stop):
        segnum = np.mod(i, args.n_gals)
        field = int(float(i) / float(args.n_gals))

        if field == 0:
            segnum += args.gal_start
        field += args.field_start

        infile = os.path.join(args.workdir,
                              "control/ground/constant/segments/seg_{:03d}.h5".format(field))
        rstr = jiffy.Roaster(args.config_file)

        # print "{:d} -- field {:d}, seg {:d}, infile: {}".format(rank, field, segnum, infile)  

        dat, noise_var, scale, gain = footprints.load_image(infile, segment=segnum)
        rstr.import_data(dat, noise_var, scale=scale, gain=gain)

        truths = get_truths(field, segnum, args.workdir)
        rstr.set_params(truths)

        jiffy.roaster.do_sampling(args, rstr)

    comm.Barrier()
    ###pool.close()
    # if master:
    #     for i in range(1, ntasks):
    #         comm.isend(_close_pool_message(), dest=i)

    return None

if __name__ == '__main__':
    main()
