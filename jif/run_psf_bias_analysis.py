#!/usr/bin/env python
# encoding: utf-8
"""
run_psf_bias_analysis.py

Created by Michael Schneider on 2016-01-27
"""

import argparse
import sys
import os
import h5py
import numpy as np
# import matplotlib.pyplot as plt

import galsim
import segments
import galsim_galaxy as gg
import psf_model as pm
import Roaster

import logging


# Print log messages to screen:
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Print log messages to file:
#logging.basicConfig(filename='log_run_psf_bias_analysis.py.txt',
#                     level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

class RoasterArgs(object):
    def __init__(self, infiles=[], outfile="../output/psf_bias/roaster_out"):
        self.infiles=infiles
        self.segment_numbers = [0]
        self.outfile = outfile
        self.galaxy_model_type = "Spergel"
        self.data_format = "jif_segment"
        self.telescope = "LSST"
        self.seed = 543726
        self.nsamples = 1000
        self.nwalkers = 32
        self.nburn = 50
        self.nthreads = 1
        self.quiet = True


def create_model_image(filter_name_ground='r', file_lab='',
                       galaxy_model="Spergel"):
    ngrid_lsst = 70
    # LSST
    print("\n----- LSST -----")
    lsst = gg.GalSimGalaxyModel(
        telescope_name="LSST",
        pixel_scale_arcsec=0.2,
        noise=gg.lsst_noise(82357),
        galaxy_model=galaxy_model,
        wavelength_meters=gg.k_lsst_filter_central_wavelengths[filter_name_ground] * 1.e-9,
        primary_diam_meters=8.4,
        filter_names=gg.k_lsst_filter_names,
        filter_wavelength_scale=1.0,
        atmosphere=True)
    # lsst.params[0].flux_sed1 = 1.e4

    # Save the image
    lsst.save_image("../output/psf_bias/model_image" + file_lab + ".fits",
        filter_name=filter_name_ground,
        out_image=galsim.Image(ngrid_lsst, ngrid_lsst))
    lsst.plot_image("../output/psf_bias/model_image" + file_lab + ".png",
        ngrid=ngrid_lsst,
        filter_name=filter_name_ground, title="LSST " + filter_name_ground)
    # Save the corresponding PSF
    lsst.save_psf("../output/psf_bias/model_psf" + file_lab + ".fits",
        ngrid=ngrid_lsst/4)
    lsst.plot_psf("../output/psf_bias/model_psf" + file_lab + ".png",
        ngrid=ngrid_lsst/4, title="LSST " + filter_name_ground)

    lsst_data = lsst.get_image(galsim.Image(ngrid_lsst, ngrid_lsst), add_noise=True,
        filter_name=filter_name_ground).array

    # -------------------------------------------------------------------------
    ### Save a file with joint image data for input to the Roaster
    segfile = os.path.join(os.path.dirname(__file__),
        '../output/psf_bias/model_image_data' + file_lab + '.h5')
    print("Writing {}".format(segfile))
    seg = segments.Segments(segfile)

    seg_ndx = 0
    src_catalog = lsst.params
    seg.save_source_catalog(src_catalog, segment_index=seg_ndx)

    dummy_mask = 1.0
    dummy_background = 0.0

    ### Ground data
    seg.save_images([lsst_data], [lsst.noise.getVariance()], [dummy_mask],
        [dummy_background], segment_index=seg_ndx,
        telescope='lsst',
        filter_name=filter_name_ground)
    seg.save_tel_metadata(telescope='lsst',
        primary_diam=lsst.primary_diam_meters,
        pixel_scale_arcsec=lsst.pixel_scale,
        atmosphere=lsst.atmosphere)
    seg.save_psf_images([lsst.get_psf_image().array], segment_index=seg_ndx,
        telescope='lsst',
        filter_name=filter_name_ground)
    gg.save_bandpasses_to_segment(seg, lsst, gg.k_lsst_filter_names, "LSST")
    return None

def main():
    parser = argparse.ArgumentParser(
        description='Run Roaster on simulated data with varying degrees of PSF bias')

    args = parser.parse_args()

    psf_fwhm_fiducial = 0.6

    ### Create model image
    create_model_image()
    image_file = "../output/psf_bias/model_image_data.h5"
    roaster_args = RoasterArgs(infiles=[image_file])

    galaxy_parameters = ['nu', 'hlr', 'e', 'beta', 'flux_sed1', 'dx', 'dy']

    ### Run Roaster for different fiducial PSFs (biased to varying degrees
    ### with respect to the truth)
    it = 0
    for fwhm_bias in [0.0, 0.01, 0.1, 0.5]:
        for bias_sign in [1., -1., 0.]:
            print "\n======================================"
            print "iteration: {:d}, bias: {:4.3f}\n".format(it, fwhm_bias)

            if np.abs(bias_sign) < 0.9:
                sampling_parameters = galaxy_parameters + ['psf_fwhm', 'psf_e', 'psf_beta']
            sampling_parameters = galaxy_parameters
            roaster_args.model_params = sampling_parameters

            ### Need to reset module level lists in Roaster for multiple
            ### iterations in this loop
            Roaster.pixel_data = []
            Roaster.pix_noise_var = []
            Roaster.src_models = []

            logging.debug("Instantiating Roaster object")
            roast = Roaster.Roaster(debug=False,
                                    data_format='jif_segment',
                                    lnprior_omega=Roaster.DefaultPriorSpergel(),
                                    lnprior_Pi=pm.DefaultPriorPSF(),
                                    galaxy_model_type="Spergel",
                                    model_paramnames=sampling_parameters,
                                    telescope="LSST")
            logging.debug("Calling Roaster.Load")
            roast.Load(image_file, use_PSFModel=True)

            psf_fwhm = (1. + bias_sign * fwhm_bias) * psf_fwhm_fiducial
            roast.set_param_by_name("psf_fwhm", psf_fwhm)

            if bias_sign > 0.5:
                sgn_lab = "p"
            elif bias_sign < -0.5:
                sgn_lab = "m"
            else:
                sgn_lab = "marg"
            outfile = "../output/psf_bias/roaster_out_fwhm_bias_{}{:4.3f}".format(
                sgn_lab, fwhm_bias)
            roaster_args.outfile = outfile
            logging.debug("Calling Roaster sampling")
            Roaster.do_sampling(roaster_args, roast)
            it += 1


    ### Make plots of posterior galaxy ellipticity

    return 0


if __name__ == "__main__":
    sys.exit(main())
