#!/usr/bin/env python
# encoding: utf-8
"""
telescopes.py

Settings for telescope-specific modeling
"""
import os
import numpy as np
import galsim
import galsim.wfirst

### Some parameters for telescopes that JIF knows about
k_telescopes = {
    "LSST": {
        "primary_diam_meters": 8.4,
        "effective_diameter": 6.4, # meters
        "obscuration": 0.548,
        "pixel_scale": 0.2,        # arcseconds / pixel
        ### Exposure time for defining the zero point reference
        "exptime_zeropoint": 30.,  # seconds
        "zeropoint": 'AB',
        "gain": 1.0,
        "filter_names": 'ugrizy',
        "filter_central_wavelengths": {'u':360., 'g':500., 'r':620., 'i':750.,
                                       'z':880., 'y':1000.},
        ### Reference filter name for defining the magnitude model parameter
        "ref_filter_mag_param": 'r',
        "atmosphere": True
    },
    "WFIRST": {
        "primary_diam_meters": galsim.wfirst.diameter,
        "effective_diameter": galsim.wfirst.diameter * (1. - galsim.wfirst.obscuration), # meters
        "obscuration": galsim.wfirst.obscuration,
        "pixel_scale": galsim.wfirst.pixel_scale,       # arcseconds / pixel
        ### Exposure time for defining the zero point reference
        "exptime_zeropoint": galsim.wfirst.exptime, # seconds
        "zeropoint": 'AB',
        "gain": galsim.wfirst.gain,
        ### Only these 4 filters will be used for the high-latitude WL survey
        "filter_names": ['Y106', 'J129', 'H158', 'F184'],
        "filter_central_wavelengths": {'r':620., 'Z087':867., 'Y106':1100.,
                                       'J129':1300., 'H158':994., 'F184':1880., 
                                       'W149':1410.},
        ### Reference filter name for defining the magnitude model parameter.
        ### GalSimGalaxyModel defines an 'r' filter for all telescopes so we can do consistent
        ### parameter definitions, even if 'r' is not actually availabele for this survey.
        "ref_filter_mag_param": 'r',
        "atmosphere": False
    },
    "SDSS": {
        "primary_diam_meters": 2.5,
        "effective_diameter": 1.2,
        "obscuration": 0.52,
        "pixel_scale": 0.396,
        ### Exposure time for defining the zero point reference
        "exptime_zeropoint": 54.,
        "zeropoint": 'AB',
        "gain": 9.36, # http://classic.sdss.org/dr7/dm/flatFiles/opECalib.html
        "filter_names": 'ugriz',
        "filter_central_wavelengths": {'u':360., 'g':500., 'r':620., 'i':750.,
                                       'z':880.},
        "ref_filter_mag_param": 'r',
        "atmosphere": True
    }
}


k_lsst_filter_names = 'ugrizy'
### 'Central' passband wavelengths in nanometers
k_lsst_filter_central_wavelengths = {'u':360., 'g':500., 'r':620., 'i':750.,
                                'z':880., 'y':1000.}

k_wfirst_filter_names = ['Z087', 'Y106', 'J129', 'H158', 'F184', 'W149']
### 'Central' passband wavelengths in nanometers
k_wfirst_filter_central_wavelengths = {'r':620., 'Z087':867., 'Y106':1100.,
    'J129':1300., 'H158':994., 'F184':1880., 'W149':1410.}

k_sdss_filter_names = 'ugriz'
### 'Central' passband wavelengths in nanometers
k_sdss_filter_central_wavelengths = {'u':360., 'g':500., 'r':620., 'i':750.,
                                     'z':880.}

### TESTING
# k_wfirst_filter_names = k_lsst_filter_names
# k_wfirst_filter_central_wavelengths = k_lsst_filter_central_wavelengths

def lsst_noise(random_seed, gain=2.1, read_noise=3.6, sky_level=720.):
    """
    See GalSim/examples/lsst.yaml

    gain: e- / ADU
    read_noise: rms of read noise in electrons (if gain > 0)
    sky_level: ADU / pixel
    """
    rng = galsim.BaseDeviate(random_seed)
    return galsim.CCDNoise(rng,
                           gain=gain,
                           read_noise=read_noise,
                           sky_level=sky_level)


def wfirst_noise(random_seed):
    """
    Deprecated in favor of GalSim WFIRST module

    From http://wfirst-web.ipac.caltech.edu/wfDepc/visitor/temp1927222740/results.jsp
    """
    rng = galsim.BaseDeviate(random_seed)
    exposure_time_s = 150.
    pixel_scale_arcsec = 0.11
    read_noise_e_rms = 0.5 #5.
    sky_background = 3.6e-2 #3.60382E-01 # e-/pix/s
    gain = galsim.wfirst.gain # e- / ADU
    return galsim.CCDNoise(rng, gain=gain,
        read_noise=(read_noise_e_rms / gain) ** 2,
        sky_level=sky_background / pixel_scale_arcsec ** 2 * exposure_time_s)


def wfirst_sky_background(filter_name, bandpass):
    """
    Calculate the approximate sky background in e-/pixel using the GalSim
    WFIRST module
    """
    sky_level = galsim.wfirst.getSkyLevel(bandpass)
    sky_level *= (1.0 + galsim.wfirst.stray_light_fraction)
    ### Approximate sky level in e-/pix, ignoring variable pixel scale
    ### See GalSim demo13.py
    sky_level *= galsim.wfirst.pixel_scale**2
    sky_level += galsim.wfirst.thermal_backgrounds[filter_name]*galsim.wfirst.exptime
    return sky_level

def load_filter_file_to_bandpass(table, wavelength_scale=1.0,
                                 effective_diameter_meters=6.4,
                                 exptime_sec=30.):
    """
    Create a Galsim.Bandpass object from a lookup table

    @param table Either (1) the name of a file for reading the lookup table
                 values for a bandpass, or (2) an instance of a
                 galsim.LookupTable
    @param wavelength_scale The multiplicative scaling of the wavelengths in the
                            input bandpass file to get units of nm (not used if
                            table argument is a LookupTable instance)
    @param effective_diameter_meters The effective diameter of the telescope
                                     (including obscuration) for the zeropoint
                                     calculation
    @param exptime_sec The exposure time for the zeropoint calculation
    """
    if isinstance(table, str):
        dat = np.loadtxt(table)
        table = galsim.LookupTable(x=dat[:,0]*wavelength_scale, f=dat[:,1])
    elif not isinstance(table, galsim.LookupTable):
        raise ValueError("table must be a file name or galsim.LookupTable")
    bp = galsim.Bandpass(table, wave_type='nm')
    bp = bp.thin(rel_err=1e-4)
    return bp.withZeropoint(zeropoint='AB',
        effective_diameter=effective_diameter_meters,
        exptime=exptime_sec)

def load_filter_files(wavelength_scale=1.0, telescope_name="LSST"):
    """
    Load filters for drawing chromatic objects.

    Makes use of the module-level dictionary `k_telescopes` with values for
    setting the zeropoints. Specifically, the type of zeropoint ('AB'),
    the effective diameter of the telescope, and the exposure time.

    Adapted from GalSim demo12.py

    @param wavelength_scale     Multiplicative scaling of the wavelengths
                                input from the filter files to get
                                nanometers from whatever the input units are
    @param telescope_name       Name of the telescope model ("LSST" or "WFIRST")
    """
    if telescope_name == "WFIRST":
        ### Use the Galsim WFIRST module
        filters = galsim.wfirst.getBandpasses(AB_zeropoint=True)
    else:
        ### Use filter information in this module
        path, filename = os.path.split(__file__)
        datapath = os.path.abspath(os.path.join(path, "input/"))
        filters = {}
        for filter_name in k_telescopes[telescope_name]['filter_names']:
            filter_filename = os.path.join(datapath, '{}_{}.dat'.format(
                telescope_name, filter_name))
            filters[filter_name] = load_filter_file_to_bandpass(
                filter_filename, wavelength_scale,
                k_telescopes[telescope_name]['effective_diameter'],
                k_telescopes[telescope_name]['exptime_zeropoint']
            )
    return filters

