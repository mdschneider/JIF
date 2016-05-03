#!/usr/bin/env python
# encoding: utf-8
"""
telescopes.py

Settings for telescope-specific modeling
"""
import galsim
import galsim.wfirst

### Some parameters for telescopes that JIF knows about
k_telescopes = {
    "LSST": {
        "effective_diameter": 6.4, # meters
        "pixel_scale": 0.2,        # arcseconds / pixel
        # Exposure time for defining the zero point reference
        "exptime_zeropoint": 30.,  # seconds
        "zeropoint": 'AB',
        # Referenc filter name for defining the magnitude model parameter
        "ref_filter_mag_param": 'r'
    },
    "WFIRST": {
        "effective_diameter": galsim.wfirst.diameter * (1. - galsim.wfirst.obscuration), # meters
        "pixel_scale": galsim.wfirst.pixel_scale,       # arcseconds / pixel
        # Exposure time for defining the zero point reference
        "exptime_zeropoint": galsim.wfirst.exptime, # seconds
        "zeropoint": 'AB',
        # Referenc filter name for defining the magnitude model parameter
        "ref_filter_mag_param": 'r'
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
    gain = 2.1 # e- / ADU
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
