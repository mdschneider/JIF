import os
import numpy as np
import galsim

exposure_time = 15.
magnitude_i_band = 22.


path, filename = os.path.split(__file__)
# datapath = os.path.abspath(os.path.join(path, "data/"))
datapath = os.path.expanduser("~/code/GalSim/examples/data")
outpath = os.path.abspath(os.path.join(path, "output/"))

# ==============================================================================
# read in the LSST filters
filter_names = 'ugrizy'
filters = {}
for filter_name in filter_names:
    filter_filename = os.path.join(datapath, 'LSST_{0}.dat'.format(filter_name))
    # Here we create some galsim.Bandpass objects to represent the filters we're observing
    # through.  These include the entire imaging system throughput including the atmosphere,
    # reflective and refractive optics, filters, and the CCD quantum efficiency.  These are
    # also conveniently read in from two-column ASCII files where the first column is
    # wavelength and the second column is dimensionless flux. The example filter files have
    # units of nanometers and dimensionless throughput, which is exactly what galsim.Bandpass
    # expects, so we just specify the filename.
    filters[filter_name] = galsim.Bandpass(filter_filename)
    # For speed, we can thin out the wavelength sampling of the filter a bit.
    # In the following line, `rel_err` specifies the relative error when integrating over just
    # the filter (however, this is not necessarily the relative error when integrating over the
    # filter times an SED)
    filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)
    # Set the zero point for a given exposure time
    filters[filter_name] = filters[filter_name].withZeropoint(
        zeropoint='AB', effective_diameter=6.4 * 100.,
        exptime=exposure_time)

# ==============================================================================
# From demo12
SED_names = ['CWW_E_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext', 'CWW_Im_ext']
SEDs = {}
for SED_name in SED_names:
    SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
    # Here we create some galsim.SED objects to hold star or galaxy spectra.  The most
    # convenient way to create realistic spectra is to read them in from a two-column ASCII
    # file, where the first column is wavelength and the second column is flux. Wavelengths in
    # the example SED files are in Angstroms, flux in flambda.  The default wavelength type for
    # galsim.SED is nanometers, however, so we need to override by specifying
    # `wave_type = 'Ang'`.
    SED = galsim.SED(SED_filename, wave_type='Ang')
    # The normalization of SEDs affects how many photons are eventually drawn into an image.
    # One way to control this normalization is to specify the flux density in photons per nm
    # at a particular wavelength.  For example, here we normalize such that the photon density
    # is 1 photon per nm at 500 nm.
    # SEDs[SED_name] = SED.withFluxDensity(target_flux_density=1.0, wavelength=500)
    SEDs[SED_name] = SED.withMagnitude(magnitude_i_band, filters['i'])

# ==============================================================================

filter_name = 'i'
SED = SEDs['CWW_E_ext']
print "{}-band magnitude: {:8.6g}".format(
    filter_name, SED.calculateMagnitude(filters[filter_name]))
print "{}-band flux: {:8.6g}".format(
    filter_name, SED.calculateFlux(filters[filter_name]))

# mono_gal = galsim.Spergel(nu=0.3, half_light_radius=1.5, flux=1.0)
# gal = galsim.Chromatic(mono_gal, SED)
