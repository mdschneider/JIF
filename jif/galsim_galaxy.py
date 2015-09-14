#!/usr/bin/env python
# encoding: utf-8
"""
galsim_galaxy.py

Wrapper for GalSim galaxy models to use in MCMC.
"""
import os
import math
import numpy as np
from operator import add
import galsim


k_SED_names = ['CWW_E_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext', 'CWW_Im_ext']
k_filter_names = 'ugrizy'
k_filter_central_wavelengths = {'u':360., 'g':500., 'r':620., 'i':750., 'z':880., 'y':1000.}

### Minimum value a flux parameter can take, since these get log-transformed
k_flux_param_minval = 1.e-12


k_spergel_paramnames = ['nu', 'hlr', 'e', 'beta']

### Numpy composite object types for the model parameters for galaxy images under different
### modeling assumptions.
k_galparams_type_sersic = [('redshift', '<f8'), ('n', '<f8'), ('hlr', '<f8'), ('e', '<f8'),
                           ('beta', '<f8')]
k_galparams_type_sersic += [('flux_sed{:d}'.format(i+1), '<f8') for i in xrange(len(k_SED_names))]

k_galparams_type_spergel = [('redshift', '<f8')] + [(p, '<f8') for p in k_spergel_paramnames]
k_galparams_type_spergel += [('flux_sed{:d}'.format(i+1), '<f8') for i in xrange(len(k_SED_names))]

k_galparams_type_bulgedisk = [('redshift', '<f8')]
k_galparams_type_bulgedisk += [('{}_bulge'.format(p), '<f8') for p in k_spergel_paramnames]
k_galparams_type_bulgedisk += [('{}_disk'.format(p), '<f8') for p in k_spergel_paramnames]
k_galparams_type_bulgedisk += [('flux_sed{:d}_bulge'.format(i+1), '<f8')
    for i in xrange(len(k_SED_names))]
k_galparams_type_bulgedisk += [('flux_sed{:d}_disk'.format(i+1), '<f8')
    for i in xrange(len(k_SED_names))]


k_galparams_types = {
    "Sersic": k_galparams_type_sersic,
    "Spergel": k_galparams_type_spergel,
    "BulgeDisk": k_galparams_type_bulgedisk
}


### The galaxy models are initialized with these values:
k_galparams_defaults = {
    "Sersic": [(1., 3.4, 1.8, 0.3, np.pi/4, 1.e5, k_flux_param_minval,
        k_flux_param_minval, k_flux_param_minval)],
    "Spergel": [(1., -0.3, 1.8, 0.3, np.pi/4, 1.e5, k_flux_param_minval,
        k_flux_param_minval, k_flux_param_minval)],
    "BulgeDisk": [(1.,
        0.5, 0.6, 0.05, 0.0,
        -0.6, 1.8, 0.3, np.pi/4,
        2.e4, k_flux_param_minval, k_flux_param_minval, k_flux_param_minval,
        k_flux_param_minval, 1.e4, k_flux_param_minval, k_flux_param_minval)]
}


def lsst_noise(random_seed):
    """
    See GalSim/examples/lsst.yaml

    gain: e- / ADU
    read_noise: Variance in ADU^2
    sky_level: ADU / arcsec^2
    """
    rng = galsim.BaseDeviate(random_seed)
    return galsim.CCDNoise(rng, gain=2.1, read_noise=3.4, sky_level=18000)


def wfirst_noise(random_seed):
    """
    From http://wfirst-web.ipac.caltech.edu/wfDepc/visitor/temp1927222740/results.jsp
    """
    rng = galsim.BaseDeviate(random_seed)
    exposure_time_s = 150.
    pixel_scale_arcsec = 0.11
    read_noise_e_rms = 5.
    sky_background = 3.60382E-01 # e-/pix/s
    gain = 2.1 # e- / ADU
    return galsim.CCDNoise(rng, gain=2.1,
        read_noise=(read_noise_e_rms / gain) ** 2,
        sky_level=sky_background / pixel_scale_arcsec ** 2 * exposure_time_s)


class GalSimGalaxyModel(object):
    """
    Parametric galaxy model from GalSim for MCMC.

    Mimics GalSim examples/demo1.py
    """
    def __init__(self,
                 psf_sigma=0.5, ### Not used
                 pixel_scale=0.11, ### arcseconds
                 noise=None,
                 galaxy_model="Spergel",
                 active_parameters=['hlr', 'e', 'beta'],
                 wavelength=1.e-6,
                 primary_diam_meters=2.4,
                 atmosphere=False):
        self.psf_sigma = psf_sigma
        self.pixel_scale = pixel_scale
        # if noise is None:
        #     noise = galsim.GaussianNoise(sigma=30.)
        self.noise = noise
        self.galaxy_model = galaxy_model
        self.active_parameters = active_parameters
        self.wavelength = wavelength
        self.primary_diam_meters = primary_diam_meters
        self.atmosphere = atmosphere

        ### Set GalSim galaxy model parameters
        self.params = np.core.records.array(k_galparams_defaults[galaxy_model],
            dtype=k_galparams_types[galaxy_model])
        self.paramtypes = k_galparams_types[galaxy_model]
        # self.paramnames = [p[0] for p in k_galparams_types[galaxy_model]]
        self.paramnames = self.active_parameters
        # self.n_params = len(self.paramnames)
        self.n_params = len(self.active_parameters)

        ### Set GalSim SED model parameters
        self._load_sed_files()
        ### Load the filters that can be used to draw galaxy images
        self._load_filter_files()

        self.gsparams = galsim.GSParams(
            folding_threshold=1.e-1, # maximum fractional flux that may be folded around edge of FFT
            maxk_threshold=2.e-2,    # k-values less than this may be excluded off edge of FFT
            xvalue_accuracy=1.e-1,   # approximations in real space aim to be this accurate
            kvalue_accuracy=1.e-1,   # approximations in fourier space aim to be this accurate
            shoot_accuracy=1.e-1,    # approximations in photon shooting aim to be this accurate
            minimum_fft_size=16)     # minimum size of ffts

    def _load_sed_files(self):
        """
        Load SED templates from files.

        Copied from GalSim demo12.py
        """
        path, filename = os.path.split(__file__)
        datapath = os.path.abspath(os.path.join(path, "../input/"))
        self.SEDs = {}
        for SED_name in k_SED_names:
            SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
            self.SEDs[SED_name] = galsim.SED(SED_filename, wave_type='Ang')
        return None

    def _load_filter_files(self):
        """
        Load filters for drawing chromatic objects.

        Copied from GalSim demo12.py
        """
        path, filename = os.path.split(__file__)
        datapath = os.path.abspath(os.path.join(path, "../input/"))
        self.filters = {}
        for filter_name in k_filter_names:
            filter_filename = os.path.join(datapath, 'LSST_{0}.dat'.format(filter_name))
            self.filters[filter_name] = galsim.Bandpass(filter_filename)
            self.filters[filter_name] = self.filters[filter_name].thin(rel_err=1e-4)
        return None

    def set_params(self, p):
        """
        Take a list of (active) parameters and set local variables.

        For use in emcee.
        """
        for ip, pname in enumerate(self.active_parameters):
            if 'flux_sed' in pname:
                ### Transform flux variables with exp -- we sample in ln(Flux)
                self.params[0][pname] = np.exp(p[ip])
            else:
                self.params[0][pname] = p[ip]
        return None

    def get_params(self):
        """
        Return a list of active model parameter values.
        """
        p = self.params[self.active_parameters].view('<f8').copy()
        ### Transform fluxes to ln(Flux) for MCMC sampling
        for ip, pname in enumerate(self.active_parameters):
            if 'flux_sed' in pname:
                p[ip] = np.log(p[ip])
        return p

    def validate_params(self):
        """
        Check that all model parameters take values inside allowed ranges.
        """
        valid_params = True
        if self.galaxy_model == "Sersic" or self.galaxy_model == "Spergel":
            if self.params[0].e < 0. or self.params[0].e > 1.:
                valid_params *= False
            for i in xrange(len(k_SED_names)):
                if self.params[0]['flux_sed{:d}'.format(i+1)] <= 0.:
                    valid_params *= False
        if self.galaxy_model == "Spergel":
            if self.params[0].nu < -0.85 or self.params[0].nu > 0.5:
                valid_params *= False
            for i in xrange(len(k_SED_names)):
                if self.params[0]['flux_sed{:d}'.format(i+1)] <= 0.:
                    valid_params *= False
        if self.galaxy_model == "BulgeDisk":
            if (self.params[0].e_bulge < 0. or self.params[0].e_bulge > 1. or
                self.params[0].e_disk < 0. or self.params[0].e_disk > 1.):
                valid_params *= False
            if (self.params[0].nu_bulge < -0.85 or self.params[0].nu_bulge > 0.8 or
                self.params[0].nu_disk < -0.85 or self.params[0].nu_disk > 0.8):
                valid_params *= False
                print self.params[0]
            for i in xrange(len(k_SED_names)):
                if self.params[0]['flux_sed{:d}_bulge'.format(i+1)] <= 0.:
                    valid_params *= False
                if self.params[0]['flux_sed{:d}_disk'.format(i+1)] <= 0.:
                    valid_params *= False
        return valid_params

    def get_psf(self):
        lam_over_diam = self.wavelength / self.primary_diam_meters
        lam_over_diam *= 206265. # arcsec
        optics = galsim.Airy(lam_over_diam, obscuration=0.548, flux=1.,
            gsparams=self.gsparams)
        if self.atmosphere:
            atmos = galsim.Kolmogorov(fwhm=0.6, gsparams=self.gsparams)
            psf = galsim.Convolve([atmos, optics])
        else:
            psf = optics
        return psf

    def get_SED(self, gal_comp='', flux_ref_wavelength=500):
        """
        Get the GalSim SED object given the SED parameters and redshift.

        This routine passes galsim_galaxy flux parameters to the GalSim SED.withFluxDensity()
        method. The flux parameters therefore have units of photons/nm at a reference wavelength
        (here defined to be 500 nm) as required by GalSim.
        """
        if len(gal_comp) > 0:
            gal_comp = '_' + gal_comp
        SEDs = [self.SEDs[SED_name].withFluxDensity(
            target_flux_density=self.params[0]['flux_sed{:d}{}'.format(i+1, gal_comp)],
            wavelength=flux_ref_wavelength).atRedshift(self.params[0].redshift)
                for i, SED_name in enumerate(self.SEDs)]
        return reduce(add, SEDs)

    def get_image(self, out_image=None, dx=0., dy=0, add_noise=False, filter_name='r', gain=1.):
        if self.galaxy_model == "Gaussian":
            # gal = galsim.Gaussian(flux=self.params.gal_flux, sigma=self.params.gal_sigma)
            # gal_shape = galsim.Shear(g=self.params.e, beta=self.params.beta*galsim.radians)
            # gal = gal.shear(gal_shape)
            raise AttributeError("Unimplemented galaxy model")

        elif self.galaxy_model == "Spergel":
            mono_gal = galsim.Spergel(nu=self.params[0].nu, half_light_radius=self.params[0].hlr,
                # flux=self.params[0].gal_flux,
                gsparams=self.gsparams)
            SED = self.get_SED()
            gal = galsim.Chromatic(mono_gal, SED)
            gal_shape = galsim.Shear(g=self.params[0].e, beta=self.params[0].beta*galsim.radians)
            gal = gal.shear(gal_shape)
            gal = gal.shift(dx, dy)

        elif self.galaxy_model == "Sersic":
            mono_gal = galsim.Sersic(n=self.params[0].n, half_light_radius=self.params[0].hlr,
                # flux=self.params[0].gal_flux,
                gsparams=self.gsparams)
            SED = self.get_SED()
            gal = galsim.Chromatic(mono_gal, SED)
            gal_shape = galsim.Shear(g=self.params[0].e, beta=self.params[0].beta*galsim.radians)
            gal = gal.shear(gal_shape)
            gal = gal.shift(dx, dy)

        elif self.galaxy_model == "BulgeDisk":
            mono_bulge = galsim.Spergel(nu=self.params[0].nu_bulge,
                half_light_radius=self.params[0].hlr_bulge,
                gsparams=self.gsparams)
            SED_bulge = self.get_SED(gal_comp='bulge')
            bulge = galsim.Chromatic(mono_bulge, SED_bulge)
            bulge_shape = galsim.Shear(g=self.params[0].e_bulge,
                beta=self.params[0].beta_bulge*galsim.radians)
            bulge = bulge.shear(bulge_shape)

            mono_disk = galsim.Spergel(nu=self.params[0].nu_disk,
                half_light_radius=self.params[0].hlr_disk,
                gsparams=self.gsparams)
            SED_disk = self.get_SED(gal_comp='disk')
            disk = galsim.Chromatic(mono_disk, SED_disk)
            disk_shape = galsim.Shear(g=self.params[0].e_disk,
                beta=self.params[0].beta_disk*galsim.radians)
            disk = disk.shear(disk_shape)

            # gal = self.params[0].bulge_frac * bulge + (1 - self.params[0].bulge_frac) * disk
            gal = bulge + disk
            gal = gal.shift(dx, dy)

        else:
            raise AttributeError("Unimplemented galaxy model")
        final = galsim.Convolve([gal, self.get_psf()])
        # wcs = galsim.PixelScale(self.pixel_scale)'
        try:
            image = final.drawImage(self.filters[filter_name],
                image=out_image, scale=self.pixel_scale, gain=gain)
            if add_noise:
                if self.noise is not None:
                    image.addNoise(self.noise)
                else:
                    raise AttributeError("A GalSim noise model must be specified to add noise to an\
                        image.")
        except RuntimeError:
            print "Trying to make an image that's too big."
            image = None
        return image

    def get_segment(self):
        pass

    def save_image(self, file_name, out_image=None, filter_name='r'):
        image = self.get_image(filter_name=filter_name, out_image=out_image)
        image.write(file_name)
        return None

    def save_psf(self, file_name, ngrid=None):
        psf = self.get_psf()
        if ngrid is None:
            ngrid = 16
        image_epsf = psf.drawImage(scale=self.pixel_scale, nx=ngrid, ny=ngrid)
        image_epsf.write(file_name)
        return None

    def plot_image(self, file_name, ngrid=None, filter_name='r', title=None):
        import matplotlib.pyplot as plt
        if ngrid is not None:
            out_image = galsim.Image(ngrid, ngrid)
        else:
            out_image = None
        ###
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(self.get_image(out_image, add_noise=True, filter_name=filter_name).array / 1.e3,
            cmap=plt.get_cmap('BrBG'), origin='lower',
            interpolation='none',
            extent=[0, ngrid*self.pixel_scale, 0, ngrid*self.pixel_scale])
        ax.set_xlabel(r"Detector $x$-axis (arcsec.)")
        ax.set_ylabel(r"Detector $y$-axis (arcsec.)")
        if title is not None:
            ax.set_title(title)
        cbar = fig.colorbar(im)
        cbar.set_label(r"$10^3$ photons / pixel")
        fig.savefig(file_name)
        return None

    def plot_psf(self, file_name, ngrid=None, title=None, filter_name='r'):
        import matplotlib.pyplot as plt
        psf = self.get_psf()
        if ngrid is None:
            ngrid = 16
        image_epsf = psf.drawImage(image=None,
            scale=self.pixel_scale, nx=ngrid, ny=ngrid)
        ###
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(1,1,1)
        ngx, ngy = image_epsf.array.shape
        im = ax.imshow(image_epsf.array,
            cmap=plt.get_cmap('BrBG'), origin='lower',
            interpolation='none',
            extent=[0, ngx*self.pixel_scale, 0, ngy*self.pixel_scale])
        ax.set_xlabel(r"Detector $x$-axis (arcsec.)")
        ax.set_ylabel(r"Detector $y$-axis (arcsec.)")
        if title is not None:
            ax.set_title(title)
        cbar = fig.colorbar(im)
        cbar.set_label(r"normalized photons / pixel")
        fig.savefig(file_name)
        return None

    def get_moments(self, add_noise=True):
        results = self.get_image(add_noise=add_noise).FindAdaptiveMom()
        print 'HSM reports that the image has observed shape and size:'
        print '    e1 = %.3f, e2 = %.3f, sigma = %.3f (pixels)' % (results.observed_shape.e1,
                    results.observed_shape.e2, results.moments_sigma)


def make_test_images(filter_name_ground='r', filter_name_space='y', file_lab='', galaxy_model="Spergel"):
    """
    Use the GalSimGalaxyModel class to make test images of a galaxy for LSST and WFIRST.
    """
    import os
    import h5py

    ngrid_lsst = 70

    print("Making test images for LSST and WFIRST")

    # LSST
    lsst = GalSimGalaxyModel(pixel_scale=0.2,
        noise=lsst_noise(82357),
        galaxy_model=galaxy_model,
        wavelength=k_filter_central_wavelengths[filter_name_ground] * 1.e-9,
        primary_diam_meters=8.4,
        atmosphere=True)

    # Save the image
    lsst.save_image("../TestData/test_lsst_image" + file_lab + ".fits",
        filter_name=filter_name_ground,
        out_image=galsim.Image(ngrid_lsst, ngrid_lsst))
    lsst.plot_image("../TestData/test_lsst_image" + file_lab + ".png",
        ngrid=ngrid_lsst,
        filter_name=filter_name_ground, title="LSST")
    # Save the corresponding PSF
    lsst.save_psf("../TestData/test_lsst_psf" + file_lab + ".fits",
        ngrid=ngrid_lsst/4)
    lsst.plot_psf("../TestData/test_lsst_psf" + file_lab + ".png",
        ngrid=ngrid_lsst/4, title="LSST")

    # WFIRST
    wfirst = GalSimGalaxyModel(pixel_scale=0.11,
        noise=lsst_noise(82357), #wfirst_noise(82357),
        galaxy_model=galaxy_model,
        wavelength=k_filter_central_wavelengths[filter_name_space] * 1.e-9,
        primary_diam_meters=2.4,
        atmosphere=False)

    ngrid_wfirst = np.ceil(ngrid_lsst * lsst.pixel_scale / wfirst.pixel_scale) #128

    # Save the image
    wfirst.save_image("../TestData/test_wfirst_image" + file_lab + ".fits",
        filter_name=filter_name_space, out_image=galsim.Image(ngrid_wfirst, ngrid_wfirst))
    wfirst.plot_image("../TestData/test_wfirst_image" + file_lab + ".png", ngrid=ngrid_wfirst,
        filter_name=filter_name_space, title="WFIRST")
    # Save the corresponding PSF
    wfirst.save_psf("../TestData/test_wfirst_psf" + file_lab + ".fits",
        ngrid=ngrid_wfirst/4)
    wfirst.plot_psf("../TestData/test_wfirst_psf" + file_lab + ".png",
        ngrid=ngrid_wfirst/4, title="WFIRST")

    lsst_data = lsst.get_image(galsim.Image(ngrid_lsst, ngrid_lsst), add_noise=True,
        filter_name=filter_name_ground).array
    wfirst_data = wfirst.get_image(galsim.Image(ngrid_wfirst, ngrid_wfirst), add_noise=True,
        filter_name=filter_name_space).array

    # -------------------------------------------------------------------------
    ### Save a file with joint image data for input to the Roaster
    f = h5py.File(os.path.join(os.path.dirname(__file__),
        '../TestData/test_image_data' + file_lab + '.h5'), 'w')

    # Define the (sub)groups
    g = f.create_group('ground')
    g_obs = f.create_group('ground/observation')
    g_obs_sex = f.create_group('ground/observation/sextractor')
    g_obs_sex_seg = f.create_group('ground/observation/sextractor/segments')

    s = f.create_group('space')
    s_obs = f.create_group('space/observation')
    s_obs_sex = f.create_group('space/observation/sextractor')
    s_obs_sex_seg = f.create_group('space/observation/sextractor/segments')


    f.attrs['num_sources'] = 1 ### Assert a fixed number of sources for all epochs

    ### Instrument/epoch 1
    g_obs_sex_seg_i = f.create_group("ground/observation/sextractor/segments/0")
    g_obs_sex_seg_i.create_dataset('image', data=lsst_data)
    ### TODO: Add object property data like that that might come out of DMstack or sextractor
    # currently a hack to allow roaster to determine number of objects the
    # same as is done for the data processed by sheller
    g_obs_sex_seg_i.create_dataset('stamp_objprops', data=np.arange(1))
    ### TODO: Add segmentation mask
    # the real data will create a dataset that is an image of the noise
    # for the galsim_galaxy only a single value characterizing the
    # variance is generated
    g_obs_sex_seg_i_noise = g_obs_sex_seg_i.create_dataset('noise', data=lsst.noise.getVariance())
    # for consistency with real data, also assign this to the noise dataset
    # attribute
    g_obs_sex_seg_i_noise.attrs['variance'] = lsst.noise.getVariance()
    ### TODO: add WCS information
    ### TODO: add background model(s)
    g.attrs['telescope'] = 'LSST'
    g.attrs['pixel_scale'] = lsst.pixel_scale
    g_obs.attrs['filter_central'] = k_filter_central_wavelengths[filter_name_ground] * 1.e-9
    g_obs.attrs['filter_name'] = filter_name_ground
    g.attrs['primary_diam'] = lsst.primary_diam_meters
    g.attrs['atmosphere'] = lsst.atmosphere


    ### Instrument/epoch 2
    s_obs_sex_seg_i = f.create_group("space/observation/sextractor/segments/0")
    s_obs_sex_seg_i.create_dataset('image', data=wfirst_data)
    ### TODO: Add object property data like that that might come out of DMstack or sextractor
    # currently a hack to allow roaster to determine number of objects the
    # same as is done for the data processed by sheller
    s_obs_sex_seg_i.create_dataset('stamp_objprops', data=np.arange(1))
    ### TODO: Add segmentation mask
    # the real data will create a dataset that is an image of the noise
    # for the galsim_galaxy only a single value characterizing the
    # variance is generated
    s_obs_sex_seg_i_noise = s_obs_sex_seg_i.create_dataset('noise', data=wfirst.noise.getVariance())
    # for consistency with real data, also assign this to the noise dataset
    # attribute
    s_obs_sex_seg_i_noise.attrs['variance'] = lsst.noise.getVariance()
    ### TODO: add WCS information
    ### TODO: add background model(s)
    s.attrs['telescope'] = 'WFIRST'
    s.attrs['pixel_scale'] = wfirst.pixel_scale
    s_obs.attrs['filter_central'] = k_filter_central_wavelengths[filter_name_space] * 1.e-9
    s_obs.attrs['filter_name'] = filter_name_space
    s.attrs['primary_diam'] = wfirst.primary_diam_meters
    s.attrs['atmosphere'] = wfirst.atmosphere

    f.close()
    # -------------------------------------------------------------------------

def make_blended_test_image(num_sources=3, random_seed=75256611):
    lsst_pixel_scale_arcsec = 0.2

    ellipticities = [0.05, 0.3, 0.16]
    hlrs = [1.8, 1.0, 2.0]
    orientations = np.array([0.1, 0.25, -0.3]) * np.pi

    ### Setup the 'segment' image that will contain all the galaxies in the blend
    npix_segment = 128
    # segment_pos = galsim.CelectialCoord(ra=90.*galsim.degrees, dec=-10.*galsim.degrees)
    segment_image = galsim.ImageF(npix_segment, npix_segment, scale=lsst_pixel_scale_arcsec)

    ### Define the galaxy positions in the segment (relative to the center of the segment image)
    ### (see galsim demo13.py)
    pos_rng = galsim.UniformDeviate(random_seed)
    x_gal = (0.4 + 0.2 * np.array([pos_rng() for i in xrange(num_sources)])) * npix_segment
    y_gal = (0.4 + 0.2 * np.array([pos_rng() for i in xrange(num_sources)])) * npix_segment

    npix_gal = 100

    for isrcs in xrange(num_sources):
        # ### Draw every source using the full output array
        # b = galsim.BoundsI(1, nx, 1, ny)
        # sub_image = segment_image[b]

        sub_image = galsim.Image(npix_gal, npix_gal, scale=lsst_pixel_scale_arcsec)

        src_model = GalSimGalaxyModel(pixel_scale=lsst_pixel_scale_arcsec,
            noise=lsst_noise(82357),
            galaxy_model="Spergel",
            wavelength=770.e-9, primary_diam_meters=8.4, atmosphere=True)
        src_model.params[0]["e"] = ellipticities[isrcs]
        src_model.params[0]["beta"] = orientations[isrcs]
        src_model.params[0]["hlr"] = hlrs[isrcs]
        # p = src_model.get_params()

        # src_model.set_params(p)

        gal_image = src_model.get_image(sub_image, filter_name='r')

        ix = int(math.floor(x_gal[isrcs]+0.5))
        iy = int(math.floor(y_gal[isrcs]+0.5))

        # Create a nominal bound for the postage stamp given the integer part of the
        # position.
        sub_bounds = galsim.BoundsI(ix-0.5*npix_gal, ix+0.5*npix_gal-1,
                                    iy-0.5*npix_gal, iy+0.5*npix_gal-1)
        sub_image.setOrigin(galsim.PositionI(sub_bounds.xmin, sub_bounds.ymin))

        # Find the overlapping bounds between the large image and the individual postage
        # stamp.
        bounds = sub_image.bounds & segment_image.bounds

        segment_image[bounds] += sub_image[bounds]

    segment_image.write("../TestData/test_lsst_blended_image.fits")


if __name__ == "__main__":
    make_test_images()
    # make_blended_test_image()
