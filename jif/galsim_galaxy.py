#!/usr/bin/env python
# encoding: utf-8
"""
galsim_galaxy.py

Wrapper for GalSim galaxy models to use in MCMC.
"""
import os
import numpy as np
from operator import add
import galsim


k_SED_names = ['CWW_E_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext', 'CWW_Im_ext']

k_galparams_type_sersic = [('redshift', '<f8'), ('n', '<f8'), ('hlr', '<f8'), ('e', '<f8'), 
                           ('beta', '<f8'),
                           ('flux_sed1', '<f8'), ('flux_sed2', '<f8'),
                           ('flux_sed3', '<f8'), ('flux_sed4', '<f8')]

k_galparams_type_spergel = [('redshift', '<f8'), ('nu', '<f8'), ('hlr', '<f8'), ('e', '<f8'), 
                            ('beta', '<f8'),
                            ('flux_sed1', '<f8'), ('flux_sed2', '<f8'),
                            ('flux_sed3', '<f8'), ('flux_sed4', '<f8')]


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
                 pixel_scale=0.2, 
                 noise=None,
                 galaxy_model="Gaussian",
                 wavelength=1.e-6,
                 primary_diam_meters=2.4,
                 atmosphere=False): 
        self.psf_sigma = psf_sigma
        self.pixel_scale = pixel_scale
        # if noise is None:
        #     noise = galsim.GaussianNoise(sigma=30.)
        self.noise = noise
        self.galaxy_model = galaxy_model
        self.wavelength = wavelength
        self.primary_diam_meters = primary_diam_meters
        self.atmosphere = atmosphere

        ### Set GalSim galaxy model parameters
        # self.params = GalSimGalParams(galaxy_model=galaxy_model)
        if galaxy_model == "Sersic":
            self.params = np.core.records.array([(1.e5, 3.4, 1.8, 0.3, np.pi/4, 1., 0., 0., 0.)],
                dtype=k_galparams_type_sersic)
            self.paramtypes = k_galparams_type_sersic
            self.paramnames = ['redshift', 'n', 'hlr', 'e','beta', 
                'flux_sed1', 'flux_sed2', 'flux_sed3', 'flux_sed4']
            self.n_params = len(self.paramnames)
        elif galaxy_model == "Spergel":
            self.params = np.core.records.array([(1.e5, -0.3, 1.8, 0.3, np.pi/4, 1., 0., 0., 0.)],
                dtype=k_galparams_type_spergel)
            self.paramtypes = k_galparams_type_spergel
            self.paramnames = ['redshift', 'nu', 'hlr', 'e', 'beta', 
                'flux_sed1', 'flux_sed2', 'flux_sed3', 'flux_sed4']
            self.n_params = len(self.paramnames)
        else:
            raise AttributeError("Unimplemented galaxy model")

        ### Set GalSim SED model parameters
        self._load_sed_files()

        self.gsparams = galsim.GSParams(
            folding_threshold=1.e-2, # maximum fractional flux that may be folded around edge of FFT
            maxk_threshold=2.e-2,    # k-values less than this may be excluded off edge of FFT
            xvalue_accuracy=1.e-2,   # approximations in real space aim to be this accurate
            kvalue_accuracy=1.e-2,   # approximations in fourier space aim to be this accurate
            shoot_accuracy=1.e-2,    # approximations in photon shooting aim to be this accurate
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

    def set_params(self, p):
        """
        Take a list of parameters and set local variables

        For use in emcee.
        """
        self.params = np.core.records.array(p, dtype=self.paramtypes)
        return None

    def get_params(self):
        """
        Return a list of model parameter values.
        """
        return self.params.view('<f8')

    def get_psf(self):
        lam_over_diam = self.wavelength / self.primary_diam_meters
        lam_over_diam *= 206265. # arcsec
        optics = galsim.Airy(lam_over_diam, obscuration=0.548, flux=1., gsparams=self.gsparams)
        if self.atmosphere:
            atmos = galsim.Kolmogorov(lam_over_r0=9.e-8, gsparams=self.gsparams)
            psf = galsim.Convolve([atmos, optics])
        else:
            psf = optics
        return psf

    def get_SED(self):
        """
        Get the GalSim SED object given the SED parameters and redshift
        """
        SEDs = [self.SEDs[SED_name].withFluxDensity(target_flux_density=self.params[0].flux_sed1, 
                                    wavelength=500).atRedshift(self.params[0].redshift)
                for SED_name in self.SEDs]
        return reduce(add, SEDs)

    def get_image(self, out_image=None, add_noise=False):
        if self.galaxy_model == "Gaussian":
            gal = galsim.Gaussian(flux=self.params.gal_flux, sigma=self.params.gal_sigma)
            gal_shape = galsim.Shear(g=self.params.e, beta=self.params.beta*galsim.radians)
            gal = gal.shear(gal_shape)

        elif self.galaxy_model == "Spergel":

            mono_gal = galsim.Spergel(nu=self.params[0].nu, half_light_radius=self.params[0].hlr,
                # flux=self.params[0].gal_flux, 
                gsparams=self.gsparams)
            SED = self.get_SED()
            gal = galsim.Chromatic(mono_gal, SED)

            gal_shape = galsim.Shear(g=self.params[0].e, beta=self.params[0].beta*galsim.radians)
            gal = gal.shear(gal_shape)

        elif self.galaxy_model == "Sersic":
            gal = galsim.Sersic(n=self.params[0].n, half_light_radius=self.params[0].hlr,
                flux=self.params[0].gal_flux, gsparams=self.gsparams)
            gal_shape = galsim.Shear(g=self.params[0].e, beta=self.params[0].beta*galsim.radians)
            gal = gal.shear(gal_shape)            

        elif self.galaxy_model == "BulgeDisk":
            bulge = galsim.Sersic(n=self.params.bulge_n, half_light_radius=self.params.bulge_re)
            bulge = bulge.shear(g=self.params.e_bulge, beta=self.params.beta_bulge*galsim.radians)
            disk = galsim.Sersic(n=self.params.disk_n, half_light_radius=self.params.disk_r0)
            disk = disk.shear(g=self.params.e_disk, beta=self.params.beta_disk*galsim.radians)
            gal = self.params.bulge_frac * bulge + (1 - self.params.bulge_frac) * disk
            gal = gal.withFlux(self.params.gal_flux)

        else:
            raise AttributeError("Unimplemented galaxy model")
        final = galsim.Convolve([gal, self.get_psf()])
        # wcs = galsim.PixelScale(self.pixel_scale)'
        try:
            image = final.drawImage(image=out_image, scale=self.pixel_scale)
            if add_noise:
                if self.noise is not None:
                    image.addNoise(self.noise)
                else:
                    raise AttributeError("A GalSim noise model must be specified to add noise to an image.")
        except RuntimeError:
            print "Trying to make an image that's too big."
            image = None                    
        return image

    def save_image(self, file_name):
        image = self.get_image()
        image.write(file_name)
        return None

    def plot_image(self, file_name, ngrid=None):
        import matplotlib.pyplot as plt
        if ngrid is not None:
            out_image = galsim.Image(ngrid, ngrid)
        else:
            out_image = None
        ###
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(1,1,1)
        im = ax.matshow(self.get_image(out_image, add_noise=True).array, 
            cmap=plt.get_cmap('coolwarm')) #, vmin=-350, vmax=350)
        fig.colorbar(im)
        fig.savefig(file_name)
        return None

    def get_moments(self, add_noise=True):
        results = self.get_image(add_noise=add_noise).FindAdaptiveMom()
        print 'HSM reports that the image has observed shape and size:'
        print '    e1 = %.3f, e2 = %.3f, sigma = %.3f (pixels)' % (results.observed_shape.e1,
                    results.observed_shape.e2, results.moments_sigma)


def make_test_images():
    """
    Use the GalSimGalaxyModel class to make test images of a galaxy for LSST and WFIRST.
    """
    import h5py

    print "Making test images for LSST and WFIRST"
    lsst = GalSimGalaxyModel(pixel_scale=0.2, noise=lsst_noise(82357), galaxy_model="Spergel",
        wavelength=500.e-9, primary_diam_meters=8.4, atmosphere=True)
    lsst.save_image("test_lsst_image.fits")
    lsst.plot_image("test_lsst_image.png", ngrid=64)

    wfirst = GalSimGalaxyModel(pixel_scale=0.11, noise=wfirst_noise(82357), galaxy_model="Spergel",
        wavelength=1.e-6, primary_diam_meters=2.4, atmosphere=False)
    wfirst.save_image("test_wfirst_image.fits")
    wfirst.plot_image("test_wfirst_image.png", ngrid=64)

    lsst_data = lsst.get_image(galsim.Image(64, 64), add_noise=True).array
    wfirst_data = wfirst.get_image(galsim.Image(64, 64), add_noise=True).array

    # -------------------------------------------------------------------------
    ### Save a file with joint image data for input to the Roaster
    f = h5py.File('test_image_data.h5', 'w')
    f.attrs['num_sources'] = 1 ### Assert a fixed number of sources for all epochs

    ### Instrument/epoch 1
    cutout1 = f.create_group("cutout1")
    dat1 = cutout1.create_dataset('pixel_data', data=lsst_data)
    ### TODO: Add segmentation mask
    noise1 = cutout1.create_dataset('noise_model', data=lsst.noise.getVariance())
    ### TODO: add WCS information
    ### TODO: add background model(s)
    cutout1.attrs['instrument'] = 'LSST'
    cutout1.attrs['pixel_scale'] = 0.2
    cutout1.attrs['wavelength'] = 500.e-9
    cutout1.attrs['primary_diam'] = 8.4
    cutout1.attrs['atmosphere'] = True

    ### Instrument/epoch 2
    cutout2 = f.create_group("cutout2")
    dat2 = cutout2.create_dataset('pixel_data', data=wfirst_data)
    ### TODO: Add segmentation mask
    noise2 = cutout2.create_dataset('noise_model', data=wfirst.noise.getVariance())
    ### TODO: add WCS information
    ### TODO: add background model(s)
    cutout2.attrs['instrument'] = 'WFIRST'
    cutout2.attrs['pixel_scale'] = 0.11
    cutout2.attrs['wavelength'] = 1.e-6
    cutout2.attrs['primary_diam'] = 2.4
    cutout2.attrs['atmosphere'] = False

    f.close()
    # -------------------------------------------------------------------------    


if __name__ == "__main__":
    make_test_images()

