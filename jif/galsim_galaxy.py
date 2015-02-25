#!/usr/bin/env python
# encoding: utf-8
"""
galsim_galaxy.py

Wrapper for GalSim galaxy models to use in MCMC.
"""

import numpy as np
import galsim

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


class GalSimGalParams(object):
    """Parameters for GalSim galaxies"""
    def __init__(self, galaxy_model="Gaussian"):
        self.galaxy_model = galaxy_model
        if galaxy_model == "Gaussian":
            self.gal_flux = 1.e5
            self.gal_sigma = 2.
            self.e = 0.3
            self.beta = np.pi/4.
            self.n_params = 4
        elif galaxy_model == "Spergel":
            raise NotImplementedError()
        elif galaxy_model == "Sersic":
            self.gal_flux = 1.e5
            self.n = 3.4
            self.hlr = 1.8
            self.e = 0.3
            self.beta = np.pi/4.
            self.n_params = 5
        elif galaxy_model == "BulgeDisk":
            self.gal_flux = 1.e5
            self.bulge_n = 3.4
            self.disk_n = 1.5
            self.bulge_re = 2.3
            self.disk_r0 = 0.85
            self.bulge_frac = 0.0 #0.3
            self.e_bulge = 0.01
            self.e_disk = 0.25
            self.beta_bulge = np.pi/4.
            self.beta_disk = 3. * np.pi/4.
            self.n_params = 10
        else:
            raise AttributeError("Unimplemented galaxy model")

    def num_params(self):
        return self.n_params

    def get_params(self):
        """
        Return an array of parameter values.
        """
        if self.galaxy_model == "Gaussian":
            return np.array([self.gal_flux, self.gal_sigma, self.e, self.beta])
        elif self.galaxy_model == "Spergel":
            raise NotImplementedError()
        elif self.galaxy_model == "Sersic":
            return np.array([self.gal_flux, self.n, self.hlr, self.e, self.beta])
        elif self.galaxy_model == "BulgeDisk":
            return np.array([self.gal_flux, self.bulge_n, self.disk_n, self.bulge_re, self.disk_r0, 
            self.bulge_frac, self.e_bulge, self.e_disk, self.beta_bulge, self.beta_disk])
        else:
            raise AttributeError("Unimplemented galaxy model")


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

        # self.params = GalSimGalParams(galaxy_model=galaxy_model)
        self.params = np.core.records.array([(1.e5, 3.4, 1.8, 0.3, np.pi/4)],
            dtype=[('gal_flux', '<f8'), ('n', '<f8'), ('hlr', '<f8'), ('e', '<f8'),
                   ('beta', '<f8')])
        self.n_params = 5

    def set_params(self, p):
        """
        Take a list of parameters and set local variables

        For use in emcee.
        """
        return NotImplementedError()

    def get_params(self):
        """
        Return a list of model parameter values.
        """
        return self.params.view('<f8')

    def get_psf(self):
        lam_over_diam = self.wavelength / self.primary_diam_meters
        lam_over_diam *= 206265. # arcsec
        optics = galsim.Airy(lam_over_diam, obscuration=0.548, flux=1.)
        if self.atmosphere:
            atmos = galsim.Kolmogorov(lam_over_r0=9.e-8)
            psf = galsim.Convolve([atmos, optics])
        else:
            psf = optics
        return psf        

    def get_image(self, out_image=None, add_noise=False):
        if self.galaxy_model == "Gaussian":
            gal = galsim.Gaussian(flux=self.params.gal_flux, sigma=self.params.gal_sigma)
            gal_shape = galsim.Shear(g=self.params.e, beta=self.params.beta*galsim.radians)
            gal = gal.shear(gal_shape)
        elif self.galaxy_model == "Spergel":
            raise NotImplementedError()
        elif self.galaxy_model == "Sersic":
            gal = galsim.Sersic(n=self.params[0].n, half_light_radius=self.params[0].hlr,
                flux=self.params[0].gal_flux)
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
        # wcs = galsim.PixelScale(self.pixel_scale)
        image = final.drawImage(image=out_image, scale=self.pixel_scale)
        if add_noise:
            if self.noise is not None:
                image.addNoise(self.noise)
            else:
                raise AttributeError("A GalSim noise model must be specified to add noise to an image.")
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
        im = ax.matshow(self.get_image(out_image, add_noise=True).array, cmap=plt.get_cmap('coolwarm')) #, vmin=-350, vmax=350)
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
    lsst = GalSimGalaxyModel(pixel_scale=0.2, noise=lsst_noise(82357), galaxy_model="Sersic",
        wavelength=500.e-9, primary_diam_meters=8.4, atmosphere=True)
    lsst.save_image("test_lsst_image.fits")
    lsst.plot_image("test_lsst_image.png", ngrid=64)

    wfirst = GalSimGalaxyModel(pixel_scale=0.11, noise=wfirst_noise(82357), galaxy_model="Sersic",
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

