'''
This is a code snipit meant to read in a fits image convolve it with a
Gaussian PSF and save the resulting image. Some of this code is based off the
ipython notebook PercolationTheory Exploratory Note.ipynb.
'''
import pyfits
from scipy import ndimage
from scipy.ndimage import measurements, filters

### Input
# HST fits image (can be downloaded from http://mercury.physics.ucdavis.edu/JIF/TestData/)
image_file_space = '../../TestData/cl0916_f814w_drz_sci.fits'

# Ground based seeing
psf_ground_arcsec = 0.7 #arcsec

# Pixel scales of the detectors
pixscale_ground = 0.2 #arcsec/pix
pixscale_space = 0.05 #arcsec/pix

### Program
# determine the size of the ground PSF convolution kernel in space pixel
# coordinates
psf_ground_spix = psf_ground_arcsec / pixscale_space # space pixels
print 'psf_ground corresponds to {0:0.1f} space pixels'.format(psf_ground_spix)

# load the HST fits image
fits_hst = pyfits.open(image_file_space)

# convolve the space image with the ground seeing
# currently assumes just a Gaussian convolution
fits_hst[0].data = ndimage.filters.gaussian_filter(fits_hst[0].data,psf_ground_spix,mode='constant')

# write the convolved image to a new fits file
fits_hst.writeto('../../TestData/cl0916_f814w_drz_sci_convolved.fits',
                 clobber=True)

print 'convolveHST: finished'
print 'wrote convolved fits image to ../../TestData/cl0916_f814w_drz_sci_convolved.fits'

fits_hst.writeto