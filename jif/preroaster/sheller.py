"""
The sheller's
Following general steps
1. load the HST convolved minus background image HST_conv_minusbackground.fits
2. load the HST convolved rms image HST_background_rms.fits
3. Create a mask, if image-background > f * rms then pixel = 1, else 0, where
f is some factor scaling the threshold for percolation regions
4. Perform percolation analysis of this mask to create segments
5. Determine the bounding box extents for each percolation region
6. Make corresponding postage stamp cutouts
"""
import numpy
import matplotlib.pyplot as plt
import pyfits
from scipy import ndimage
from scipy.ndimage import measurements, filters
import skimage
from skimage import filter

## Input
# for now I am just entering the input directly here for debugging purposes
# later I will make a class structure

# Specify the "gound" fits files
# file name for the sextractor output -BACKGROUND fits file
img0_image_name = '../../'
img0_minback_name = 'HST_conv_minusbackground.fits'
img0_rms_name = 'HST_conv_background_rms.fits'
img0_segmentation_name = 


# Threshild factor, multiplacitive factor above rms that a pixel must be to 
# be considered during the percolation analysis.
f_thresh = 3

# percolation structure, used to determine if pixels should be linked
struc = [[0,1,0],
         [1,1,1],
         [0,1,0]]

## Program
# load the HST fits images
fits_img0_minback = pyfits.open(img0_minback_name)
fits_img0_rms = pyfits.open(img0_rms_name)

# assign the fits image data to arrays
# assign the image data to an array
data_img0_minback = fits_img0_minback[0].data
data_img0_rms = fits_img0_rms[0].data

# make the mask for the percolation analysis
# mask_img0 = data_img0_minback >= f_thresh * data_img0_rms
# I changed to the following line to try and exclude some detected regions
# off the observed part of the field
mask_img0 = numpy.logical_and(data_img0_minback >= f_thresh * data_img0_rms,
                              data_img0_minback > 3)

# perform the percolation analysis
img0_seg, N_img0_seg = ndimage.measurements.label(mask_img0,structure=struc)
# img0_seg is an (x,y) pixel array with values equal to the numbered segment
# N_img0_seg is the total number of segments identified in the image

# Create a plot of the identified segments
plt.imshow(img0_seg, origin='lower', interpolation='nearest')
plt.colorbar()


# Save this segmentation map as a fits file
# Need to update this so that I am not using a previously loaded fits file
fits_img0_rms[0].data = img0_seg
fits_img0_rms.writeto('HST_conv_seg.fits',clobber=True)

# Loop through the segements saving associated postage stamps



plt.show()

print 'end'