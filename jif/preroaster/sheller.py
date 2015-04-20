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
import h5py

## Input
# for now I am just entering the input directly here for debugging purposes
# later I will make a class structure

# Specify the "gound" fits files
# file name for the sextractor output -BACKGROUND fits file
img0_minback_name = 'HST_conv_minusbackground.fits'
# science image
img0_image_name = '../../TestData/cl0916_f814w_drz_sci_convolved.fits'
# noise fits file
img0_rms_name = 'HST_conv_background_rms.fits'
# segmentation fits file, note currently we make our own in this program,
# since the sextractor segmentation image already deblends
img0_segmentation_name = 'HST_conv_segmentation.fits'
# the fits catalog
cat0_name = 'HST_conv_catalog.fits'

# Specify the space fits files
# file name for the sextractor output -BACKGROUND fits file
img1_minback_name = 'HST_minusbackground.fits'
img1_image_name = '../../TestData/cl0916_f814w_drz_sci.fits'
img1_rms_name = 'HST_background_rms.fits'
img1_segmentation_name = 'HST_segmentation.fits'
cat1_name = 'HST_catalog.fits'

# Threshild factor, multiplacitive factor above rms that a pixel must be to 
# be considered during the percolation analysis.
f_thresh = 3

# percolation structure, used to determine if pixels should be linked
struc = [[0,1,0],
         [1,1,1],
         [0,1,0]]

# since there is a lot of junk at the edges of the image lets cut this out
# by defining min and max bounds for a relatively clean area of the image
x_bounds = (550,8050)
y_bounds = (650,4150)

## Program
# load the "ground" fits images
fits_img0_image = pyfits.open(img0_image_name)
fits_img0_minback = pyfits.open(img0_minback_name)
fits_img0_rms = pyfits.open(img0_rms_name)
fits_img0_segmentation = pyfits.open(img0_segmentation_name)
fits_cat0 = pyfits.open(cat0_name)

# assign the fits image data to arrays
# assign the image data to an array
data_img0_image = fits_img0_image[0].data
data_img0_minback = fits_img0_minback[0].data
data_img0_rms = fits_img0_rms[0].data
data_img0_segmentation = fits_img0_segmentation[0].data
data_cat0 = fits_cat0[2].data

# load the Space fits images
fits_img1_image = pyfits.open(img1_image_name)
fits_img1_minback = pyfits.open(img1_minback_name)
fits_img1_rms = pyfits.open(img1_rms_name)
fits_img1_segmentation = pyfits.open(img1_segmentation_name)
fits_cat1 = pyfits.open(cat1_name)

# assign the fits image data to arrays
# assign the image data to an array
data_img1_image = fits_img1_image[0].data
data_img1_minback = fits_img1_minback[0].data
data_img1_rms = fits_img1_rms[0].data
data_img1_segmentation = fits_img1_segmentation[0].data
data_cat1 = fits_cat1[2].data

# since there is a lot of junk at the edge of the field let's cut that out
# define pixel coordinate arrays to determine segment pixel (x,y) locations
mask_clean = (slice(y_bounds[0],y_bounds[1]),slice(x_bounds[0],x_bounds[1]))
data_img0_image = data_img0_image[mask_clean]
data_img0_minback = data_img0_minback[mask_clean]
data_img0_rms = data_img0_rms[mask_clean]
data_img0_segmentation = data_img0_segmentation[mask_clean]
data_img1_image = data_img1_image[mask_clean]
data_img1_minback = data_img1_minback[mask_clean]
data_img1_rms = data_img1_rms[mask_clean]
data_img1_segmentation = data_img1_segmentation[mask_clean]

# Setup the hdf5 data structure
f = h5py.File('spaceground.hdf5',mode='w')
# Define the (sub)groups
g = f.create_group('ground')
g_obs = f.create_group('ground/observation')
g_obs_sex = f.create_group('ground/observation/sextractor')
g_obs_sex_grp = f.create_group('ground/observation/sextractor/segments')

s = f.create_group('space')
s_obs = f.create_group('space/observation')
s_obs_sex = f.create_group('space/observation/sextractor')
s_obs_sex_grp = f.create_group('space/observation/sextractor/segments')


# Save some basic metadata, later this will be copied from the fits headers
# instrument specific
s.attrs['telescope'] = 'HST'
s.attrs['primary_diam'] = 2.4
s.attrs['instrument'] = 'ACS'
s.attrs['detector'] = 'WFC'
s.attrs['pixel_scale'] = 0.05
s.attrs['atmosphere'] = False
# observation specific
s_obs.attrs['filter'] = 'F814W'
# note that we should just have a group for filter curves with datasets
# containing througput as a function of wavelength
s_obs.attrs['filter_central'] = 805.98e-9 #meters
s_obs.attrs['filter_mean'] = 808.74e-9 #meters
s_obs.attrs['filter_peak'] = 746.02e-9 #meters
s_obs.attrs['filter_fwhm'] = 154.16e-9 #meters
s_obs.attrs['filter_range'] = 287e-9 #meters
s_obs.attrs['crpix'] = numpy.array([[6142.33544921875],
                                    [1764.46228027344]])
s_obs.attrs['crval'] = numpy.array([[139.076115042054],
                                    [29.8330615113333]])
s_obs.attrs['cd'] = numpy.array([[9.489339000000001E-06,-1.0141681E-05],
                                 [-1.0141681E-05,-9.489339000000001E-06]])
# copy some metadata from the fits header
s_obs.attrs['TELESCOP'] = 'HST' # telescope name
s_obs.attrs['INSTRUME'] = 'ACS' # instrument name
s_obs.attrs['DETECTOR'] = 'WFC' # detector name
s_obs.attrs['FILTER1'] = 'CLEAR1L' # a clear filter
s_obs.attrs['FILTER2'] = 'F814W' # non-clear filter
s_obs.attrs['CRPIX1']  = 6142.33544921875 # x-coordinate of reference pixel                
s_obs.attrs['CRPIX2']  = 1764.46228027344 # y-coordinate of reference pixel                
s_obs.attrs['CRVAL1']  = 139.076115042054 # first axis value at reference pixel            
s_obs.attrs['CRVAL2']  = 29.8330615113333 # second axis value at reference pixel           
s_obs.attrs['CTYPE1']  = 'RA---TAN' # the coordinate type for the first axis         
s_obs.attrs['CTYPE2']  = 'DEC--TAN' # the coordinate type for the second axis        
s_obs.attrs['CD1_1']  = 9.489339000000001E-06 # partial of first axis coordinate w.r.t. x     
s_obs.attrs['CD1_2']  = -1.0141681E-05 # partial of first axis coordinate w.r.t. y      
s_obs.attrs['CD2_1']  = -1.0141681E-05 # partial of second axis coordinate w.r.t. x     
s_obs.attrs['CD2_2']  = -9.489339000000001E-06 # partial of second axis coordinate w.r.t. y   
s_obs.attrs['LTV1']  = 0.0 # offset in X to subsection start                
s_obs.attrs['LTV2']  = 0.0 # offset in Y to subsection start                
s_obs.attrs['LTM1_1']  = 1.0 # reciprocal of sampling rate in X               
s_obs.attrs['LTM2_2']  = 1.0 # reciprocal of sampling rate in Y

# ground based attributes
# instrument specific
g.attrs['telescope'] = 'fauxSubaru'
g.attrs['primary_diam'] = 2.4
g.attrs['instrument'] = 'ACS'
g.attrs['detector'] = 'WFC'
g.attrs['pixel_scale'] = 0.05
g.attrs['atmosphere'] = True
# observation specific
g_obs.attrs['filter_central'] = 805.98e-9 #meters
g_obs.attrs['filter_mean'] = 808.74e-9 #meters
g_obs.attrs['filter_peak'] = 746.02e-9 #meters
g_obs.attrs['filter_fwhm'] = 154.16e-9 #meters
g_obs.attrs['filter_range'] = 287e-9 #meters
g_obs.attrs['crpix'] = numpy.array([[6142.33544921875],
                                    [1764.46228027344]])
g_obs.attrs['crval'] = numpy.array([[139.076115042054],
                                    [29.8330615113333]])
g_obs.attrs['cd'] = numpy.array([[9.489339000000001E-06,-1.0141681E-05],
                                 [-1.0141681E-05,-9.489339000000001E-06]])
# copy some metadata from the fits header
g_obs.attrs['TELESCOP'] = 'HST' # telescope name
g_obs.attrs['INSTRUME'] = 'ACS' # instrument name
g_obs.attrs['DETECTOR'] = 'WFC' # detector name
g_obs.attrs['FILTER1'] = 'CLEAR1L' # a clear filter
g_obs.attrs['FILTER2'] = 'F814W' # non-clear filter
g_obs.attrs['CRPIX1']  = 6142.33544921875 # x-coordinate of reference pixel                
g_obs.attrs['CRPIX2']  = 1764.46228027344 # y-coordinate of reference pixel                
g_obs.attrs['CRVAL1']  = 139.076115042054 # first axis value at reference pixel            
g_obs.attrs['CRVAL2']  = 29.8330615113333 # second axis value at reference pixel           
g_obs.attrs['CTYPE1']  = 'RA---TAN' # the coordinate type for the first axis         
g_obs.attrs['CTYPE2']  = 'DEC--TAN' # the coordinate type for the second axis        
g_obs.attrs['CD1_1']  = 9.489339000000001E-06 # partial of first axis coordinate w.r.t. x     
g_obs.attrs['CD1_2']  = -1.0141681E-05 # partial of first axis coordinate w.r.t. y      
g_obs.attrs['CD2_1']  = -1.0141681E-05 # partial of second axis coordinate w.r.t. x     
g_obs.attrs['CD2_2']  = -9.489339000000001E-06 # partial of second axis coordinate w.r.t. y   
g_obs.attrs['LTV1']  = 0.0 # offset in X to subsection start                
g_obs.attrs['LTV2']  = 0.0 # offset in Y to subsection start                
g_obs.attrs['LTM1_1']  = 1.0 # reciprocal of sampling rate in X               
g_obs.attrs['LTM2_2']  = 1.0 # reciprocal of sampling rate in Y



# To do: write full field images to the data structure.


## Operate on the "ground" image to determine blended groups

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
fits_img0_segmentation[0].data = img0_seg
fits_img0_segmentation.writeto('HST_conv_seg.fits',clobber=True)


# 

## Create datasets
#dset = f.create_dataset('space/observation/sex/', data=arr)

# create a compund array (table) with headers
# reference: ftp://www.hdfgroup.org/HDF5/examples/Py/h5_compound.py

# define the column names and data types

# create an array with the row values

# create the hdf5 dataset with that dtype and the number of rows

# assign the array to the dataset


def fitshead2hdfattrib(fitsfile,hdfdset):
    '''
    Converts a fits header into the attributes of an hdf5 dataset.
    '''
    print 'Need to make this function.'

# For the effort of determining whether objects from the sextractor catalog
# are within a given stamp or segmentation region let us round the pixel
# coordinates to the nearest integer pixel value. Since objects should should
# be well within the segmentation region this rounding shouldn't cause any
# objects to be lost. (we could think about growing the segmentation region)
# the x_bound and y_bound terms are to correct for the fact that the image
# boarders have been cut off but this was not done for sextractor
x_round_0 = numpy.round(data_cat0.field('XWIN_IMAGE')) - x_bounds[0]
y_round_0 = numpy.round(data_cat0.field('YWIN_IMAGE')) - y_bounds[0]
x_round_1 = numpy.round(data_cat1.field('XWIN_IMAGE')) - x_bounds[0]
y_round_1 = numpy.round(data_cat1.field('YWIN_IMAGE')) - y_bounds[0]


# define pixel coordinate arrays to determine segment pixel (x,y) locations
N_x, N_y = numpy.shape(img0_seg) # image pixel dimensions
X, Y = numpy.meshgrid(range(N_y), range(N_x))

# Save the fits catalog dtype tuple so that it can be used in the creation
# of hdf5 compound arrays
dt = data_cat0.dtype

# Determine the rectangular bounds of each of the identified segments
stamps = measurements.find_objects(img0_seg)
# Loop through the segements saving associated postage stamps
print 'Making stamps for segment:'
for i in range(N_img0_seg):
    print '{0} of {1}'.format(i+1,N_img0_seg)
    # get the pixel coordinates bounding the local group
    stamp_i = stamps[i]
    
    # Create groups for this segment
    group_name_s = 'space/observation/sextractor/segments/'+str(i)
    group_name_g = 'ground/observation/sextractor/segments/'+str(i)
    
    s_obs_sex_grp_i = f.create_group(group_name_s)    
    g_obs_sex_grp_i = f.create_group(group_name_g)
    
    # Ground Datasets
    # Create the image-background dataset
    g_obs_sex_grp_i.create_dataset('image', data=data_img0_minback[stamp_i])
    # Create the rms noise dataset
    g_obs_sex_grp_i_noise = g_obs_sex_grp_i.create_dataset('noise', data=data_img0_rms[stamp_i])
    # estimate the varinace of this noise image and save as attribute
    g_obs_sex_grp_i_noise.attrs['variance'] = numpy.median(g_obs_sex_grp_i_noise)**2
    # Create the local group segment mask dataset
    g_obs_sex_grp_i.create_dataset('segmask', data=mask_img0[stamp_i])
    
    # Space Datasets
    # Create the image-background dataset
    s_obs_sex_grp_i.create_dataset('image', data=data_img1_minback[stamp_i])
    # Create the rms noise dataset
    s_obs_sex_grp_i_noise = s_obs_sex_grp_i.create_dataset('noise', data=data_img1_rms[stamp_i])
    # estimate the varinace of this noise image and save as attribute
    s_obs_sex_grp_i_noise.attrs['variance'] = numpy.median(s_obs_sex_grp_i_noise)**2
    # Create the local group segment mask dataset, 
    # for now just the same as the ground 
    s_obs_sex_grp_i.create_dataset('segmask', data=mask_img0[stamp_i])
    
    # Determine which objects from the sextractor analysis are within the 
    # postage stamp
    # first slice is x-coord, and 0th slice is y-coord
    stamp_i_xbounds = stamp_i[1]
    stamp_i_ybounds = stamp_i[0]
    mask_stamp_0 = numpy.logical_and(numpy.logical_and(x_round_0>=stamp_i_xbounds.start,
                                                       x_round_0<=stamp_i_xbounds.stop),
                                     numpy.logical_and(y_round_0>=stamp_i_ybounds.start,
                                                       y_round_0<=stamp_i_ybounds.stop))
    N_stamp_0 = numpy.sum(mask_stamp_0)
    # in the future will need to change this to enable different pixel scales,
    # position angles, etc., but not now since using same image just convolved
    mask_stamp_1 = numpy.logical_and(numpy.logical_and(x_round_1>=stamp_i_xbounds.start,
                                                       x_round_1<=stamp_i_xbounds.stop),
                                     numpy.logical_and(y_round_1>=stamp_i_ybounds.start,
                                                       y_round_1<=stamp_i_ybounds.stop))
    N_stamp_1 = numpy.sum(mask_stamp_1)
    # Save the object properties for those objects within the stamp bounds
    # note that this might not work, since the fits table may need to be
    # processed a bit more to get it in the correc format
    data_stamp0 = data_cat0[mask_stamp_0]
    data_stamp1 = data_cat1[mask_stamp_1]
    g_obs_sex_grp_i.create_dataset('stamp_objprops', data = data_stamp0)
    s_obs_sex_grp_i.create_dataset('stamp_objprops', data = data_stamp1)
    
    # Determine which objects from the sextractor analysis are within the 
    # segmentation region
    mask_pixels_seg = img0_seg == i+1 # identify pixels belonging to i^th seg
    mask_seg_0 = numpy.logical_and(numpy.in1d(x_round_0[mask_stamp_0],
                                              X[mask_pixels_seg]),
                                   numpy.in1d(y_round_0[mask_stamp_0],
                                              Y[mask_pixels_seg]))
    N_seg_0 = numpy.sum(mask_seg_0)
    
    mask_seg_1 = numpy.logical_and(numpy.in1d(x_round_1[mask_stamp_1],
                                              X[mask_pixels_seg]),
                                   numpy.in1d(y_round_1[mask_stamp_1],
                                              Y[mask_pixels_seg]))
    N_seg_1 = numpy.sum(mask_seg_1)

    # create a compound dataset with the column labels and dtypes
    # note that some of this may not be necessary since the data_cat0 is a
    # recarray, thus we might be able to just plug it in the create_dataset
    ds_temp = g_obs_sex_grp_i.create_dataset('seg_objprops', (N_seg_0,), dtype = dt)
    ds_temp[...] = data_stamp0[mask_seg_0]

    ds_temp = s_obs_sex_grp_i.create_dataset('seg_objprops', (N_seg_1,), dtype = dt)
    ds_temp[...] = data_stamp1[mask_seg_1]

f.close()

plt.show()

print 'end'