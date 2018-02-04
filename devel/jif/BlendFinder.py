'''
The purpose of this program is to separate a subaru image catalog into blends
and non-blends based on an overlapping HST catalog.
'''
from __future__ import division
import numpy
import tools
import obsplan
import ds9tools

## User Input

# Musket Ball HST catalog
catalog_hst = '/Users/dawson/OneDrive/Research/ShapeComparison/PureCatalogs/HSTPureCat_revB.txt'
# Musket Ball Subaru catalog
catalog_sub = '/Users/dawson/OneDrive/Research/ShapeComparison/PureCatalogs/SubaruPureCat_revB.txt'

# HST footprint region in Subaru pixel coordinates
regfile = 'RoughHSTarea_WCS.reg'

# matching tolerance for HST objects to be considered possible match
tolerance = 2 #arcsec

# define ttype column names, since 
objid_sub_id = 'NUMBER'
x_sub_id = 'X_IMAGE'
y_sub_id = 'Y_IMAGE'
ra_sub_id = 'alpha'
dec_sub_id = 'delta'
fwhm_sub_id = 'FWHM_IMAGE'

objid_hst_id = 'number'
ra_hst_id = 'alpha'
dec_hst_id = 'delta'
fwhm_hst_id = 'FWHM_IMAGE814'
mag_hst_id = 'mag_auto814'

# Magnitude limit
maglimit_hst = 27 #objects fainter than this should not influence the 

# Approximate PSF of the Subaru image
psf_sub = 0.7 #arcsec
# when considering whether and HST object is part of a Subaru blend it will be
# import to convolve the HST object with a seeing kernel of this size.

# Pixel scales of the detectors
pixscale_sub = 0.2 #arcsec/pix
pixscale_hst = 0.05 #arcsec/pix

# Largest radius on which to consider objects as possibly being blended, this
# should probably be the radius of the largest object in the field
r_blend_max = 13 #arcsec

# Number of sigma blend. 1 means that they are at least closer than the 1sigma
# distance that define the width of each Gaussian representative of the galaxies
N_sigma = 1.0

# Automated election results file
autoNblendfile = 'NumberOfBlendsPerObject.txt'

# User vote blend file name
userblendfile = 'UserVerifiedBlends.txt'

# File containing the hst and corresponding subaru objid of matched objects
matchesfile = 'HSTSubaruMatches.txt'

## Program
# read in the HST catalog
cat_hst = tools.readcatalog(catalog_hst)
key_hst = tools.readheader(catalog_hst)
# read in the Subaru catalog
cat_sub = tools.readcatalog(catalog_sub)
key_sub = tools.readheader(catalog_sub)

# assign some variables that will be useful in filtering the Subaru catalog
x_sub = cat_sub[:,key_sub[x_sub_id]]
y_sub = cat_sub[:,key_sub[y_sub_id]]
ra_sub = cat_sub[:,key_sub[ra_sub_id]]
dec_sub = cat_sub[:,key_sub[dec_sub_id]]

# filter the subaru catalog, only keeping objects within the HST footprint
mask_hstfootprint = obsplan.createSlitmaskMask(regfile,ra_sub,dec_sub)
cat_sub = cat_sub[mask_hstfootprint,:]

print '{0} Subaru objects are within the HST footprint mask.'.format(numpy.sum(mask_hstfootprint))
print '{0} Subaru objects are excluded by the HST footprint mask.'.format(numpy.sum(~mask_hstfootprint))

# assign variables useful for initial filtering of HST catalog
mag_hst = cat_hst[:,key_hst[mag_hst_id]]

# filter the HST catalog, only keeping objects brighter than the magnitude limit
mask_hst_maglimit = mag_hst <= maglimit_hst
cat_hst = cat_hst[mask_hst_maglimit,:]

# assign variables
objid_sub = cat_sub[:,key_sub[objid_sub_id]]
ra_sub = cat_sub[:,key_sub[ra_sub_id]]
dec_sub = cat_sub[:,key_sub[dec_sub_id]]
fwhm_sub = cat_sub[:,key_sub[fwhm_sub_id]]

objid_hst = cat_hst[:,key_hst[objid_hst_id]]
ra_hst = cat_hst[:,key_hst[ra_hst_id]]
dec_hst = cat_hst[:,key_hst[dec_hst_id]]
fwhm_hst = cat_hst[:,key_hst[fwhm_hst_id]]

'''
Matching Plan:
1) For each Subaru object match the closest HST object (within some tolerance).
   These are defined as primary matches.
2) For each HST object that was not part of a primary match:
2a) Find the Subaru object with the most overlap, where this metric could be
    something like I did in the ipython notebook EllipticityDistOfBlends.ipynb.
    This is done rather than the closest object because amount of overlap
    matters.
2b) Determine if this ancillary match is overlapped enough to be considered a
    blended object.
2c) If so cast a vote that Subaru object being a blend.
3) Define Subaru blends as objects that have at least one vote.
'''

# 1) Match Subaru objects with closest HST object


def primarymatch(ra_sub,dec_sub,ra_hst,dec_hst,objid_hst,fwhm_sub,tolerance):
    '''
    This code is borrowed somewhat from: 
    https://github.com/MCTwo/DEIMOS.git/slitcatmatch.py
    
    Input:
    ra_sub = [1D array of floats; units=degrees] Subaru RA
    dec_sub = [1D array of floats; units=degrees] Subaru Dec
    ra_hst = [1D array of floats; units=degrees] HST RA
    dec_hst = [1D array of floats; units=degrees] HST Dec
    objid_hst = [1D array of ints; unitless] HST object id
    fwhm_sub = [1D array of floats; units=arcsec] Subaru object full-width-half-
       max
    tolerance = [float; units=arcsec] the maximum separation that two objects
       can be and still be considered a match.
    '''
    # Number of HST objects
    N_hst = numpy.size(objid_hst)
    # Create an output array that will have a row for each hst object and a 
    # corresponding column that contains any best Subaru objid match to that HST
    # ojbect
    match_array = numpy.ones((N_hst,2))*(-99)
    match_array[:,0] = objid_hst
    # thus if an HST object has -99 for a corresponding Subaru object it was
    # not a viable match to any Subaru objects

    # Convert the object FWHM into term of sigma, assuming each galaxy is
    # approximately Gaussian
    sigma_sub = fwhm_sub/2.36*pixscale_sub #definition of sigma for a Gaussian

    # Determine quick trim windows for each Subaru object, to reduce
    # calculation time during matching later
    # Note that I could switch this to being tolearance based, rather than sigma
    # based.
    ramin = ra_sub-tolerance/(60.**2*numpy.cos(dec_sub*numpy.pi/180.))
    ramax = ra_sub+tolerance/(60.**2*numpy.cos(dec_sub*numpy.pi/180.))
    decmin = dec_sub-tolerance/(60.**2)
    decmax = dec_sub+tolerance/(60.**2)
    
    
    # Number of Subaru objects to match
    N_sub = numpy.size(ra_sub)
    # Create a blank array to store the primary HST matches
    match_prime_id = numpy.ones(N_sub)*(-99) #nonmatches will have -99 values
    # Loop through Subaru objects looking for matches
    for i in numpy.arange(N_sub):
        #apply quick trim window filter
        mask_rough = numpy.logical_and(numpy.logical_and(ra_hst>ramin[i],
                                                         ra_hst<ramax[i]),
                                       numpy.logical_and(dec_hst>decmin[i],
                                                         dec_hst<decmax[i]))
        N = numpy.sum(mask_rough)
        if N == 0:
            # then there are no viable matches
            continue
        # Calculated the angular separation between quick trim HST objects and
        # the given Subaru object
        ra = ra_hst[mask_rough]
        dec = dec_hst[mask_rough]
        delta = numpy.zeros(N)
        for j in numpy.arange(N):
            delta[j] = numpy.abs(tools.angdist(ra[j],dec[j],
                                               ra_sub[i],dec_sub[i])*60**2)
        # find the closest HST object to identify as primary match
        index = numpy.argmin(delta)
        # store that object id in the match catalog
        match_prime_id[i] = objid_hst[mask_rough][index]
        # store the Subaru id in the match_array
        mask_match = match_array[:,0] == match_prime_id[i]
        match_array[mask_match,1] = objid_sub[i]
        

    mask_sub_nonmatch = match_prime_id == -99
    N_sub_nonmatch = numpy.sum(mask_sub_nonmatch)
    print 'There were {0} ({1:0.2f}%) Subaru objects without a matching HST object.'.format(N_sub_nonmatch,N_sub_nonmatch/N_sub*100)
    # create point regions for the non-match objects
    ds9tools.pointregions('SubaruNonmatchObj',ra_sub[mask_sub_nonmatch],dec_sub[mask_sub_nonmatch])
    return match_prime_id, match_array

match_prime_id, match_array = primarymatch(ra_sub,dec_sub,ra_hst,dec_hst,objid_hst,fwhm_sub,tolerance)



# 2) Identify all HST object that were not matched to a Subaru object
N_hst = numpy.size(objid_hst)

# Create a blank truth array
mask_hst_nonmatch = objid_hst == 0
for i in numpy.arange(N_hst):
    mask_hst_nonmatch[i] = numpy.sum(objid_hst[i] == match_prime_id) == 0
    
N_hst_nonmatch = numpy.sum(mask_hst_nonmatch)

print 'There were {0} ({1:0.2f}%) HST objects without a matching Subaru object.'.format(N_hst_nonmatch,N_hst_nonmatch/N_hst*100)

# create point regions for the non-match objects
ds9tools.pointregions('HSTNonmatchObj',ra_hst[mask_hst_nonmatch],dec_hst[mask_hst_nonmatch],style='X',color='magenta')


def SigConv(sigma_object,sigma_psf):
    '''
    The sigma for two convolved Gaussians, since that is simply another Gaussian.
    '''
    return numpy.sqrt(sigma_object**2+sigma_psf**2)



# 2a) For each HST object without a primary match find the Subaru object with the most overlap

# estimate the size of each HST object as viewed by Subaru
sigma_hst = fwhm_hst/2.36*pixscale_hst #definition of sigma for a Gaussian
# convolve this with the Subaru seeing PSF
sigma_convolv_hst = SigConv(sigma_hst,psf_sub) #units = arcsec

# estimate the sixe of each Subaru object
sigma_sub = fwhm_sub/2.36*pixscale_sub

# define ra and dec arrays for the non-primary-matched HST objects
ra_hst_npm = ra_hst[mask_hst_nonmatch]
dec_hst_npm = dec_hst[mask_hst_nonmatch]

# define ra and dec arrays for the Subaru objects that were found to have a
# primary match with an HST object, since this is a requirement for them being
# a blend in the first place
mask_sub_primematch = match_prime_id != -99
objid_sub_pm = objid_sub[mask_sub_primematch]
ra_sub_pm = ra_sub[mask_sub_primematch]
dec_sub_pm = dec_sub[mask_sub_primematch]
sigma_sub_pm = sigma_sub[mask_sub_primematch]

# determine the rough filter bounds for each blend HST object
ramin = ra_hst-r_blend_max/(60.**2*numpy.cos(dec_hst*numpy.pi/180.))
ramax = ra_hst+r_blend_max/(60.**2*numpy.cos(dec_hst*numpy.pi/180.))
decmin = dec_hst-r_blend_max/(60.**2)
decmax = dec_hst+r_blend_max/(60.**2)

# create a blank blend vote array to house votes for Subaru objects being a
# blend
ballotbox = []

for i in numpy.arange(N_hst):
    # Check if HST was already matched to a Subaru object, if not then match it
    # to the Subaru with the most overlap if any, else continue
    if mask_hst_nonmatch[i]:
        # then the HST object was not previously matched to a subaru object
        # roughly filter the Subaru catalog only keeping object "near" the HST
        # the HST object in question
        mask_rough = numpy.logical_and(numpy.logical_and(ra_sub_pm>ramin[i],
                                                         ra_sub_pm<ramax[i]),
                                       numpy.logical_and(dec_sub_pm>decmin[i],
                                                         dec_sub_pm<decmax[i]))
        N = numpy.sum(mask_rough)
        if N == 0:
            # then there are no possible matches
            continue
        # Calculated the angular separation between quick trim HST objects and
        # the given Subaru object
        objid = objid_sub_pm[mask_rough]
        ra = ra_sub_pm[mask_rough]
        dec = dec_sub_pm[mask_rough]
        sigma = sigma_sub_pm[mask_rough]
        # Calculate the total size of the Subaru and HST object
        sigma_coadd = sigma_convolv_hst[i]+sigma    
        delta = numpy.zeros(N)
        for j in numpy.arange(N):
            # calculate the separation between the objects
            delta[j] = numpy.abs(tools.angdist(ra[j],dec[j],
                                               ra_hst[i],dec_hst[i])*60**2)
        # Calculate the effective separation of the HST and Subaru objects
        sep = delta/(N_sigma*sigma_coadd)
        # Note that if sep < 1 then the objects satisfy the blend criteria
        mask_sep = sep < 1
        # add subaru objid's to blendvote array if passed blend criteria
        ballotbox = numpy.append(ballotbox,objid[mask_sep])
        # if there were any valid matches then associate the Subaru object with the
        # min sep with the HST object
        if numpy.sum(mask_sep > 0):
            # find the closest HST object to identify as primary match
            index = numpy.argmin(sep)
            # store that object id in the match catalog
            match_array[i,1] = objid[index]

# Save the matches
F = open(matchesfile,'w')
F.write('#Created by BlendFinder.py\n')
F.write('#For each HST object, this lists the corresponding Subaru object (if any, if there are no matches then -99 in second column).\n')
F.write('#A single Subaru object may be matched to more than one HST object (i.e. blended).\n')
F.write('#ttype0 = objid_hst\n')
F.write('#ttype1 = objid_subaru\n')
for i in numpy.arange(N_hst):
    F.write('{0:0.0f}\t{1:0.0f}\n'.format(match_array[i,0],match_array[i,1]))
F.close()
    
#3) Tally the votes for Subaru objects being blends
# initialize the ellection polling results array
N_sub = numpy.size(objid_sub)
poll = numpy.zeros(N_sub)

# Save the results of the ellection
F = open(autoNblendfile,'w')
F.write('# Created by BlendFinder.py\n')
F.write('# For each Subaru object this lists the number of blend votes it received \n# in the automated blend detection scheme. If N > 0 then the \n# object is classed as a blend. The number of composite galaxies in the blend \n# is then N+1.\n')
F.write('#ttype0 = objid_subaru\n')
F.write('#ttype1 = N\n')
# for each Subaru object count how many blend votes it received
for i in numpy.arange(N_sub):
    mask_votes = objid_sub[i] == ballotbox
    poll[i] = numpy.sum(mask_votes)
    F.write('{0:0.0f}\t{1:0.0f}\n'.format(objid_sub[i],poll[i]))
F.close()


# Determine which Subaru objects are defined as blends
mask_sub_blends = poll > 0
N_sub_blend = numpy.sum(mask_sub_blends)

print '{0} ({1:0.2f}%) Subaru objects are defined as blends.'.format(N_sub_blend,N_sub_blend/N_sub*100)

# Create ds9 regions for the objects defined as blends
ds9tools.pointregions('SubaruBlendedObjects',ra_sub[mask_sub_blends],dec_sub[mask_sub_blends],style='circle',color='red')

## Visual inspection of blend candidates
'''
Much of this code is copied from ds9backend
'''
import ds9

## USER INPUT

# Subaru image and visulation parameters
file_sub = '/Users/dawson/OneDrive/Research/Clusters/DLSCL09162953/subaru_i_j0916struc.fits'
# Define (min, max) pixel scale values
scale_limits_sub = (0,1000)

# HST image and visualization parameters
file_hst_814 = '/Users/dawson/OneDrive/Research/Clusters/DLSCL09162953/HST/cl0916_f814w_drz_sci.fits'
file_hst_606 = '/Users/dawson/OneDrive/Research/Clusters/DLSCL09162953/HST/cl0916_f606w_drz_sci.fits'
# Define (min, max) pixel scale values
scale_limits_hst_814 = (0,50000)
scale_limits_hst_606 = (0,25000)

# Define Subaru and HST object region files
region_sub = 'Subaru_ellipses.reg'
region_hst = 'HST_ellipses.reg'

# this scale is divided by the object FWHM (in pixels) to determine the zoom
# for that object
zoom_scale = 200

## PROGRAM
# setup ds9
# call ds9
d = ds9.ds9()
# turn off colorbar
d.set('colorbar no')

# load the subaru image in the first frame
cmd = 'file '+file_sub
d.set(cmd)

# correct the WCS issues (deleted PV1_ and PV2_ values)
d.set('wcs replace Subaru.wcs')

# change the scale to log and apply set limits
d.set('scale log')
cmd = 'scale limits {0} {1}'.format(scale_limits_sub[0],scale_limits_sub[1])
d.set(cmd)

# create a new frame
d.set('frame new rgb')
# make that fram active
d.set('frame last')
# by default the r channel is active
# load the HST 814 image into the R channel
cmd = 'file '+file_hst_814
d.set(cmd)
# change the scale
d.set('scale log')
cmd = 'scale limits {0} {1}'.format(scale_limits_hst_814[0],
                                    scale_limits_hst_814[1])
d.set(cmd)

# switch to the G channel and load the HST 814 image
d.set('rgb channel green')
cmd = 'file '+file_hst_814
d.set(cmd)
# change the scale
d.set('scale log')
cmd = 'scale limits {0} {1}'.format(scale_limits_hst_814[0],
                                    scale_limits_hst_814[1])
d.set(cmd)

# switch to the G channel and load the HST 814 image
d.set('rgb channel blue')
cmd = 'file '+file_hst_606
d.set(cmd)
# change the scale
d.set('scale log')
cmd = 'scale limits {0} {1}'.format(scale_limits_hst_606[0],
                                    scale_limits_hst_606[1])
d.set(cmd)

# show both frames
d.set('tile yes')

# set the subaru frame as active
d.set('frame first')

# lock the crosshairs of the two images
d.set('crosshair lock wcs')

# load Subaru object regions into both frames
cmd = 'regions load all '+region_sub
d.set(cmd)
cmd = 'regions load all '+region_hst
d.set(cmd)

# set up an array to contain the user's votes on the blend candidates,
# default is False so that non-blend candidates don't need to be inspected
votes_user = numpy.zeros(N_sub) != 0
# create a file to record the user votes
# clear the old file
F = open(userblendfile,'w')
F.close()
# create the file in append mode
F = open(userblendfile,'a')
F.write('#Output from BlendFinder.py\n')
F.write('#Subaru Object ID \t Blend?\n')
# now loop through the blend candidates and ask for visual confirmation
for i in numpy.arange(N_sub):
    if mask_sub_blends[i]:
        # then object was flagged as a possible blend
        # get the blend candidate parameters from the subaru catalog
        objid_blend = cat_sub[i,key_sub[objid_sub_id]]
        ra_blend = cat_sub[i,key_sub[ra_sub_id]]
        dec_blend = cat_sub[i,key_sub[dec_sub_id]]
        fwhm_blend = cat_sub[i,key_sub[fwhm_sub_id]]
        # pan to the blend candidate
        cmd = 'pan to {0} {1} wcs fk5 degrees'.format(ra_blend,dec_blend)
        d.set(cmd)
        # place the crosshair over the object of interest
        cmd = 'crosshair {0} {1} wcs fk5 degrees'.format(ra_blend,dec_blend)
        d.set(cmd)
        # zoom in on the object
        zoom = zoom_scale/fwhm_blend
        cmd = 'zoom to {0}'.format(zoom)
        d.set(cmd)
        # match the two frames
        d.set('match frame wcs')
        # Ask the user if it is a blend
        selection = raw_input('Is this a blend? blank/n: ')
        if selection == 'n':
            votes_user[i] = False
        elif selection == '':
            votes_user[i] = True
        else:
            selection = raw_input("Input invalid. Please enter n followed by return for no, or simply hit return for yes. Is this a blend?")
        F.write('{0:0.0f}\t{1}\n'.format(cat_sub[i,key_sub[objid_sub_id]],votes_user[i]))
    else:
        # object was not flagged as a blend go on to the next object
        F.write('{0:0.0f}\t{1}\n'.format(cat_sub[i,key_sub[objid_sub_id]],False))
        continue
# tally the user votes
N_sub_user = numpy.sum(votes_user)
mask_sub_user = votes_user

print '{0} ({1:0.2f}%) Subaru objects are defined as blends by the user.'.format(N_sub_user,N_sub_user/N_sub*100)
print '{0:0.2f}% of the Subaru blend candidate objects are defined as blends by the user.'.format(N_sub_user/N_sub_blend*100)

# save the user defined blend votes to file
numpy.savetxt('UserVotedSubaruBlends',mask_sub_user)


# Plot the results of the automated blend detection algorithm

e1_sub_id = 'e1'
e2_sub_id = 'e2'

e1_sub_blend = cat_sub[mask_sub_blends,key_sub[e1_sub_id]]
e2_sub_blend = cat_sub[mask_sub_blends,key_sub[e2_sub_id]]

e1_sub_nonblend = cat_sub[~mask_sub_blends,key_sub[e1_sub_id]]
e2_sub_nonblend = cat_sub[~mask_sub_blends,key_sub[e2_sub_id]]

import CDFanalysis

CDFanalysis.CDF_boot_analysis(abs(e1_sub_blend),abs(e1_sub_nonblend),N_boot=10000,N_resample=1000,prefix='e1_musketball_newblendmethod',a_name='Blends',b_name='Non-blends')

CDFanalysis.CDF_boot_analysis(abs(e2_sub_blend),abs(e2_sub_nonblend),N_boot=10000,N_resample=1000,prefix='e2_musketball_newblendmethod',a_name='Blends',b_name='Non-blends')

# Concatinate the e1 and e2 arrays
e_12_blend = numpy.concatenate((abs(e1_sub_blend),abs(e2_sub_blend)))
e_12_nonblend = numpy.concatenate((abs(e1_sub_nonblend),abs(e2_sub_nonblend)))

CDFanalysis.CDF_boot_analysis(e_12_blend,e_12_nonblend,N_boot=10000,N_resample=1000,prefix='e12_musketball_newblendmethod',a_name='Blends',b_name='Non-blends')

# Plot the results of the user verified blends

e1_sub_user = cat_sub[mask_sub_user,key_sub[e1_sub_id]]
e2_sub_user = cat_sub[mask_sub_user,key_sub[e2_sub_id]]

e1_sub_nonuser = cat_sub[~mask_sub_user,key_sub[e1_sub_id]]
e2_sub_nonuser = cat_sub[~mask_sub_user,key_sub[e2_sub_id]]

import CDFanalysis

CDFanalysis.CDF_boot_analysis(abs(e1_sub_user),abs(e1_sub_nonuser),N_boot=10000,N_resample=1000,prefix='e1_musketball_newusermethod',a_name='Blends',b_name='Non-blends')

CDFanalysis.CDF_boot_analysis(abs(e2_sub_user),abs(e2_sub_nonuser),N_boot=10000,N_resample=1000,prefix='e2_musketball_newusermethod',a_name='Blends',b_name='Non-blends')

# Concatinate the e1 and e2 arrays
e_12_user = numpy.concatenate((abs(e1_sub_user),abs(e2_sub_user)))
e_12_nonuser = numpy.concatenate((abs(e1_sub_nonuser),abs(e2_sub_nonuser)))

CDFanalysis.CDF_boot_analysis(e_12_user,e_12_nonuser,N_boot=10000,N_resample=1000,prefix='e12_musketball_newusermethod',a_name='Blends',b_name='Non-blends')

print 'finished'