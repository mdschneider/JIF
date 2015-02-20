# Summary
This is currently some rough code with the objective of taking an image and generating an hdf5 catalog to be input into the roaster. This breaks down into two basic steps: harvesting and shelling.

## Harvesting

Harvesting is basically the reduction proceedure performed by the DM stack pipeline (or similarly Source Extractor). The important output being:

* A catalog of objects with basic properties (location, flux, size, etc.)
* A segmentation map (to determine groups; CHECKIMAGE_TYPE SEGMENTATION option of sextractor), actually note that the Sextractor segmentation image will not work since it already breaks appart blended objects. Thus we will need to make our own segementation image.
* A background map (CHECKIMAGE_TYPE BACKGROUND option of sextractor)
* A noise map (CHECKIMAGE_TYPE BACKGROUND_RMS option of sextractor)
* A background subtracted image (CHECKIMAGE_TYPE -BACKGROUND option of sextractor)

## Shelling 

Shelling is the reduction proceedure of taking the output of the Harvestor create the MIMEDS (Multi-Instrument Multi-Epoch Data Structure) file. The basic information to be stored and the data structure can be seen [here](https://www.lucidchart.com/documents/view/3123e050-bbfc-4167-9dba-4659715d668b).

The basic shelling proceedure is as follows:

1. Unload peanuts: Read in the DM stack (or SourceExtractor) output from the harvesting stages of the various epochs, bands, or instruments. 
2. Size and screen peanut pods: Determine which image from the various epochs, band, or instrument has the worst seeing (or perhaps largest pixels if for some reason the image is not Nyquist sampled).
3. Shell the peanuts: Use the segmentation map from that image to define the postage stamp extents in all of the images as well as segment group membership for each of the images (using the catalog output of the Harvest in conjunction with the segmentation bounds).
4. Bag the peanuts: package the input Harvest results as well as the group properties and postage stamps in the hdf5 file structure.

# Current Proceedure
Since I am just trying to hack together something so that the Harvestor can be tested I will note here exactly what I am doing and where these steps fall short of our ultimate objective.

Ideally I would:

* Run sextractor on both the Subaru and HST image outputting the aformentioned files in the Harvesting section
* Use the Subaru segmentation fits file to define group membership and postage stamps in both the Subaru and HST data.

There are several reasons I don't want to do this at this stage. 

* At the start I would like to avoid the complexities of using the segmentation map in one image to define it in another image, similarly for the resulting postage stamps.
* By using the same image for the space and ground (but with the ground convolved with some "seeing") it should provide for an easier initial comparison of the roaster results.

So I think what I will do is:

1. Run Sextractor on the HST F814W image to create the necessary Harvester outputs.
2. Convolve the HST F814 image with a Gaussian approximating the Subaru seeing (this image will represent a ground based image).
3. Run Sextractor on this image to generate the necessary Harvester outputs.
4. Use the "ground" segmentation map to determine groups.
5. Save the results in two different hdf5 files.


#To Do
Since the current objective is to quickly hack together something so that the Roaster can be tested, the following is a to do list where action items are meant to ultimately remedy the shortcuts taken.

- [ ] 

# Detailed Analysis Notes
Used convolveHST.py to make '../../TestData/cl0916_f814w_drz_sci_convolved.fits'

