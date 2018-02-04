## currently there is an issue with  os.system('setup pipe_tasks') command
## not being interpreted correctly so these are just run from the terminal
## manually before calling python.
#import os
#os.system('source ~/lsst/loadLSST.bash')
#os.system('setup pipe_tasks')

import lsst.afw.math        as math
import lsst.afw.table       as afwTable
import lsst.afw.image       as afwImg
import lsst.afw.detection   as afwDetect
import lsst.meas.algorithms as measAlg

## User input
fitsfile = '../TestData/test_lsst_image_paper1_ex1.fits'
gain = 2.1

# load the fits image into an exposure
exposure = afwImg.ExposureF(fitsfile)

# add links to the various exposure parts
maskedImage = exposure.getMaskedImage()
image = maskedImage.getImage()
mask = maskedImage.getMask()
variance = maskedImage.getVariance()

## Note that we need to manually make the variance image; lets just define
## this a image pixel values / gain
## other options are in https://github.com/LSST-nonproject/obs_file/blob/master/python/lsst/obs/file/processFile.py
variance_array = variance.getArray()
image_array = image.getArray()
variance_array[:] = image_array / gain

## Currently we don't do anything with the mask

# Configure the detection and measurement algorithms
schema                = afwTable.SourceTable.makeMinimalSchema()
detectSourcesConfig   = measAlg.SourceDetectionConfig(thresholdType='value')
measureSourcesConfig  = measAlg.SourceMeasurementConfig()

# Setup the detection and measurement tasks
detect  = measAlg.SourceDetectionTask(config=detectSourcesConfig,  schema=schema)
# turn off the background reestimation since this is such a small field, otherwise it won't
# will not detect the source due to subtracting it as background
detectSourcesConfig.reEstimateBackground = False
measure = measAlg.SourceMeasurementTask(config=measureSourcesConfig, schema=schema)

# Set flux aliases to None; a hack for an incompatability between
# makeMinimalSchema() and the default SourceMeasurementConfig() options.
measureSourcesConfig.slots.psfFlux    = None
measureSourcesConfig.slots.apFlux     = None
measureSourcesConfig.slots.modelFlux  = None
measureSourcesConfig.slots.instFlux   = None
measureSourcesConfig.validate()

# Detect the sources,then put them into a catalog (the table is where the
# catalog atually stores stuff)
table   = afwTable.SourceTable.make(schema)
catalog = detect.makeSourceCatalog(table, exposure, sigma=5)

# Get the sources out of the catalog
sources = catalog.sources

# Apply the measurement routines to the exposure using the sources as input
measure.run(exposure, sources)

# Now let's look at the output from some of the measurment algorithms.
fields = ['centroid.sdss', 'shape.sdss', 'shape.sdss.centroid','flux.gaussian']
keys   = [schema.find(f).key for f in fields]

for source in sources:
    print source.getCentroid() #This uses one of the aliases

    # Now loop through the keys we want
    for f,k in zip(fields, keys):
        print '    ', f, source.get(k)