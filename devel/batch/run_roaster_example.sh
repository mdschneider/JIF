#!/bin/tcsh
#
# run_roaster_example
#
# Run a simple pipeline for Roaster:
# - Generate fake data
# - Apply Roaster to the fake data
# - Make diagnostic plots
#
#  Created by Michael Schneider on 2015-04-02
#
## Check usage
if ($#argv != 1) then
    echo "Usage: run_roaster_example.sh nsteps"
    echo "Run the Roaster with test input images from galsim_galaxy and make MCMC diagnostic plots"
    goto done
endif

### Assuming make_test_images() gets called here:
python galsim_galaxy.py
python Roaster.py ../TestData/test_image_data.h5 --nsamples $1 --seed 253674 --nwalkers 16 --nburn 100 --nthreads 1 --quiet
python RoasterInspector.py ../output/roasting/test_roaster_out.h5 --truths 1.8 0.3 0.7854

## labels for proper exit or error
done:
    growlnotify -s -n Script -m "Finished run_roaster_example"
    exit 0
error:
    exit 1
