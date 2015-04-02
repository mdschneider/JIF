#!/bin/tcsh
#
# run_roaster_example
#
#  Created by Michael Schneider on 2015-04-02
#
## Check usage
if ($#argv != 1) then
    echo "Usage: run_roaster_example.sh nsteps"
    echo "Run the Roaster with test input images from galsim_galaxy and make MCMC diagnostic plots"
    goto done
endif

python galsim_galaxy.py
python Roaster.py test_image_data.h5 --nsamples $1 --nwalkers 16 --nburn 500 --nthreads 1
python RoasterInspector.py ../output/roasting/roaster_out.h5 --truths 1e5 -0.3 1.8 0.3 0.7854

## labels for proper exit or error
done:
    growlnotify -s -n Script -m "Finished run_roaster_example"
    exit 0
error:
    exit 1
