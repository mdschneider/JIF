#!/bin/tcsh
#
# run_psf_sampling_test
#
#  Created by Michael Schneider on 2016-01-25
#
## Check usage
if ($#argv != 1) then
    echo "Usage: run_psf_sampling_test.sh nsamples"
    echo "Test pipeline for sampling PSF model parameters in Roaster"
    goto done
endif

set outdir=../output/roasting

# python galsim_galaxy.py
python Roaster.py ../TestData/test_image_data.h5 --nsamples $1 --telescope LSST --model_params nu hlr e beta flux_sed1 dx dy psf_fwhm psf_e psf_beta --quiet --nwalkers 32
python RoasterInspector.py $outdir/roaster_out_LSST.h5 --truths 0.3 1.0 0.1 0.785 1.e4 0.0 0.0 0.6 0.01 0.4
open $outdir/roaster_out_LSST_data_and_model_epoch0.png $outdir/roaster_out_LSST_roaster_inspector_triangle.png $outdir/roaster_out_LSST_roaster_inspector_walkers.png

## labels for proper exit or error
done:
    # growlnotify -s -n Script -m "Finished run_psf_sampling_test"
    exit 0
error:
    exit 1
