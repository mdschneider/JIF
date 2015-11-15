#!/bin/tcsh
#
# run_great3_cgc
#
#  Created by Michael Schneider on 2015-11-08
#
## Check usage
if ($#argv != 2) then
    echo "Usage: run_great3_cgc.sh segment_number nsteps"
    echo "Run Roaster.py on a GREAT3 galaxy from CGC sub-field 000"
    goto done
endif

python Roaster.py ../great3/control/ground/constant/segments/seg_000.h5 --segment_numbers $1 --outfile ../output/great3/roaster_CGC_000_$1 --seed 2586766 --nsamples $2 --nwalkers 16 --nburn 1 --quiet

## labels for proper exit or error
done:
    growlnotify -n Script -m "Finished run_great3_cgc"
    exit 0
error:
    exit 1
