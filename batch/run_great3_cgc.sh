#!/bin/tcsh
#
# run_great3_cgc
#
#  Created by Michael Schneider on 2015-11-08
#
## Check usage
if ($#argv != 0) then
    echo "Usage: run_great3_cgc.sh"
    echo "Run Roaster.py on a GREAT3 galaxy from CGC sub-field 000"
    goto done
endif

foreach subfield_index (`seq 2 200`)
	echo "Creating footprint file for CGC field "$subfield_index
	jif_sheller --subfield_index $subfield_index --n_gals 10000
end

# jif_roaster ../great3/control/ground/constant/segments/seg_000.h5 --segment_numbers $1 --outfile ../output/great3/CGC/000/roaster_CGC_000_$1 --seed 2586766 --nsamples $2 --nwalkers 32 --nburn 1 --quiet --model_params 'nu' 'hlr' 'e' 'beta' 'flux_sed1' 'dx' 'dy'

# jif_roaster_inspector ../output/great3/CGC/000/roaster_CGC_000_$1.h5 --keeplast 500

## labels for proper exit or error
done:
    growlnotify -n Script -m "Finished run_great3_cgc"
    exit 0
error:
    exit 1
