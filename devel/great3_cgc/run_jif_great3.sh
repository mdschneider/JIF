#!/bin/tcsh
#
# run_jif_great3
#
# Run JIF Roaster for a single 'field' of a GREAT3 branch.
#
# Requires: 'sheller' has already been run to create a footprints file.
#
#  Created by Michael Schneider on 2016-08-04
#
## Check usage
if ($#argv != 0) then
    echo "Usage: run_jif_great3.sh"
    echo "Run JIF Roaster for a single 'field' of a GREAT3 branch"
    goto done
endif

set config_file=devel/great3_cgc/roaster_cgc.cfg

### Run Roaster
foreach segnum (`seq 0 1000`)
	echo " "
	echo "================================================="
	echo "Fitting segment number "$segnum
	echo "================================================="
	jif_roaster --config_file $config_file --segment_number $segnum || goto error

	echo " "
	echo "================================================="
	echo "Making Inspector plots for segment "$segnum
	echo "================================================="
	set roaster_file=output/great3/CGC/000/roaster_CGC_000_seg${segnum}_LSST.h5
	jif_roaster_inspector $roaster_file $config_file || goto error	
end

## labels for proper exit or error
done:
    # growlnotify -s -n Script -m "Finished run_jif_great3"
    exit 0
error:
    exit 1
