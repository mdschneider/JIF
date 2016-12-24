#!/bin/tcsh
#
# @file run_jif_cgc_test.sh
# 
# Modified from ../great3_cgc/run_jif_cgc.sh
#
# Run JIF Roaster for a single 'field' of a GREAT3-like branch.
#
# Requires: 'sheller' has already been run to create a footprints file,
# 			via the run_footprints.sh script in this directory.
# 			
# See MagicBeans/devel/shear_bias_test_167 for the input file generation.
#
#  Created by Michael Schneider on 2016-12-23.
#
## Check usage
if ($#argv != 0) then
    echo "Usage: run_jif_cgc_test.sh"
    echo "Run JIF Roaster for a single 'field' of a GREAT3 branch"
    goto done
endif

set config_file=roaster_cgc.cfg
set field=000

### Run Roaster
foreach segnum (`seq 0 399`)
	echo " "
	echo "================================================="
	echo "Fitting segment number "$segnum
	echo "================================================="
	jif_roaster --config_file $config_file --segment_number $segnum || goto error

	echo " "
	echo "================================================="
	echo "Making Inspector plots for segment "$segnum
	echo "================================================="
	set roaster_file=/Users/mdschnei/work/MagicBeans/devel/shear_bias_test_167/reaper/JIF/${field}/roaster_${field}_seg${segnum}_LSST.h5
	echo jif_roaster_inspector $roaster_file $config_file	
	jif_roaster_inspector $roaster_file $config_file --segment_number $segnum || goto error	
end

## labels for proper exit or error
done:
    # growlnotify -s -n Script -m "Finished run_jif_cgc_test"
    echo " "
    echo "Finished run_jif_cgc_test"
    echo " "
    exit 0
error:
    exit 1
