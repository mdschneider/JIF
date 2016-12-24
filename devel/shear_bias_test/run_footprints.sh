#!/bin/tcsh
#
# @file run_footprints.sh
# 
# See MagicBeans/devel/shear_bias_test_167 for the input file generation.
#
#  Created by Michael Schneider on 2016-12-23
#
## Check usage
if ($#argv != 0) then
    echo "Usage: run_footprints.sh"
    echo "Create footprints files from CGC-like input FITS images"
    goto done
endif

## Path to the input images.
## Assume we're working within JIF/devel/shear_bias_test/
## Assume JIF and MagicBeans are installed in the same top-level directory.
set datadir=../../../MagicBeans/devel/shear_bias_test_167/

## Have 20 x 20 galaxies in each 'field'.
## See MagicBeans/devel/shear_bias_test_167/mbi_no_shape_noise.yaml
set n_gals=400

foreach subfield_index (`seq 0 49`)
	# echo "Creating footprint file for CGC-like field "$subfield_index
	jif_sheller --subfield_index $subfield_index --data_path $datadir \
	--catfile_head "epoch_catalog" \
	--n_gals $n_gals || goto error
end

## labels for proper exit or error
done:
    # growlnotify -n Script -m "Finished shear_bias_test/run_footprints"
    echo " "
    echo "Finished shear_bias_test/run_footprints"
    echo " "
    exit 0
error:
    exit 1
