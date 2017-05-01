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
if ($#argv != 1) then
    echo "Usage: run_jif_cgc_test.sh field"
    echo "Run JIF Roaster for a single 'field' of a GREAT3 branch"
    echo "The field values should be in the format, e.g., '009'"
    goto done
endif

##
## Set some top-level parameters
##
set user=`whoami`
set workdir=./
# set workdir=./midsnr

set config_file=jiffy.yaml
# set config_file=jiffy_midsnr.yaml
set field=$1

##
## Modify the config file fed to Roaster for the currently selected field
##
rm -rf update_config.py

cat >>update_config.py <<EOF
#!/usr/bin/env python
import yaml

config = yaml.load(open("${config_file}"))
config['io']['infile'] = '${workdir}/control/ground/constant/segments/seg_${field}.h5'
config['io']['roaster_outfile'] = '${workdir}/reaper/jif/${field}/roaster_${field}'

with open("${config_file}", "w") as f:
    yaml.dump(config, f, indent=4, default_flow_style=False)
EOF

python update_config.py

##
## Run Roaster and Roasting Inspector for each footprint in the selected field
##
foreach segnum (`seq 0 3`)
	# Edit parameter file to contain true (sheared) ellipticities

	python update_roaster_params.py --field $field --gal $segnum || goto error

	echo " "
	echo "================================================="
	echo "Fitting segment number "$segnum
	echo "================================================="
	jiffy_roaster --config_file $config_file --footprint_number $segnum || goto error

	echo " "
	echo "================================================="
	echo "Making Inspector plots for segment "$segnum
	echo "================================================="
	set roaster_file=${workdir}/reaper/jif/${field}/roaster_${field}_seg${segnum}.h5
	echo jiffy_roaster_inspector $roaster_file $config_file	
	jiffy_roaster_inspector $roaster_file $config_file --footprint_number $segnum || goto error	
end

##
## labels for proper exit or error
## 
done:
    echo " "
    echo "Finished run_jif_cgc_test"
    echo " "
    exit 0
error:
    exit 1
