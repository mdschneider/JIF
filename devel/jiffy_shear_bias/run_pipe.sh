#!/bin/tcsh
#
# Execute the top-level pipeline for shear bias testing
#

set galsim=~/code/GalSim/bin/galsim

set datadir=/Volumes/PromisePegasus/JIF/cgc2/
set n_gals=10000
set n_fields=200
set sim_yaml=mbi_cgc2.yaml

#
# Make the directory tree for the data
#
mkdir -p ${datadir}/control/ground/constant
mkdir -p ${datadir}/reaper/jif
mkdir -p ${datadir}/thresher/CPP

#
# Simulate the data
#
$galsim $sim_yaml
$galsim mbi_no_shape_noise_psf.yaml
mv *.fits ${datadir}/control/ground/constant

#
# Extract footprints
#
./run_footprints.sh $n_gals $n_fields

#
# Run Roaster on all footprints across all fields
# (takes a while)
#
# ./run_jif_all_fields.sh $n_gals
# ./run_stooker_all_fields.sh

# [run Thresher here]