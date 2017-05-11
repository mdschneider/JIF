#!/bin/tcsh
#
# Execute the top-level pipeline for shear bias testing
#

set galsim=~/code/GalSim/bin/galsim

set datadir=small_shapenoise
set n_gals = 400
set sim_yaml=mbi_small_shape_noise_highsnr.yaml

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
./run_footprints.sh $n_gals

#
# Run Roaster on all footprints across all fields
# (takes a while)
#
./run_jif_all_fields.sh $n_gals
./run_stooker_all_fields.sh

# [run Thresher here]