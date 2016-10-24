#!/bin/tcsh
#
# run_blend_examples
#
## Check usage
if ($#argv != 0) then
    echo "Usage: run_blend_examples.sh"
    echo "Simulate & fit blend stamps in LSST & WFIRST passbands"
    goto done
endif


### Loop over stamps
foreach stamp_index (`seq 0 1 8`)
	@ counter = 0
	### Loop over bands
	foreach passband (u g r i z y Y106 J129 H158 F184)
		if ($counter < 6) then
			set tel=LSST
		else
			set tel=WFIRST
		endif

		echo $stamp_index $passband $tel

		set workdir=blend_fit_test/stamp${stamp_index}/$passband
		mkdir -p $workdir

		# Copy the parameter files for the image
		set roaster_params=jif_roaster_params_${stamp_index}.cfg
		cp blend_fit_test/$roaster_params $workdir

		### Create the Roaster config file
		set roaster_config=${workdir}/jif_roaster_settings.cfg
		### The `cat` method here can't have indented lines in the shell script
cat >>$roaster_config <<EOF
; This is an example config file that can be passed to Roaster
; Most parameters are optional and will be given defaults by Roaster
; if not present.
;
; Input image files to process
; If more than 1 input file, list as separate key=value pairs
;
[infiles]
infile_1=${workdir}/roaster_model_image.h5

;
; Control some top-level settings of the Roaster script
;
[metadata]
; Output HDF5 file to record posterior samples and ln-posterior densities
; The segment index and the extension will be added to the outfile
outfile=${workdir}/roaster

;
; Control what input data is loaded to Roaster
;
[data]
; Specifier indicating format of the input file
data_format=jif_segment
; Index of the 'segments' or 'footprints' to load from each input file.
; Must be a single integer value of one segment.
; Roaster can only fit models to one segment at a time.
; The segment_number specified here can be overridden by a command line argument 
; to JIF Roaster.
segment_number=0
; If provided, select only the names single telescope data from the input file.
telescope=${tel}
; Names of a subset of filters to load from the input file.
filters=${passband}
; -1 means get all available epochs
epoch_num=-1

;
; Control the source model 
;
[model]
; Type of parametric source model.
; Can be 'Spergel', 'Sersic', 'BulgeDisk', 'star' - see parameters.py
galaxy_model_type=Spergel
; Names of the source model parameters to sample in.
; Must be consistent with the 'galaxy_model_type'.
model_params=nu hlr e beta mag_sed1 dx dy
; model_params=nu mag_sed1 dx dy
; How many sources in the single footprint?
num_sources=2
; Use an achromatic galaxy or star model?
; If True then skip all the GalSim ChromaticObject stuff.
; This means no rendering of SEDs through passbands or chromatic aberrations.
; Also, it makes the 'mag' parameters less relateable to realistic values.
achromatic=True

;
; Control the intial state of the Roaster instance
;
[init]
; Name of a config file with model parameter values to initialize the MCMC chain.
; This file can contain non-sampling parameters, in which case those parameters 
; are asserted to have the values in the file (useful to impose known 'truths').
init_param_file=${workdir}/${roaster_params}
; Seed for the pseudo-random number generator.
seed=9216526

;
; Control the runtime settings
;
[run]
; Suppress most standard outputs (does not apply to 'debug' statements)
quiet=True
; Print extra debugging information
debug=False
; Just save an image of the model initialized in Roaster?
; If false then proceed to MCMC sampling.
output_model=False

;
; Control the MCMC settings
;
[sampling]
; Name of the MC sampler to use. 
; Available samplers: 'emcee', 'sirs'
sampler=emcee
; Number of samples for each emcee walker
nsamples=2000
; Number of emcee walkers
nwalkers=32
; Number of burn-in steps to discard from the start of the MCMC chain.
nburn=1
; Number of compute threads to use - buggy
nthreads=1
EOF
		@ counter += 1

		### Simulate the image for this stamp and passband
		python sim_image_from_roaster_config.py $roaster_config
	end
end

## labels for proper exit or error
done:
    # growlnotify -s -n Script -m "Finished run_blend_examples"
    exit 0
error:
    exit 1
