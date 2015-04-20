# JIF
Joint Image Framework (JIF) for modeling astronomical images of stars and galaxies as seen by multiple telescopes.

## Motivation

How do we optimally combine of galaxies seen from space and ground? The different PSFs, wavelength coverage, 
and pixel sizes can lead to biases in inferred galaxy properties unless included in a joint model 
of all images of the same source. If sources are blended together in any observations, the need for 
joint modeling becomes even more acute.

## People

- Will Dawson (LLNL)
- Michael Schneider (LLNL)
- Joshua Meyers (Stanford)

## Steps in making peanut butter (i.e., cosmology from images)

1. Shell - source extraction from space data
2. Roast - Interim sampling
3. Grind - Hierarchical inference via importance sampling
4. Mix / salt - posterior inferences

## Tools / dependencies

- Our image models are built around the [GalSim](https://github.com/GalSim-developers/GalSim/wiki) image simulation framework.
- For parameter inference, we use [emcee](http://dan.iel.fm/emcee/current/) (and probably other samplers TBD).
- The sources (and source groups) we extract from raw imaging are stored in [HDF5](http://www.hdfgroup.org/HDF5/) file formats, with a custom grouping.
 
## Test Analysis

The following proceedure outlines the steps that can be take to create some test data with `galasim_galaxy`, analyze it with `Roaster`, and inspect the results with `RoasterInspector`.

1. Create the directory structure (if not already in place). cd to the parent repository directory (i.e. the one containing jif).
  
  ```
  mkdir TestData
  mkdir output
  mkdir output/roasting
  ```
2. Generate the test data with galsim:
  
  ```
  cd jif
  python galsim_galaxy.py
  ```
  This will make the file `test_image_data.h5` in TestData
3. Run the roaster on this data:
  
  ```
  python Roaster.py ../TestData/test_image_data.h5
  ```
  This will create an hdf5 results file `../output/roasting/roaster_out.h5`.
4. Inspect the data with `RoasterInspector` 
  
  ```
  python RoasterInspector.py ../output/roasting/roaster_out.h5
  ```
