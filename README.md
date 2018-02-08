# JIF
Joint Image Framework (JIF) for probabilistic modeling of astronomical images of stars and galaxies as seen by multiple telescopes.

## Motivation

How do we optimally combine of galaxies seen from space and ground? The different PSFs, wavelength coverage,
and pixel sizes can lead to biases in inferred galaxy properties unless included in a joint model
of all images of the same source. If sources are blended together in any observations, the need for
joint modeling becomes even more acute.

## People

- Will Dawson (LLNL)
- Michael Schneider (LLNL)
- Joshua Meyers (Stanford)

## Analysis pipeline steps

1. Shell - source extraction
2. Roast - Interim sampling of source model parameters
3. Grind - Hierarchical inference of source distributions via importance sampling

## Installation

Everything in this project is in python (not including our external dependencies).
Install with,

    python setup.py install

or

    python setup.py develop

to install while working on the package.

## Tools / dependencies

- Our image models are built around the [GalSim](https://github.com/GalSim-developers/GalSim/wiki) image simulation framework.
- For some of the comparative analysis we require [LSST DM Stack](https://confluence.lsstcorp.org/display/LSWUG/LSST+Software+User+Guide). Note that it is possible to install GalSim as part of the LSST DM environment, see [Building GalSim with the LSST stack](https://github.com/GalSim-developers/GalSim/wiki/Building-GalSim-with-the-LSST-stack).
  - Note that if using the LSST stack install of GalSim this needs to be "setup", by runnning `source loadLSST.bash` then `setup GalSim` in the LSST stack install location.  
- For parameter inference, we use [emcee](http://dan.iel.fm/emcee/current/) (and probably other samplers TBD).
- The sources (and source groups) we extract from raw imaging are stored in [HDF5](http://www.hdfgroup.org/HDF5/) file formats, with a custom grouping.
- For part of the results visualization we use [corner](https://github.com/dfm/corner.py).

## Test Analysis

The following procedure outlines the steps that can be take to create some test data with `galasim_galaxy`, analyze it with `Roaster`, and inspect the results with `RoasterInspector`. This example does not require installation via `setup.py`.

1. Create the directory structure (if not already in place). cd to the parent repository directory (i.e. the one containing jif).

  ```
  mkdir -p data/TestData
  mkdir -p output/TestData
  ```
2. Generate the test data with GalSim:

  ```
  cd jiffy
  python galsim_galaxy.py
  ```
  This will make the file `test_image_data.h5` in data/TestData
3. Run the roaster on this data:

  ```
  python roaster.py --config_file ../config/roaster_defaults.cfg
  ```
  This will create an hdf5 results file `../output/roasting/roaster_out.h5`.
4. Inspect the data with `RoasterInspector`. 

  ```
  python RoasterInspector.py ../output/roasting/roaster_out.h5 ../config/roaster_defaults.cfg
  ```
  This will print summary statistics and make some plots.
