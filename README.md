# JIF
Joint Image Framework (JIF) for probabilistic modeling of astronomical images of stars and galaxies as seen by multiple telescopes.

## Motivation

How do we optimally combine of galaxies seen from space and ground? The different PSFs, wavelength coverage,
and pixel sizes can lead to biases in inferred galaxy properties unless included in a joint model
of all images of the same source. If sources are blended together in any observations, the need for
joint modeling becomes even more acute.

## Analysis

This package embeds [GalSim](https://github.com/GalSim-developers/GalSim) image models of galaxies and stars into a Markov Chain Monte Carlo (MCMC) framework for probabilistic forward modeling of images. The primary module is `jiffy/roaster.py`, which defines the image model likelihood. Most parameters to `roaster` can be specified in configuration files as in `config/jiffy.yaml`. 

## Installation

Everything in this project is in python (not including our external dependencies).
Install with,

    python setup.py install

or

    python setup.py develop

to install while working on the package.

Install all python requirements with,

    pip install requirements.txt

## Tools / dependencies

- Our image models are built around the [GalSim](https://github.com/GalSim-developers/GalSim/wiki) image simulation framework.  
- For parameter inference, we use [emcee](http://dan.iel.fm/emcee/current/).
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
  python roaster.py --config_file ../config/jiffy.yaml
  ```
  This will create an hdf5 results file `../output/TestData/jiffy_roaster_out_seg0.h5`.
4. Inspect the data with `RoasterInspector`. 

  ```
  python roaster_inspector.py ../output/TestData/jiffy_roaster_out_seg0.h5 ../config/jiffy.yaml
  ```
  This will print summary statistics and make some plots.
