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