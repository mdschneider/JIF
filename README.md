# JIF
Joint Image Framework (JIF) for probabilistic modeling of astronomical images of stars and galaxies in optical wavelengths.

## Motivation

How do we optimally combine images of galaxies seen from space and ground? The different PSFs, wavelength coverage,
and pixel sizes can lead to biases in inferred galaxy properties unless included in a joint model
of all images of the same source. If sources are blended together in any observations, the need for
joint modeling becomes even more acute.

## Analysis

This package embeds [GalSim](https://github.com/GalSim-developers/GalSim) image models of galaxies and stars into a Markov Chain Monte Carlo (MCMC) framework for probabilistic forward modeling of images. The primary module is `jiffy/roaster.py`, which defines the image model likelihood. Most parameters to `roaster` can be specified in configuration files as in `config/jiffy.yaml`. 

## Installation

To create a conda environment named "jiftutorial" and install the minimum necessary packages:

    conda create -n jiftutorial python=3.8.12 numpy tqdm h5py yaml matplotlib jupyter astropy pandas scipy
    conda activate jiftutorial
    conda install -c conda-forge galsim emcee scikit-learn
    python -m pip install -U corner

In addition, this package (JIF) as well as the [footprints](https://github.com/mdschneider/footprints) package need to be cloned and installed from their respective repos, as follows:

    git clone git@github.com:mdschneider/footprints.git
    git clone git@github.com:mdschneider/JIF.git
    cd footprints
    python setup.py install
    cd ../JIF
    python setup.py install

### More details

Everything in this project is in python (not including our external dependencies).
Install with,

    python setup.py install

or

    python setup.py develop

to install while working on the package.

Install python requirements available with PIP via:

    pip install -r requirements.txt

## Tools / dependencies

- Our image models are built around the [GalSim](https://github.com/GalSim-developers/GalSim/wiki) image simulation framework.  
- For parameter inference, we use [emcee](http://dan.iel.fm/emcee/current/).
- The sources (and source groups) we extract from raw imaging are stored in [HDF5](http://www.hdfgroup.org/HDF5/) file formats, with a custom grouping.
- For part of the results visualization we use [corner](https://github.com/dfm/corner.py).

## Versions

Branch 0.1 contains the specific code used for the [first arXiv draft](https://arxiv.org/abs/2309.10321v1) of "Markov Chain Monte Carlo for Bayesian Parametric Galaxy Modeling in LSST", submitted 19 September 2023.
