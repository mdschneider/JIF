# Setting Up custom GalSim with LSST DM-stack

Since we are working with a number of custom GalSim features we need to install and use our own GalSim rather than the version installed with the LSST Stack.

Note that this assumes that the [DM Stack has been built from source](https://confluence.lsstcorp.org/display/LSWUG/Building+the+LSST+Stack+from+Source) and installed at `$HOME/lsst`.

## Preping the DM stack environment

Start by installing the base version of GalSim packaged with the DM Stack. This will ensure that all of the dependencies are installed (everything but tmv is probably already installed).

```
cd $HOME/lsst/
source loadLSST.bash
eups distrib install -t v10_1 GalSim
setup GalSim
```

The `setup GalSim` command will just gurantee that all of the dependencies will be loaded for our following custom installation.

## Getting the development version of GalSim

Make a development directory, which is where eups will assume by default custom packages are located.

```
mkdir $HOME/lsst/dev
cd $HOME/lsst/dev
```

clone the galsim repo to this development location (prefered)

```
git clone https://github.com/GalSim-developers/GalSim.git
``` 

Checkout the master (not the current release).

```
cd $HOME/lsst_dev/GalSim
git checkout master
```

Build with scons: 

```
scons
```

Install with scons:

```
scons install
```

Declare this versions existence with eups.

```
eups declare GalSim v1_3+1 -r $HOME/lsst/dev/GalSim -c
```

The -r gives the path to the built distribution and the -c sets this version as the default (so that in the future you can just run `setup GalSim` without specifying a version (e.g. `setup GalSim v1_3+1`)).

### (optional) If installing GalSim outside of $HOME/lsst/dev

The eups setup table will by default look for an eups table in `$HOME/lsst/dev/GalSim/ups` so if you installed GalSim somewhere else it is a good idea to copy a table to this location. Note that it will already be there if you followed the instructions above

Create the eups table file for this version. The following just copies the table file from the LSST distributed version of GalSim. This should be fine since GalSim has some pretty stable dependencies (numpy, python, pyfits, boost, fftw, tmv, and scons). Note that this assumes that you have built the DM stack in `$HOME/lsst`.

```
mkdir -p $HOME/lsst/dev/GalSim/ups
```

Copy the eups table file from the regular 

```
cp $HOME/lsst/DarwinX86/GalSim/1.2/ups/GalSim.table $HOME/lsst/dev/GalSim/ups/GalSim.table
```

## Loading GalSim

Once this is done then GalSim can be setup like any other LSST DM package, e.g.:

```
setup GalSim
```

after running `source loadLSST.bash` from the `$HOME/lsst` directory.

### Check GalSim
It is probably worthwhile to verify that the intended galsim is being run, e.g.:

```
python
import galsim
galsim.__version__
```

It might also be worthwhile running the galsim tests, see Section 3 of the [Installation Instructions](https://github.com/GalSim-developers/GalSim/blob/releases/1.2/INSTALL.md).