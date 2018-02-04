#!/bin/bash

# This bash script is meant to be run from your local machine and will setup a
# python virtual environment on a remote LC machine (specifically cab).

###############################################################################
# USER INPUT (edit values freely, variables must be kept the same)
###############################################################################

# directory where downloaded packages will be saved
packdir="packages"
# a file containing the packages and versions to download
requirementfile="requirements.txt"
# JIF directory on LC
lcjifdir="/g/g20/mdschnei/ldrd2016/JIF"
# LC user account
#lcaccount="dawson29"
lcaccount="mdschnei"
galsim_version=1.4

###############################################################################
# Script (should be no need to edit)
###############################################################################

# Delete possible existing packdir to avoid installing old packages
echo "Remove any existing package directory."
# Give the user the option to prevent themselves from doing something stupid
# like setting packdir="/"
for i in {1..3}; do printf '\7'; done
echo "Are you sure you want to delete:"
echo $packdir
read -p  "and all subfolders/files? y/n " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    rm -r $packdir
else
    echo "Exiting. Please check packdir definition in setup_env_cab.sh script."
    exit 1
fi

# Download the pip package files, trusing the http site since https 
# certificates are blocked by the LLNL firewall
echo "Downloading the packages in $requirementfile."
pip download --index-url=http://pypi.python.org/simple/ --trusted-host pypi.python.org --no-binary :all: -d $packdir -r $requirementfile
# Download the TMV library
echo "Downloading the TMV library"
wget -P $packdir https://github.com/rmjarvis/tmv/archive/master.zip
# Download the boost library
# We have to install our own version of boost to make sure we link with the Python 
# packages in our custom virtual environment.
echo "Downloading the Boost library"
wget -P $packdir https://dl.bintray.com/boostorg/release/1.64.0/source/boost_1_64_0.tar.gz
# Download GalSim
echo "Downloading GalSim version "$galsim_version
wget -P $packdir https://github.com/GalSim-developers/GalSim/archive/releases/${galsim_version}.zip
# Tar up the package directory for easy transport to cab.
tar -zcvf $packdir.tar.gz $packdir

# scp the requirements and packages to the JIF directory on LC
for i in {1..3}; do printf '\7'; done
echo "Enter your LC password to scp the tarball package to $lcjifdir/env/"
scp $requirementfile $packdir.tar.gz $lcaccount@oslic.llnl.gov:$lcjifdir/env/.

# A function to run the setup on the remote LC machine
# Arguments are
# $1 = $lcjifdir
# $2 = $packdir
# $3 = $requirementfile
setupLC()
{
    # Untar the package files on LC
    tar -xvzf $1/env/$2.tar.gz -C $1/env/
    cd $1/env/$2
    # Specify the C++ compiler
    # export CC=/usr/local/tools/mvapich2-gnu-2.2/bin/mpicc
    export CC=/usr/bin/cc
    # Install the TMV library
    rm -r tmv-master
    unzip master.zip
    cd tmv-master
    echo "Installing TMV to:"
    echo $HOME/local
    scons install PREFIX=$HOME/local
    # Create the virtual environment
    # Remove any potential existing virtualenv
    echo "Removing any potential old mbi virtualenv"
    cd $1/env
    rm -r mbi
    # Create a new mbi virtualenv using LC default site packages
    echo "Creating a new mbi virtual environment in:"
    pwd
    /usr/apps/python/bin/virtualenv --system-site-packages mbi
    # Add the h5py built with the MPI option to the python path
    echo "export PYTHONPATH=/collab/usr/gapps/python/build/spack/opt/spack/chaos_5_x86_64_ib/gcc-4.4.7/py-h5py-2.6.0-mpi-bwkorvpojthvkei3awbg2adcznv4rp4k/lib/python2.7/site-packages" >> ./mbi/bin/activate
    # Add the local c installed package library paths to the virtual env
    echo "export PATH=\$HOME/local/bin:\$PATH" >> ./mbi/bin/activate
    echo "export LD_LIBRARY_PATH=\$HOME/local/lib" >> ./mbi/bin/activate
    # Activate the virtual environment
    source mbi/bin/activate
    echo "The mbi virtual environment has been activated."
    echo "The current PATH is:"
    echo $PATH
    # Update pip to the latest version
    pip install --upgrade pip
    # List the LC default packages installed in the virtualenv.
    echo "LC installed python packages include."
    pip list
    # Install python packages in the requirements file.
    echo "Installing python packages in the virtualenv."
    pip install --no-index --find-links=$1/env/$2 -r $1/env/$3
    # Activate the virtual environment
    source $lcjifdir/env/mbi/bin/activate
    # Install Boost library
    cd $lcjifdir/env/$packdir
    rm -r boost_1_64_0
    tar -zxvf boost_1_64_0
    cd boost_1_64_0
    ./bootstrap.sh
    ./b2 --with-python --with-math link=shared
    ./b2 --prefix=$HOME/local --with-python --with-math link=shared install
    # Install GalSim
    cd $lcjifdir/env/$packdir
    rm -r GalSim-releases-$galsim_version
    unzip 1.4.zip
    cd GalSim-releases-$galsim_version
    scons TMV_DIR=$HOME/local FFTW_DIR=/usr/local/tools/fftw3-3.3.4 BOOST_DIR=$HOME/local
    scons install PREFIX=$HOME/local PYPREFIX=$lcjifdir/mbi/lib/python2.7

    # Add a note for the user
    echo "*************************************************************"
    echo "******************** IMPORTANT NOTE *************************"
    echo "Please make sure that you add \$HOME/local/bin to the _start_"
    echo "of your PATH in your local \$HOME/.profile file, e.g.:"
    echo "export PATH=~/local/bin:\$PATH"
    echo "*************************************************************"
}
for i in {1..3}; do printf '\7'; done
echo "Enter your LC password to build python packages in LC mbi virtualenv"
ssh $lcaccount@cab.llnl.gov "$(typeset -f); setupLC $lcjifdir $packdir $requirementfile"