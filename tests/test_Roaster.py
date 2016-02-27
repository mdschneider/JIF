import sys
import os
import numpy as np
import jif

class RoasterArgs(object):
    """
    Mimic the arguments to the JIF Roaster.py script
    """
    def __init__(self, infile=None, model_paramnames=[]):
        self.infiles = [infile]
        self.segment_numbers = None # Load all segments
        self.outfile = 'tmp.txt'
        self.galaxy_model_type = 'Spergel'
        self.data_format = 'jif_segment'
        self.model_params = model_paramnames
        self.telescope = None # Load all telescopes
        self.seed = 1
        self.nsamples = 10
        self.nwalkers = 8
        self.nburn = 1
        self.nthreads = 1
        self.quiet = True
        self.debug = False

model_paramnames = ['hlr', 'e', 'beta', 'mag_sed1', 'psf_fwhm', 'psf_e']
lnprior_omega = jif.DefaultPriorSpergel()
lnprior_Pi = jif.DefaultPriorPSF()
roaster = jif.Roaster(lnprior_omega=lnprior_omega,
                      lnprior_Pi=lnprior_Pi,
                      galaxy_model_type='Spergel',
                      model_paramnames=model_paramnames,
                      achromatic_galaxy=False)

### 1 galaxy, 1 telescope, 1 filter, 2 epochs
infile = 'roaster_files/planter_images.h5'
roaster.Load(infile)

def test_roaster_get_params():
    roaster.set_param_by_name('hlr', 1.0)
    roaster.set_param_by_name('e', 0.1)
    roaster.set_param_by_name('beta', 1.2)
    roaster.set_param_by_name('mag_sed1', 20.0)
    roaster.set_param_by_name('psf_fwhm', 0.8)
    roaster.set_param_by_name('psf_e', 0.01)
    #
    p = roaster.get_params()
    assert np.isclose(p[0], 1.0)
    assert np.isclose(p[1], 0.1)
    assert np.isclose(p[2], 1.2)
    assert np.isclose(p[3], 20.0)
    assert np.isclose(p[4], 0.8)
    assert np.isclose(p[5], 0.01)
    assert np.isclose(p[6], 0.8)
    assert np.isclose(p[7], 0.01)
    return None
