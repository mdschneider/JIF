import unittest
import numpy as np
import jif.parameters as jifparams


class TestParameters(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestParameters, self).__init__(*args, **kwargs)
        ### Initialize a dictionary of model parameters for each supported model type
        self.params = {}
        for galaxy_model in jifparams.k_galparams_types:
            self.params[galaxy_model] = np.core.records.array(
                jifparams.k_galparams_defaults[galaxy_model],
                dtype=jifparams.k_galparams_types[galaxy_model])
        ### Some representative model parameter names
        self.model_paramnames = ['e', 'beta', 'dx', 'dy', 'psf_fwhm', 'psf_e', 'psf_beta']

    def test_select_psf_paramnames(self):
        p_psf = jifparams.select_psf_paramnames(self.model_paramnames)
        self.assertEqual(p_psf, ['psf_fwhm', 'psf_e', 'psf_beta'])

    def test_select_galaxy_paramnames(self):
        p_gal = jifparams.select_galaxy_paramnames(self.model_paramnames)
        self.assertEqual(p_gal, ['e', 'beta', 'dx', 'dy'])

        
if __name__ == '__main__':
    unittest.main()