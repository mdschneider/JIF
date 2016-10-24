import unittest
import numpy as np
import galsim
import jif

class TestGalSimGalaxyModel(unittest.TestCase):

	def test_init_fails_on_bad_arguments(self):
		with self.assertRaises(AssertionError):
			jif.GalSimGalaxyModel(galaxy_model="UnimplementedModel")
		with self.assertRaises(AssertionError):
			jif.GalSimGalaxyModel(telescope_model="FakeTelescope")

	def test_init_telescope(self):
		gg = jif.GalSimGalaxyModel(telescope_model="LSST")
		self.assertAlmostEqual(gg.pixel_scale_arcsec, 0.2)
		gg = jif.GalSimGalaxyModel(telescope_model="WFIRST")
		self.assertAlmostEqual(gg.pixel_scale_arcsec, 0.11)

	def test_init_model_params(self):
		gg = jif.GalSimGalaxyModel(active_parameters=['e', 'psf_fwhm'])
		self.assertEqual(gg.active_parameters_galaxy, ['e'])
		self.assertEqual(gg.psf_paramnames, ['psf_fwhm'])
		self.assertEqual(gg.paramnames, ['e', 'psf_fwhm'])
		self.assertEqual(gg.n_params, 2)
		self.assertEqual(gg.n_psf_params, 1)

	def test_init_psf_fails_with_incompatible_args(self):
		with self.assertRaises(AttributeError):
			psf = galsim.Kolmogorov(fwhm=0.6).drawImage()
			jif.GalSimGalaxyModel(galaxy_model='star',
				psf_model=psf)
		with self.assertRaises(ValueError):
			## Don't accept a simple numpy array as a PSF - we need a galsim.Image instance
			jif.GalSimGalaxyModel(psf_model=np.zeros((8,8)))
		psf_model = jif.PSFModel(telescope="WFIRST",
			active_parameters=['psf_fwhm', 'psf_e'])
		with self.assertRaises(AssertionError):
			jif.GalSimGalaxyModel(psf_model=psf_model, telescope_model="LSST")
		with self.assertRaises(AssertionError):
			jif.GalSimGalaxyModel(psf_model=psf_model, 
				active_parameters=['e', 'psf_fwhm'])

	def test_init_psf(self):
		## InterpolatedImage PSF
		psf = galsim.Kolmogorov(fwhm=0.6).drawImage()
		gg = jif.GalSimGalaxyModel(psf_model=psf)
		self.assertEqual(gg.psf_model_type, 'InterpolatedImage')
		#
		## PSFModel PSF
		psf = jif.PSFModel(telescope="LSST", active_parameters=[])
		gg = jif.GalSimGalaxyModel(psf_model=psf)
		self.assertEqual(gg.psf_model_type, "PSFModel class")
		self.assertEqual(psf, gg.psf_model)
		#
		## PSF model to be instantiated internally
		gg = jif.GalSimGalaxyModel(psf_model="model")
		self.assertEqual(gg.psf_model_type, "PSFModel class")


if __name__ == "__main__":
	unittest.main()