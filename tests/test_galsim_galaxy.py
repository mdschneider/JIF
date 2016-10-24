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

	def test_set_param_by_name(self):
		gg = jif.GalSimGalaxyModel()
		## Standard galaxy parameter
		gg.set_param_by_name('e', 0.4)
		self.assertAlmostEqual(gg.get_param_by_name('e'), 0.4)
		## Galaxy ellipticity angle - modulo pi
		gg.set_param_by_name('beta', 0.1 + 3 * np.pi)
		self.assertAlmostEqual(gg.get_param_by_name('beta'), 0.1)
		## PSF parameter
		gg.set_param_by_name('psf_e', 0.12)
		self.assertAlmostEqual(gg.get_param_by_name('psf_e'), 0.12)
		## Set a PSF parameter when using InterpolatedImage PSF
		psf = galsim.Kolmogorov(fwhm=0.6).drawImage()
		gg = jif.GalSimGalaxyModel(psf_model=psf)
		with self.assertRaises(ValueError):
			gg.set_param_by_name('psf_e', 0.12)
		with self.assertRaises(ValueError):
			gg.get_param_by_name('psf_e')

	def test_set_params(self):
		gg = jif.GalSimGalaxyModel(active_parameters=['e', 'beta', 'psf_fwhm'])
		self.assertEqual(len(gg.get_params()), 3)
		p0 = [0.17, 0.1, 0.12]
		gg.set_params(p0)
		for i in xrange(len(p0)):
			self.assertAlmostEqual(gg.get_params()[i], p0[i])
		p1 = [0.17]
		with self.assertRaises(AssertionError):
			gg.set_params(p1)
		self.assertAlmostEqual(gg.get_psf_params()[0], p0[2])
		self.assertEqual(len(gg.get_psf_params()), 1)

	def test_validate_params(self):
		gg = jif.GalSimGalaxyModel(active_parameters=['e', 'beta', 'psf_fwhm'])
		valid = gg.validate_params()
		self.assertTrue(valid)
		## Set some bad galaxy model parameter
		gg.set_param_by_name('e', 2.0)
		valid = gg.validate_params()
		self.assertFalse(valid)
		gg.set_param_by_name('e', 0.1)
		## Set a bad PSF model parameter
		gg.set_param_by_name('psf_fwhm', 20.)
		valid = gg.validate_params()
		self.assertFalse(valid)

	def test_get_image(self):
		## Use all default parameters
		gg = jif.GalSimGalaxyModel()
		im = gg.get_image()
		self.assertAlmostEqual(im.scale, 0.2)
		rms = np.sqrt(np.sum(im.array.ravel()**2))
		self.assertAlmostEqual(rms, 31177.847656250)

if __name__ == "__main__":
	unittest.main()
