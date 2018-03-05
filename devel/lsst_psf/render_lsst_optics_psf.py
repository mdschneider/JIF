import warnings
import numpy as np
import matplotlib.pyplot as plt
import jiffy

warnings.simplefilter('ignore')

print "================================"
print "Initialize PSF object"
print "================================"
psf = jiffy.GalsimPSFLSST()
psf.set_param_by_name("psf_fwhm", 0.6)
print psf
print psf.params
print np.sum(psf.aberrations, axis=1)

print "================================"
print "Get PSF model object"
print "================================"
psf_model = psf.get_model()
print psf_model

print "================================"
print "Get wavefront"
print "================================"
wf = psf.get_wavefront()
print "wf range:", np.min(wf.ravel()), np.max(wf.ravel())
plt.imshow(wf)
plt.colorbar()
plt.savefig("wf_image.png")
plt.close()

print "================================"
print "Get PSF image"
print "================================"
import galsim
lam_over_diam = 500. / 8.4e9 # radians
lam_over_diam *= 206265  # Convert to arcsec
print lam_over_diam, "arcseconds"
# psf = galsim.Airy(lam_over_diam)
# psf_im = psf.drawImage(nx=64, ny=64, scale=0.1*lam_over_diam)

# psf_im = psf.get_image(scale=0.1*lam_over_diam, ngrid_x=128, ngrid_y=128,
					   # theta_x_arcmin=-1.6*60., theta_y_arcmin=0.0)

psf_im = psf.get_image(scale=lam_over_diam/40, ngrid_x=1024, ngrid_y=1024,
					   theta_x_arcmin=-1.75*60., theta_y_arcmin=0.,
					   with_atmos=False)

# nx = 256
# psf_im = psf.get_image(scale=0.6/nx, ngrid_x=nx, ngrid_y=nx,
# 					   theta_x_arcmin=0., theta_y_arcmin=-1.75*60.,
# 					   with_atmos=False)

print psf_im
plt.imshow(psf_im.array)
plt.colorbar()
plt.savefig("psf_image.png")

print "================================"
print "PSF moments"
print "================================"
results = psf_im.FindAdaptiveMom()
print 'e1 = %.5e, e2 = %.5e, sigma = %.5f (pixels)'%(results.observed_shape.e1,
                results.observed_shape.e2, results.moments_sigma)
