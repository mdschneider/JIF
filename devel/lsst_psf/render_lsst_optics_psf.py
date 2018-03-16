import warnings
import numpy as np
import matplotlib.pyplot as plt
import jiffy

from make_whisker import moments

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
print "Set DZ terms"
print "================================"
j_pupil = 2
j_field = 2
a_nmrs = np.zeros_like(psf.aberrations)
a_nmrs[j_pupil, j_field] = 1.
psf.aberrations = a_nmrs

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
lam_over_diam = 600. / 8.4e9 # radians
lam_over_diam *= 206265  # Convert to arcsec
print lam_over_diam, "arcseconds"
# psf = galsim.Airy(lam_over_diam)
# psf_im = psf.drawImage(nx=64, ny=64, scale=0.1*lam_over_diam)

# psf_im = psf.get_image(scale=0.1*lam_over_diam, ngrid_x=128, ngrid_y=128,
					   # theta_x_arcmin=-1.6*60., theta_y_arcmin=0.0)

r = 1.75 * 60.
x = r * np.cos(180.*np.pi/180.)
y = r * np.sin(180.*np.pi/180.)

# psf_im = psf.get_image(scale=lam_over_diam/40, ngrid_x=1024, ngrid_y=1024,
# 					   theta_x_arcmin=x, theta_y_arcmin=y,
# 					   with_atmos=False)

psf_im = psf.get_image(scale=0.2, ngrid_x=32, ngrid_y=32,
					   theta_x_arcmin=x, theta_y_arcmin=y,
					   with_atmos=True)

Ixx,Iyy,Ixy,xbar,ybar = moments(psf_im.array)
print "Moments: ", Ixx,Iyy,Ixy,xbar,ybar
csq = np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2)
phi = 0.5 * np.arctan2(2*Ixy, Ixx-Iyy)
print "c^2, phi:", csq, phi
e1 = np.sqrt(csq) * np.cos(phi)
e2 = np.sqrt(csq) * np.sin(phi)
print "e1, e2:", e1, e2

# nx = 256
# psf_im = psf.get_image(scale=0.6/nx, ngrid_x=nx, ngrid_y=nx,
# 					   theta_x_arcmin=0., theta_y_arcmin=-1.75*60.,
# 					   with_atmos=False)

print psf_im
plt.imshow(psf_im.array)
plt.colorbar()
plt.savefig("psf_image.png")

# print "================================"
# print "PSF moments"
# print "================================"
# results = psf_im.FindAdaptiveMom()
# print 'e1 = %.5e, e2 = %.5e, sigma = %.5f (pixels)'%(results.observed_shape.e1,
#                 results.observed_shape.e2, results.moments_sigma)
