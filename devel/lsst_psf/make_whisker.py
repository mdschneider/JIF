import numpy as np
import matplotlib.pyplot as plt
import jiffy

fov_arcmin = 1.75*60.
pixel_scale_arcmin = 0.2

#
# Setup coordinate arrays over the focal plane
#
x = np.linspace(-1., 1., 30)
xx, yy = np.meshgrid(x, x)
r = np.sqrt(xx**2 + yy**2)
ndx = np.where(r <= 1.)
xx = xx[ndx]
yy = yy[ndx]

#
# Initialize the PSF model object
#
psf = jiffy.GalsimPSFLSST()
psf.set_param_by_name("psf_fwhm", 0.6)

#
# Set all but one aberration to zero
#
a_nmrs = np.zeros_like(psf.aberrations)
a_nmrs[5,5] = psf.aberrations[5,5]
a_nmrs[6,6] = psf.aberrations[6,6]
psf.aberrations = a_nmrs

e1 = []
e2 = []
for x, y in zip(xx, yy):
	# print x, y
	psf_im = psf.get_image(scale=pixel_scale_arcmin,
						   ngrid_x=32, ngrid_y=32,
						   theta_x_arcmin=x*fov_arcmin,
						   theta_y_arcmin=y*fov_arcmin,
						   with_atmos=True)
	results = psf_im.FindAdaptiveMom()

	# eta = results.observed_shape.eta
	# g1 = results.observed_shape.g1
	# g2 = results.observed_shape.g2
	# print g1, g2
	# print results.moments_sigma
	# print "eta:", results.observed_shape.eta
	e1.append(results.observed_shape.e1)
	e2.append(results.observed_shape.e2)

print e1
print e2
print np.sqrt(np.array(e1)**2 + np.array(e2)**2)

fig, axs = plt.subplots(1, 1)
# plt.title('Arrows scale with plot width, not view')
Q = axs.quiver(xx, yy, e1, e2, headaxislength=0, headlength=0)
# qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                    coordinates='figure')
axs.axis('equal')
plt.xlim(-1., 1.)
plt.ylim(-1., 1.)
plt.show()