import numpy as np
import matplotlib.pyplot as plt
import jiffy

fov_arcmin = 1.75*60.
pixel_scale_arcmin = 0.2


def moments(image):
    x, y = np.mgrid[:image.shape[0],:image.shape[1]]
    M00 = np.sum(image)
    M10 = np.sum(x*image)
    M01 = np.sum(y*image)
    xbar = M10 / M00
    ybar = M01 / M00
    Ixx = np.sum(x**2*image) - xbar * M10
    Iyy = np.sum(y**2*image) - ybar * M01
    Ixy = np.sum(x*y*image) - xbar * M01
    return Ixx, Iyy, Ixy


def get_whisker_map(psf, xx, yy):
    e1 = []
    e2 = []
    i = 0
    for x, y in zip(xx, yy):
        i += 1
        if np.mod(i, 50) == 0:
            print "=== {:d}/{:d} ===".format(i, len(xx))
        psf_im = psf.get_image(scale=pixel_scale_arcmin,
                               ngrid_x=32, ngrid_y=32,
                               theta_x_arcmin=x*fov_arcmin,
                               theta_y_arcmin=y*fov_arcmin,
                               with_atmos=True)

        Ixx,Iyy,Ixy = moments(psf_im.array)
        csq = np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2)
        phi = 0.5 * np.arctan2(2*Ixy, Ixx-Iyy)
        e1.append(np.sqrt(csq) * np.cos(phi))
        e2.append(np.sqrt(csq) * np.sin(phi))

        # results = psf_im.FindAdaptiveMom()
        # e1.append(results.observed_shape.e1)
        # e2.append(results.observed_shape.e2)
    return e1, e2


def plot_whisker(xx, yy, e1, e2, j_pupil=1, j_field=1):
    """
    Make a whisker plot

    TODO: add option to use absolute rather than relative scale for 
    whisker lengths.
    """
    fig, axs = plt.subplots(1, 1)
    # plt.title('Arrows scale with plot width, not view')
    Q = axs.quiver(xx, yy, e1, e2, headaxislength=0, headlength=0)#,
                   # units='width',
                   # scale=1./(32*pixel_scale_arcmin))
    # qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
    #                    coordinates='figure')
    axs.axis('equal')
    plt.xlim(-1., 1.)
    plt.ylim(-1., 1.)
    plt.title("Pupil: {:d}, Field: {:d}".format(j_pupil, j_field))
    # plt.show()
    plt.savefig("whisker_image_{:d}_{:d}.png".format(j_pupil, j_field))
    plt.close()

#
# Setup coordinate arrays over the focal plane
#
x = np.linspace(-1., 1., 24)
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

for j_pupil in xrange(2, psf.aberrations.shape[0]):
    for j_field in xrange(2, psf.aberrations.shape[1]):
        print "/////////// {:d}, {:d} //////////".format(j_pupil, j_field)
        # Set all but one aberration to zero
        a_nmrs = np.zeros_like(psf.aberrations)
        a_nmrs[j_pupil, j_field] = 1.
        psf.aberrations = a_nmrs

        e1, e2 = get_whisker_map(psf, xx, yy)
        plot_whisker(xx, yy, e1, e2, j_pupil, j_field)


