import numpy as np
import matplotlib.pyplot as plt
import jiffy

#
# Initialize the PSF model object
#
psf = jiffy.GalsimPSFLSST()

dat = np.log10(np.abs(psf.aberrations))
ndx = np.where(psf.aberrations < 0.)
dat[ndx] *= -1.

plt.figure(figsize=(8,12))
plt.imshow(dat, cmap=plt.cm.PuOr)
plt.colorbar()
plt.xlabel(r"Field Noll index", fontsize=18)
plt.ylabel(r"Pupil Noll index", fontsize=18)
plt.savefig("dz_array.png")