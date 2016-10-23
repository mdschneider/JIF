Transformation of probability distributions under shear
=======================================================

**Created on October 13, 2016**

Do the marginal posteriors for the galaxy parameters stay invariant under the action of a shear?

We have inclinations that the answer is ‘no’ if:

- The PSF has higher order moments than quadrupole (and these are not in the PSF model)
- The galaxy has isophotal twisting (i.e., a non-concentric ellipsoidal surface brightness profile)
- color gradients

If any condition above holds (and maybe other conditions we haven’t thought of) then our Reaper + Thresher algorithm may yield large shear biases because of the way we apply the shear transformation to galaxy model ellipticity parameters.

Some potential mitigations in our algorithm could include:

- PSF marginalization
- non-uniform weighting of the surface brightness profile in the pixel likelihood evaluation


## Simulation study plan

1. Reap an unsheared galaxy image
2. Shear the image
3. Reap the sheared image
4. Unshear the 2nd set of galaxy parameter samples
5. Compare the galaxy parameter posteriors from the unsheared & sheared images

Use the same noise map and PSF in the image simulations; at least initially.


## Image-based study plan

Prior to comparing probability distributions output by Reaper+Thresher, we can gain some insight by studying the (data-model) residuals as a function of applied shear. Here’s a procedure:

1. Deconvolve the PSF from the data image
2. Apply a known shear to the deconvolved data image
3. Deconvolve the sheared data image with the PSF
4. Shear the model image
5. Compute the residual and the chi-sq
6. Repeat 1-5 for different applied shears
7. Plot chi-sq vs applied shear to look for a consistent (i.e., flat) response
