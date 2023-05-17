import numpy as np
import pickle

from scipy.ndimage import gaussian_filter, correlate
from scipy.stats import norm

class EmptyDetectionCorrection(object):
    '''
    DetectionCorrection form for the image model parameters
    Applies no correction for all parameters (for any given parameterization).
    '''
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return 0.0

'''
Assumes that variance and mask are provided as image-sized arrays.
'''
class IsolatedFootprintDetectionCorrection(object):
    def __init__(self, args=None):
        # Flux bins in nJy
        self.flux_bins_upper = np.array([11.527841402053834, 12.589147148132325, 13.72355423927307, 14.884612380981446,
                                         16.09600067138672, 17.40477798461914, 18.91361146545411, 20.47610092163086,
                                         21.477143478393554, 22.54180450439453, 23.71131935119629, 24.93340446472168,
                                         26.175127639770505, 27.591835784912114, 29.061195220947265, 30.638700485229492,
                                         32.41033309936523, 34.34035903930664, 36.5208546447754, 38.827496643066404,
                                         41.36696090698241, 44.14577476501465, 47.194010925292964, 50.77131050109864,
                                         54.83609207153321, 59.29889907836913, 64.48213623046875, 70.2841764831543,
                                         77.55862945556642, 86.1080140686035, 96.86723937988296, 109.97297958374023,
                                         126.20048187255851, 149.33343566894533])
        # Convert from nJy to instrumental flux, using calibrations from DC2 study
        self.flux_bins_upper *= 0.017100155374837712
        self.flux_bins_upper += -1.073090269533189
        # Fraction of galaxies in each flux bin that pass all isolated footprint criteria.
        # This is a lower bound on the probability of having sufficient flux to be detected.
        detected_frac = np.array([0.0007389110182013105, 0.0007851891802036702, 0.0010106735953169163, 0.001017000051464549,
                                   0.0010235495219841231, 0.0012299710175904973, 0.0012792173299028132, 0.001454382992592537,
                                   0.0018617649431903996, 0.0021245124416635966, 0.0023617543592913386, 0.002602798388947123,
                                   0.0030302305480409, 0.0031886509667079896, 0.0038042989243343983, 0.005019089345324304,
                                   0.00641525648926068, 0.007270910714866668, 0.009818681777706076, 0.011819661417832326,
                                   0.014812218940266967, 0.020211556269328333, 0.023801183595974985, 0.029870365615255863,
                                   0.03627161780031299, 0.04188025740219244, 0.05101675683678421, 0.06081006509381948,
                                   0.07036615146818923, 0.07772014621795387, 0.09189516129801277, 0.10343125983915473,
                                   0.11477903043863565, 0.12468318654152415])
        self.neg_log_detected_frac = -np.log(detected_frac)

        # 5sigma threshold for pixels in PSF-convolved image, used by the Pipelines for detection,
        # computed for each patch.
        thresholds_by_patch = [0.060315693663387994, 0.06057508708859115, 0.05966113235785343, 0.054778438708690175,
                            0.057934465741441954, 0.056547256212552445, 0.057913419725767294, 0.060809309480234226,
                            0.0610707302688821, 0.05876922045028114, 0.060267535120656045, 0.05853281082991228,
                            0.05627093408164527, 0.05730620073879363, 0.061120725811735346, 0.06047271364193935,
                            0.06059025355908097, 0.05797534777633805, 0.056354882949623604, 0.059018083631529,
                            0.05398557570459672, 0.05949826265911269, 0.05963283362922741, 0.05830351707767113,
                            0.057894146554702895, 0.06191057924206567, 0.05900712695208939, 0.05762208459963227,
                            0.05931928895618016, 0.06407642908652113, 0.059373885352736196, 0.061308561066120155,
                            0.06475719456810601, 0.05708688979382738, 0.062232389948980685, 0.05771295204512132,
                            0.061731789266247075, 0.05996351183563551, 0.05890244566764345, 0.0640354145184056,
                            0.058410014664573406, 0.05909394723030767, 0.06099275740551516, 0.06512571891684148,
                            0.059138206220482374, 0.055702019809542785, 0.05474982905947143, 0.055950217978148906,
                            0.06185339615092024]
        self.threshold = thresholds_by_patch[args.patch]

        # Values from psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()
        # in the Pipelines, computed for each patch.
        psf_sigma_pixels_by_patch = [1.7271920588285004, 1.7180812514058172, 1.7216984769937729, 1.7076190024129758,
            1.7273626016811945, 1.7020369890356006, 1.70419799673249, 1.721129361686903,
            1.7246199325247937, 1.7080286187427656, 1.715198599799439, 1.7186398477552562,
            1.7192051124187757, 1.7043016242824174, 1.7077635251303207, 1.7302093666340908,
            1.7155592741012253, 1.7151996885715248, 1.6990983937395165, 1.7107961376776806,
            1.7009109711883887, 1.7131169257949153, 1.7083676133722268, 1.6905067834572662,
            1.702420607679118, 1.7054037429680624, 1.6817549821779594, 1.694096113975984,
            1.713643363908017, 1.7116248817906614, 1.715864075048302, 1.6914826322040306,
            1.695105566091944, 1.7066003566720391, 1.7024507637350015, 1.72278740771248,
            1.7140519878831586, 1.710081173517995, 1.6881656673877286, 1.7175166345909458,
            1.7048513882471386, 1.7147794456988585, 1.7187243621342134, 1.7154865351343598,
            1.730737144627177, 1.7193849108343384, 1.7000010769682796, 1.716646995554665,
            1.7139323098810513]
        self.psf_sigma_pixels = psf_sigma_pixels_by_patch[args.patch]

        self.valid_pixels = None
        self.n_valid_pixels = 0
        self.psf = None
        self.convolved_variance = None

    # n_sigma_for_kernel=7.0 comes from the default value of
    # sourceDetectionTask.config.nSigmaForKernel in the Pipelines
    def approximate_gaussian_psf(self, sigma_pixels, n_sigma_for_kernel=7.0):
        # This is the same procedure used by sourceDetectionTask.calculateKernelSize
        # in the Pipelines
        width = (int(sigma_pixels * n_sigma_for_kernel + 0.5) // 2) * 2 + 1
        center = (width - 1) // 2

        tophat = np.zeros((width, width))
        tophat[center, center] = 1

        gaussian_psf_image = gaussian_filter(tophat, sigma=sigma_pixels, order=0,
                                       mode='constant', cval=0.0, truncate=n_sigma_for_kernel+2)

        return gaussian_psf_image

    def set_valid_pixels(self, variance, mask):
        self.valid_pixels = mask.astype(bool)
        self.valid_pixels &= (variance != 0)
        self.valid_pixels = self.valid_pixels.flatten()

        self.n_valid_pixels = np.sum(self.valid_pixels)

    def set_convolved_variance(self, variance, psf):
        psf_squared = psf**2

        self.convolved_variance = correlate(variance, psf_squared,
                                            mode='reflect')
        self.convolved_variance = self.convolved_variance.flatten()
    
    def evaluate(self, model_image, psf):
        convolved_image = correlate(model_image, psf,
                                    mode='constant', cval=0)
        convolved_image = convolved_image.flatten()

        n_sigma_above_threshold = (convolved_image - self.threshold) / np.sqrt(self.convolved_variance)
        idx = np.argmax(n_sigma_above_threshold[self.valid_pixels])
        max_prob = norm.sf(self.threshold, convolved_image[self.valid_pixels][idx],
            np.sqrt(self.convolved_variance[self.valid_pixels][idx]))

        neg_log_prob = -np.log(max_prob)

        return neg_log_prob

    def find_neg_log_detected_frac(self, flux):
        idx = np.digitize(flux, self.flux_bins_upper, right=False)
        idx = min(idx, len(self.neg_log_detected_frac) - 1)
        return self.neg_log_detected_frac[idx]

    # Note: Here we assume variance is known in a given footprint,
    # though some analyses have treated this probabilistically
    # (and therefore as something to sample in each MCMC step).
    def __call__(self, model_image, src_models, variance, mask):
        update_valid_pixels = (self.valid_pixels is None)
        if update_valid_pixels:
            self.set_valid_pixels(variance, mask)
        # If the entire image is masked, the posterior should equal the prior.
        if self.n_valid_pixels == 0:
            return 0

        update_convolved_variance = False
        # Assume a single galsim_galaxy.GalsimGalaxyModel source
        model = src_models[0]
        if self.convolved_variance is None:
            update_convolved_variance = True
            # The Pipelines convolve the image with a Gaussian approximation to the PSF
            self.psf = self.approximate_gaussian_psf(self.psf_sigma_pixels)
        if update_convolved_variance:
            self.set_convolved_variance(variance, self.psf)

        neg_log_prob = self.evaluate(model_image, self.psf)
        # Use overall detection fraction (which includes other criteria than just having sufficiently bright pixels)
        # as lower bound on the bright-enough-pixel detection probability,
        # in case the preceding computation runs into trouble ("trouble" might mean numerical issues from having
        # to compute the probability before finding its log, or it might mean reaching a pathological corner of
        # parameter space where the approximations we've made here break down).
        flux = model.params.flux[0]
        neg_log_prob = min(neg_log_prob, self.find_neg_log_detected_frac(flux))
        return neg_log_prob

detection_corrections = {None: False,
          'Empty': EmptyDetectionCorrection,
          'EmptyDetectionCorrection': EmptyDetectionCorrection,
          'IsolatedFootprintDetectionCorrection': IsolatedFootprintDetectionCorrection}

def initialize_detection_correction(form=None, module=None, args=None, **kwargs):
    if module is None:
        # form should be one of the names of detection_corrections in this file
        detection_correction = detection_corrections[form]
    else:
        module = __import__(module)
        # detection_correction_form should be the name of a class in module
        detection_correction = getattr(module, form)

    # Initialize an instance of the prior with the given arguments
    return detection_correction(args, **kwargs)
