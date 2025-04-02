import numpy as np
import pickle

from scipy.ndimage import gaussian_filter, correlate
from scipy.stats import norm

import galsim


class EmptyDetectionCorrection(object):
    '''
    Applies no correction for all parameters (for any given parameterization).
    '''
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return 0.0


class IsolatedFootprintDetectionCorrection(object):
    def __init__(self, args=None):
        # Flux bins in nJy
        self.flux_bins_upper = np.array([11.501688903808594, 12.559800148010254, 13.681500434875488, 14.833292015075685,
                                         16.035306533813475, 17.317237075805664, 18.807233276367192, 20.354932098388673,
                                         21.34309158325195, 22.39406440734863, 23.51925964355469, 24.744515380859376,
                                         25.97533576965332, 27.36300880432129, 28.78671936035156, 30.364900588989258,
                                         32.0707926940918, 33.965559082031255, 36.05099868774414, 38.34932952880859,
                                         40.78902130126952, 43.48423324584961, 46.43856353759766, 49.87947631835938,
                                         53.82615737915039, 58.12320022583007, 63.091081085205076, 68.56482025146485,
                                         75.52024200439452, 83.86514312744141, 93.90503845214847, 106.23336151123047,
                                         121.44884002685538, 142.55807312011717, 172.03124389648437, 190.79404418945313,
                                         213.35159912109356, 240.5479565429689, 277.69400268554705, 301.78500366210955,
                                         327.47181152343745, 357.847478027344, 391.83363281249956, 433.07816406250004,
                                         485.8312133789061, 549.661381835937, 636.5133251953132, 744.6211181640588,
                                         895.7324267578117, 1100.019970703126, 1272.3333769531168, 1543.3866142578129,
                                         1915.0106503906247, 2469.883574218744, 3371.430322265627])
        # Convert from nJy to instrumental flux, using calibrations from DC2 study
        self.flux_bins_upper *= 0.016909286233862435
        self.flux_bins_upper += -5.068159542178663

        # Fraction of galaxies in each flux bin that pass all isolated footprint criteria.
        # This is a lower bound on the probability of having sufficient flux to be detected.
        detected_frac = np.array([0.0003372251362058207, 0.00037547404629017985, 0.0005079345423100115, 0.0006210550298602316,
                                  0.0005600530089216412, 0.0008342297325762504, 0.0008614031214394953, 0.0011268275043950433,
                                  0.0014112881809709061, 0.0015104824193163295, 0.001960806769950181, 0.002309168336781974,
                                  0.002695871690461189, 0.00317543145690176, 0.003634532196722487, 0.004748572124815331,
                                  0.0059132236781954265, 0.0071353438158466296, 0.008918355074486569, 0.011448698388200602,
                                  0.014379613392696309, 0.018704956093659967, 0.022933745224959556, 0.02889452953371621,
                                  0.03406006892998796, 0.040891280932161295, 0.04882403219677775, 0.05845053244240347,
                                  0.068207230156956, 0.0751866935659933, 0.08923380329579565, 0.09963813495545171,
                                  0.11319181102556908, 0.12142464551698011, 0.12200590857902849, 0.1125162432579852,
                                  0.10431426426867411, 0.09230942159646977, 0.07847701804168496, 0.07113499862810062,
                                  0.06702490778582709, 0.06152659271435398, 0.055963579124216074, 0.051690745880010235,
                                  0.04789545545148102, 0.04428164860847402, 0.04164096032168137, 0.03796172513326467,
                                  0.03538378838390968, 0.03141789024590937, 0.029514310011440486, 0.027284080780688627,
                                  0.02516607232110521, 0.02213634696054955, 0.019955442147266647])
        self.neg_log_detected_frac = -np.log(detected_frac)

        # 5sigma threshold for pixels in PSF-convolved image, used by the Pipelines for detection,
        # computed for each patch.
        thresholds_by_patch = [0.040795198211916954, 0.03894099770978275, 0.040699778571068365, 0.04047920476782825,
                               0.03378751619049805, 0.035192559133064356, 0.0382360261578627, 0.0381832775816528,
                               0.03989896136350646, 0.03900023668869419, 0.0386061868300802, 0.03690437667505849,
                               0.04147436027523751, 0.04005326663598364, 0.042107574476070775, 0.04074763029367439,
                               0.03921743282012365, 0.038615287184084954, 0.03847273893418333, 0.037004542009712676,
                               0.037971055601106585, 0.04176647592602702, 0.04095110199190659, 0.04224745231486727,
                               0.040299049128291194, 0.038829676768793425, 0.03990485566986899, 0.037204894926686966,
                               0.04209777341335332, 0.04040400351169758, 0.04000221861804741, 0.03941965441015578,
                               0.041247331708626864, 0.03781332186138613, 0.04034548992108185, 0.03976765506931181,
                               0.04311363547227075, 0.03967363530111516, 0.036472393242932784, 0.038264460037286445,
                               0.03981438790230997, 0.03833696702884804, 0.03999396075326277, 0.03952052867299043,
                               0.039263852859281, 0.03860626653236923, 0.03884589580438467, 0.03894999190591754,
                               0.04010447402371439]
        self.threshold = thresholds_by_patch[args.patch]

        # Values from psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()
        # in the Pipelines, computed for each patch.
        psf_sigma_pixels_by_patch = [1.729669842605186, 1.7181720639972147, 1.725759208900632, 1.7085647889876263,
                                     1.7232989543966706, 1.7021791104147888, 1.704162454575403, 1.7210361856157548,
                                     1.727087891347213, 1.7102169731522257, 1.7185169676055985, 1.7185975956446722,
                                     1.7223123515510796, 1.7040338967910533, 1.7095150709102669, 1.7318705028867436,
                                     1.718679753241698, 1.7167431438866585, 1.704677797405927, 1.7124607579734512,
                                     1.7009015072485478, 1.7147158544783232, 1.7154766130823078, 1.6950429561091915,
                                     1.711509105312507, 1.710814740516908, 1.681559761863998, 1.6972334159250289,
                                     1.7139713913340078, 1.7114053063370693, 1.7161452776596835, 1.6944592103259901,
                                     1.6957735850594704, 1.7111481312255186, 1.7028499204816483, 1.7225364438595572,
                                     1.7136327767087582, 1.7103210875509547, 1.6895746981611557, 1.7174042752899277,
                                     1.7050120781898868, 1.7180974888727945, 1.7194154878871428, 1.7180596364640004,
                                     1.7305753239535984, 1.7196090126232195, 1.7019594518653811, 1.7135161695595709,
                                     1.7171457989216676]
        self.psf_sigma_pixels = psf_sigma_pixels_by_patch[args.patch]

        self.valid_pixels = None
        self.n_valid_pixels = 0
        # The Pipelines convolve the image with a Gaussian approximation to the PSF
        self.psf = self.approximate_gaussian_psf(self.psf_sigma_pixels)
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

    def evaluate(self, model_image, roaster):
        valid_pixels = (roaster.variance > 0)
        if roaster.mask is not None:
            valid_pixels &= roaster.mask.astype(bool)
        if np.sum(valid_pixels) == 0:
            return 0.0

        # Compute a PSF-convolved image, to sharpen PSF-like signals
        convolved_image = correlate(model_image, self.psf,
                                    mode='constant', cval=0)

        # Compute the variance of the PSF-convolved image
        psf_squared = self.psf**2
        convolved_variance = correlate(roaster.variance, psf_squared,
                                        mode='constant', cval=0)

        # Flatten everything
        valid_pixels = valid_pixels.flatten()
        convolved_image = convolved_image.flatten()
        convolved_variance = convolved_variance.flatten()

        # Find the highest point-source S/N pixel in the image
        n_sigma_above_threshold = (convolved_image - self.threshold) / np.sqrt(convolved_variance)
        idx = np.argmax(n_sigma_above_threshold[valid_pixels])

        # Find the Gaussian probability of creating a random fluctuation with at least this much S/N
        max_prob = norm.sf(self.threshold, convolved_image[valid_pixels][idx],
            np.sqrt(convolved_variance[valid_pixels][idx]))

        neg_log_prob = -np.log(max_prob)

        return neg_log_prob

    def find_neg_log_detected_frac(self, flux):
        idx = np.digitize(flux, self.flux_bins_upper, right=False)
        idx = min(idx, len(self.neg_log_detected_frac) - 1)
        return self.neg_log_detected_frac[idx]

    def __call__(self, model_image, roaster):
        # Convert model_image into a numpy array
        if model_image is None:
            # Ensure the log-posterior is just the log-prior
            return 0.0
        if isinstance(model_image, galsim.Image):
            model_image = model_image.array
        elif not isinstance(model_image, np.ndarray):
            model_image = np.array(model_image)

        # We are interested in the probability that an object with given true
        # parameters would be observed by the detection algorithm.
        # The detection-corrected likelihood is the standard likelihood
        # divided by the detection probability, so the corrected log-likelihood
        # is the standard log-likelihood minus the log-probability of detection.
        # The value we return will be added to the log-likelihood,
        # so we return the negative log-probability of detection.

        # First we estimate the probability that an object with given true
        # parameters will produce a PSF-convolved signal bright enough to
        # pass the detection threshold somewhere in the exposure.
        neg_log_prob = self.evaluate(model_image, roaster)
        
        # We use the overall detection fraction
        # (which includes other criteria than just having sufficiently bright pixels)
        # as a lower bound on the bright-enough-pixel detection probability,
        # in case the preceding computation runs into trouble
        # ("trouble" might mean numerical issues from having to compute the
        # probability before finding its log, or it might mean reaching a
        # pathological corner of parameter space where the approximations we've
        # made here break down).
        # Assume a single galsim_galaxy.GalsimGalaxyModel source.
        model = roaster.src_models[0]
        flux = model.params.flux[0]
        neg_log_detected_frac = self.find_neg_log_detected_frac(flux)

        neg_log_prob = min(neg_log_prob, neg_log_detected_frac)
        
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
