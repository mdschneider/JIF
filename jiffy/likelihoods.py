import numpy as np
import galsim


class EmptyLikelihood(object):
    def __init__(self, *args, **kwargs):
        pass


    def __call__(self, *args, **kwargs):
        return 0.0


class GaussianLikelihood(object):
    def __init__(self, args=None):
        pass


    def evaluate(self, model_image, roaster):
        valid_pixels = (roaster.variance > 0)
        valid_pixels &= model_image >= roaster.bias
        if roaster.mask is not None:
            valid_pixels &= roaster.mask.astype(bool)

        if np.sum(valid_pixels) == 0:
            return 0.0

        logden = -0.5 * np.log(2 * np.pi * roaster.variance)
        loglike_normalization = np.sum(logden[valid_pixels])

        delta = roaster.data - model_image
        sum_chi_sq = np.sum(delta[valid_pixels]**2 / roaster.variance[valid_pixels])

        res = -0.5 * sum_chi_sq + loglike_normalization

        return res


    def set_variance(self, model_image, roaster):
        pass


    def __call__(self, model_image, roaster):
        # Convert model_image into a numpy array
        if model_image is None:
            # Ensure the log-posterior is just the log-prior
            return 0.0
        if isinstance(model_image, galsim.Image):
            model_image = model_image.array
        elif not isinstance(model_image, np.ndarray):
            model_image = np.array(model_image)

        variance = self.set_variance(model_image, roaster)

        res = self.evaluate(model_image, roaster)

        return res


class LinearGaussianLikelihood(GaussianLikelihood):
    def __init__(self, args=None):
        super().__init__(args)

        var_slopes_by_patch = [0.00016078367, 0.0001684142, 0.00016333732, 0.00015648307,
                                0.00014618666, 0.00013916066, 0.00013767356, 0.00016218777,
                                0.00016697544, 0.00015934753, 0.00015830783, 0.00014973426,
                                0.00014962642, 0.00014931195, 0.00017144646, 0.00017588535,
                                0.00016993104, 0.00016486642, 0.000151099, 0.00014622364,
                                0.0001521168, 0.000173647, 0.00017530046, 0.00016912306,
                                0.00017095181, 0.00016052145, 0.00014976536, 0.00014690323,
                                0.00017047825, 0.0001733831, 0.00015871016, 0.00015433389,
                                0.0001593125, 0.00015554296, 0.00014600332, 0.00017267886,
                                0.00016782251, 0.00016062784, 0.00015223109, 0.00015296412,
                                0.00015874299, 0.0001548143, 0.00017820546, 0.0001745737,
                                0.00016082652, 0.00015726773, 0.00015971293, 0.00016239811,
                                0.00015751726]
        var_intercepts_by_patch = [0.003561471, 0.0036504567, 0.0035157874, 0.00332196,
                                    0.0031327, 0.003031202, 0.0030724355, 0.0034585623,
                                    0.0036161453, 0.0034484565, 0.0034186456, 0.0032687443,
                                    0.0032660142, 0.0032581955, 0.0037175245, 0.0037625472,
                                    0.0036626891, 0.0036099278, 0.0033835461, 0.003208242,
                                    0.0032916311, 0.0038090895, 0.0038400085, 0.0037061945,
                                    0.0037036326, 0.0035637114, 0.0033395162, 0.0032172399,
                                    0.003780944, 0.003683052, 0.0034167878, 0.0034477012,
                                    0.0035252462, 0.0034816298, 0.0033078485, 0.0037422278,
                                    0.0037145305, 0.0034858226, 0.0033993863, 0.0035052283,
                                    0.0035462584, 0.0034227055, 0.003992284, 0.0038529928,
                                    0.0035346877, 0.0034960718, 0.0035807046, 0.0035883912,
                                    0.0034764956]
        self.var_slope = var_slopes_by_patch[args.patch]
        self.var_intercept = var_intercepts_by_patch[args.patch]


    def set_variance(self, model_image, roaster):
        roaster.variance = self.var_slope * model_image + self.var_intercept


likelihoods = {None: False,
    'Empty': EmptyLikelihood,
    'EmptyLikelihood': EmptyLikelihood,
    'LinearGaussian': LinearGaussianLikelihood,
    'LinearGaussianLikelihood': LinearGaussianLikelihood,
    'Gaussian': GaussianLikelihood,
    'GaussianLikelihood': GaussianLikelihood}


def initialize_likelihood(form=None, module=None, args=None, **kwargs):
    if module is None:
        # form should be one of the names of likelihoods in this file
        likelihood = likelihoods[form]
    else:
        module = __import__(module)
        likelihood = getattr(module, form)

    # Initialize an instance of the likelihood with the given arguments
    return likelihood(args, **kwargs)