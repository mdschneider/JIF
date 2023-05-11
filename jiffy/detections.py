import numpy as np
import pickle

class EmptyDetectionCorrection(object):
    '''
    DetectionCorrection form for the image model parameters
    Applies no correction for all parameters (for any given parameterization).
    '''
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return 0.0

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
                                         126.20048187255851, 149.33343566894533, 180.8871972656249, 201.32699584960938,
                                         225.43019714355478, 256.11255187988326, 298.5987194824225, 323.7379516601566, 
                                         354.6388146972656, 385.9149784851075, 425.59907165527227, 476.167833404541,
                                         536.8243225097649, 621.530991821288, 722.1686499023436, 858.95066619873, 
                                         1039.0751892089845, 1323.2170349121072, 1584.897214355465, 1954.9564749755848,
                                         2508.5373203124877, 3314.6579921874927, 4567.138803710906])
        # Convert from nJy to instrumental flux, using calibrations from DC2 study
        self.flux_bins_upper *= 0.017100155374837712
        self.flux_bins_upper += -1.073090269533189
        
        detected_frac = np.array([0.0007389110182013105, 0.0007851891802036702, 0.0010106735953169163, 0.001017000051464549,
                                       0.0010235495219841231, 0.0012299710175904973, 0.0012792173299028132, 0.001454382992592537,
                                       0.0018617649431903996, 0.0021245124416635966, 0.0023617543592913386, 0.002602798388947123,
                                       0.0030302305480409, 0.0031886509667079896, 0.0038042989243343983, 0.005019089345324304,
                                       0.00641525648926068, 0.007270910714866668, 0.009818681777706076, 0.011819661417832326,
                                       0.014812218940266967, 0.020211556269328333, 0.023801183595974985, 0.029870365615255863,
                                       0.03627161780031299, 0.04188025740219244, 0.05101675683678421, 0.06081006509381948,
                                       0.07036615146818923, 0.07772014621795387, 0.09189516129801277, 0.10343125983915473,
                                       0.11477903043863565, 0.12468318654152415, 0.11879159430261399, 0.11215910678685571,
                                       0.09779695035132922, 0.08689096355565942, 0.07554777907105747, 0.06730674577533267,
                                       0.06320871635364964, 0.05687542361962912, 0.05331862809000617, 0.04933409219803814,
                                       0.04636108123825295, 0.04208074621293895, 0.04025619844442545, 0.03581281356628938,
                                       0.03318474436787526, 0.03092606432754239, 0.02778274067041314, 0.02664995321179559,
                                       0.024019552532210272, 0.021412421269366242, 0.020695216056077402])
        self.neg_log_detected_frac = -np.log(detected_frac)
    
    def evaluate(self, flux):
        idx = np.digitize(flux, self.flux_bins_upper, right=False)
        idx = min(idx, len(self.neg_log_detected_frac) - 1)
        neg_log_prob = self.neg_log_detected_frac[idx]
        
        return neg_log_prob

    def __call__(self, src_models):
        flux = src_models[0].params.flux[0]
        
        return self.evaluate(flux)

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
