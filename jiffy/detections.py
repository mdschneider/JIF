import numpy as np
import pickle

class EmptyDetectionCorrection(object):
    '''
    DetectionCorrection form for the image model parameters
    Applies no correction for all parameters (for any given parameterization).
    '''

    def __init__(self):
        pass

    def __call__(self, *args):
        return 0.0

class DetectionCorrectionFromPickle(object):
    def __init__(self, detection_correction_filename='detection_correction_file.pkl'):
        with open(detection_correction_filename, mode='rb') as detection_correction_file:
            self.detection_correction_function = pickle.load(detection_correction_file)
    
    def __call__(self, params):
        detection_correction = self.detection_correction_function(params)
        return detection_correction

detection_corrections = {None: False,
          'Empty': EmptyDetectionCorrection,
          'EmptyDetectionCorrection': EmptyDetectionCorrection,
          'DetectionCorrectionFromPickle': DetectionCorrectionFromPickle}

def initialize_detection_correction(form=None, module=None, **kwargs):
    if module is None:
        # form should be one of the names of detection_corrections in this file
        detection_correction = detection_corrections[form]
    else:
        module = __import__(module)
        # detection_correction_form should be the name of a class in module
        detection_correction = getattr(module, form)

    # Initialize an instance of the prior with the given keyword arguments
    return detection_correction(**kwargs)
