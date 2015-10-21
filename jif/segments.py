#!/usr/bin/env python
# encoding: utf-8
"""
Utilities for creating and parsing 'segment' files

A segment file must include the following information:

    - Segment image data in [group_name]/observation/[algorithm]/segments
    - Bandpass information in [group_name]/filters/[filter_name]

Optional additional information might include:
"""
import numpy as np
import h5py


class Segments(object):
    """
    I/O for galaxy image segments
    """
    def __init__(self, segment_file):
        self.segment_file = segment_file

        self.file = h5py.File(segment_file, 'w')

    def save_tel_metadata(self, group_name='ground', telescope='lsst',
                          primary_diam=8.4, pixel_scale_arcsec=0.2,
                          atmosphere=True):
        if group_name not in self.file:
            g = self.file.create_group(group_name)
        else:
            g = self.file[group_name]
        g.attrs['telescope'] = telescope
        g.attrs['primary_diam'] = primary_diam
        g.attrs['pixel_scale_arcsec'] = pixel_scale_arcsec
        g.attrs['atmosphere'] = atmosphere
        return None

    def save_images(self,
                    image_list,
                    noise_list,
                    mask_list,
                    group_name = 'ground',
                    source_extraction = 'sextractor',
                    filter_name = 'r'):
        """
        Save images for the segments from a single telescope
        """
        segment_name = '{}/observation/{}/segments'.format(group_name, source_extraction)
        # g_obs_sex_seg = self.file.create_group(segment_name)

        for iseg, im in enumerate(image_list):
            seg = self.file.create_group(segment_name + '/{:d}'.format(iseg))
            # image - background
            seg.create_dataset('image', data=im)
            # rms noise
            seg.create_dataset('noise', data=noise_list[iseg])
            # estimate the variance of this noise image and save as attribute
            seg.attrs['variance'] = np.var(noise)
            seg.create_dataset('segmask', data=mask_list[iseg])
        return None

    def save_bandpasses(self, filters_list, waves_nm_list, throughputs_list,
                        group_name='ground'):
        """
        Save bandpasses for a single telescope as lookup tables.
        """
        for i, filter_name in enumerate(filters_list):
            bp = self.file.create_group('{}/filters/{}'.format(group_name,
                filter_name))
            bp.create_dataset('waves_nm', data=waves_nm_list[i])
            bp.create_dataset('throughput', data=throughputs_list[i])
        return None
