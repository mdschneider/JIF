#!/usr/bin/env python
# encoding: utf-8
"""
Plot the bandpasses supplied in the GalSim WFIRST module
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import galsim
import galsim.wfirst as wfirst

colors = ['#348ABD', '#7A68A6', '#A60628', '#467821', '#CF4457', '#188487', '#E24A33']
# k_SED_names = ['CWW_E_ext', 'CWW_Sbc_ext', 'CWW_Scd_ext', 'CWW_Im_ext']
k_SED_names = ['NGC_0695_spec', 'NGC_4125_spec', 'NGC_4552_spec', 'CGCG_049-057_spec']

def load_SEDs():
    path, filename = os.path.split(__file__)
    datapath = os.path.abspath(os.path.join(path, "../input/"))
    SEDs = {}
    for SED_name in k_SED_names:
        SED_filename = os.path.join(datapath, '{0}.sed'.format(SED_name))
        SEDs[SED_name] = galsim.SED(SED_filename, wave_type='Ang')
    return SEDs

### Load
filters = wfirst.getBandpasses(AB_zeropoint=True)
seds = load_SEDs()

### Print the wavelength ranges for each filter
print "\nFilter red / blue limits (nm):"
for key, val in filters.iteritems():
    print "\t{}: {:5.4f} / {:5.4f}".format(key, val.red_limit, val.blue_limit)

### Print the wavelength ranges for each SED
print "\nSED red / blue limits (nm):"
for key, val in seds.iteritems():
    print "\t{}: {:5.4f} / {:5.4f}".format(key, val.red_limit, val.blue_limit)


### Plot
waves_nm = np.linspace(600., 2100., 200)
fig = plt.figure(figsize=(8., 8/1.618))
i = 0
for key, bp in filters.iteritems():
    if key != 'W149':
        plt.plot(waves_nm, bp(waves_nm), color=colors[i])
        plt.annotate(key, xy=(bp.effective_wavelength, 0.7), color=colors[i])
        i = i+1
plt.ylim(0, 1)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Throughput")

# Add the SEDs to the plot
waves_nm_seds = np.linspace(20., 2500., 200)
for key, sed in seds.iteritems():
    s = sed(waves_nm_seds)
    s /= (2 * np.max(s))
    plt.plot(waves_nm_seds, s, linestyle='dashed', label=key)

plt.savefig("wfirst_bandpasses.pdf", bbox_inches='tight')
