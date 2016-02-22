#!/usr/bin/env python
# encoding: utf-8
"""
Plot the bandpasses supplied in the GalSim WFIRST module
"""
import numpy as np
import matplotlib.pyplot as plt
import galsim
import galsim.wfirst as wfirst

### Load
filters = wfirst.getBandpasses(AB_zeropoint=True)

### Print the wavelength ranges for each filter
print "\nred / blue limits (nm):"
for key, val in filters.iteritems():
    print "\t{}: {:5.4f} / {:5.4f}".format(key, val.red_limit, val.blue_limit)

### Plot
waves_nm = np.linspace(700., 2100., 200)
fig = plt.figure(figsize=(8., 8/1.618))
i = 0
for key, bp in filters.iteritems():
    plt.plot(waves_nm, bp(waves_nm))
    plt.annotate(key, xy=(bp.effective_wavelength, 0.7))
    i = i+1
plt.ylim(0, 1)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Throughput")

plt.savefig("wfirst_bandpasses.pdf", bbox_inches='tight')
