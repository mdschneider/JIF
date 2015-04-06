"""
JIF wrapper for CosmoSIS

CosmoSIS handles the MCMC sampling, I/O, and plotting.
We only need the Load() and __call__() methods of Roaster.
"""
import numpy as np
from cosmosis.datablock import names as section_names
from cosmosis.datablock import option_section
from cosmosis.runtime.declare import declare_module

import Roaster


class RoasterModule(object):

    likes = section_names.likelihoods
    
    def __init__(self,my_config,my_name):
        self.mod_name = my_name
        self.infiles = my_config[my_name, "infiles"]

        self.roaster = Roaster.Roaster()
        self.roaster.Load(self.infiles)
        print "\nRoaster:", self.roaster.__dict__, "\n"
        # print "\nsource models:",src_models[0][0].__dict__, "\n"

    def execute(self, block):
        ### Load all galaxy model parameters from the data block and pass to the likelihood
        gal_flux = block["spergel_galaxy", "gal_flux"]
        nu = block["spergel_galaxy", "nu"]
        hlr = block["spergel_galaxy", "hlr"]
        e = block["spergel_galaxy", "e"]
        beta = block["spergel_galaxy", "beta"]
        omega = np.array([gal_flux, nu, hlr, e, beta])

        like = self.roaster(omega)

        block[RoasterModule.likes, "JIF_ROASTER_LIKE"] = like
        return 0
        
    def cleanup(self):
        return 0

declare_module(RoasterModule)

