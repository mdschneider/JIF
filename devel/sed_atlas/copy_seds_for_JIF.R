#
# Copy select SEDs to JIF input directory
#

SED_names <- c('NGC_0695_spec', 'NGC_4125_spec', 'NGC_4552_spec', 'CGCG_049-057_spec')


for (i in 1:length(SED_names)) {
 sed <- read.table(file.path("123/605", paste(SED_names[i], 'dat', sep='.')),
                   col.names=c("wavelength", "flux", "obs_wavelength", "source"))
 outfile <- file.path("../../input", paste(SED_names[i], 'sed', sep='.'))
 write.table(sed[c("wavelength", "flux")], outfile, row.names=FALSE, col.names=FALSE)
}

