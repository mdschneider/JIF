Shear bias validations with CGC-like simulations
================================================

See: https://github.com/mdschneider/MagicBeans/issues/167 

## Simulation plans

Run label | Profile    | Pr(e_int)   | PSF                     | Magnification
--------- | ---------- | ----------- | ----------------------- | -------------
CGC-1     | Spergel    | Gaussian    | Asserted / isotropic    | 1
CGC-2     | Spergel    | 2 Gaussians | Asserted / isotropic    | 1
CGC-3     | Bulge+Disk | Gaussian    | Asserted / isotropic    | 1
CGC-4     | Spergel    | Gaussian    | Marginalized Kolmogorov | 1
CGC-5     | Spergel    | Gaussian    | Marginalized Kolmogorov | Gaussian dist.

### TODO

- [ ] Compare HSM and Thresher shear inference for each run

## Execution

See `devel/jiffy_shear_bias/run_pip.sh`

1. galsim mbi_cgc*.yaml
2. run_footprints.sh
3. Copy to LC
4. batch/roast_great3_branch.sh
5. run_stooker_all_fields.sh
6. MagicBeans/app/run_thresher.x