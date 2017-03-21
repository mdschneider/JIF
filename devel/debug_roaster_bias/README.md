Debugging biased marginal posteriors from Roaster
=================================================

**Created on February 11, 2017**

We observe biases between the mean or mode of marginal posteriors from Roaster and the input truth values of simulated galaxy images. Why is this?

Our expectation is that the mean of Roaster marginal posteriors should match the input parameter value to arbitrary precision as the SNR becomes large (and perhaps assuming a sufficiently large stamp size).


## Simulation study plan

1. Specify the (i) noise rms relative to the galaxy flux and (ii) stamp size relative to the galaxy size and SB profile slope
2. Simulate the galaxy image with GalSimGalaxy
3. Run Roaster with only one parameter 'active'
4. Run Roaster with all parameters 'active'
5. Repeat 1-4 with different stamp sizes and SNRs
6. Compare differences in the marginal posterior means and modes with 'truth' values as a function of varied settings