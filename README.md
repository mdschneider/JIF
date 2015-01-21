# JIF
Joint Image Framework (JIF) for modeling astronomical images of stars and galaxies as seen by multiple telescopes.

"Choosy astronomers choose JIF"

## Motivation

How do we optimally combine of galaxies seen from space and ground? The different PSFs, wavelength coverage, 
and pixel sizes can lead to biases in inferred galaxy properties unless included in a joint model 
of all images of the same source. If sources are blended together in any observations, the need for 
joint modeling becomes even more acute.

## People

- Will Dawson (LLNL)
- Michael Schneider (LLNL)

## Steps in making peanut butter

1. Shell - source extraction from space data
2. Roast - Interim sampling
3. Grind - Hierarchical inference via importance sampling
4. Mix / salt - posterior inferences