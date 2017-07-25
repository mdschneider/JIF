#!/bin/bash

datadir=/Volumes/PromisePegasus/JIF/cgc1/reaper/jif/068/

convert -set delay 100 -colors 256 -dispose 1 -loop 0 -scale 50% ${datadir}/*data_and_model_epoch0.png ./doc/data_and_model_068.gif

#convert -set delay 100 -colors 256 -dispose 1 -loop 0 -scale 50% ${datadir}/*walkers.png ./doc/walkers_000.gif

convert -set delay 100 -colors 256 -dispose 1 -loop 0 -scale 50% ${datadir}/*triangle.png ./doc/triangle_068.gif

#convert -set delay 100 -colors 256 -dispose 1 -loop 0 -scale 50% ${datadir}/*gr_statistic.png ./doc/gr_statistic_000.gif
