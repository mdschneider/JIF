#!/bin/tcsh
#
# run_stooker_all_fields.sh
# 
# Run MBI Stooker on all GREAT3-like 'fields' to thin and re-package JIF Roaster
# outputs for input to Thresher.
# 

#cd small_shapenoise/reaper/jif
cd /Volumes/PromisePegasus/JIF/cgc1/reaper/jif
foreach field (`seq 198 199`)
    set field_lab=`echo $field | awk '{printf "%03d\n", $0;}'`
    cd $field_lab
    echo $field_lab
    rm -f reaper_${field_lab}.h5
    # echo roaster_${field_lab}_seg*.h5
    # jiffy_stooker roaster_${field_lab}_seg*.h5 -o reaper_${field_lab}.h5
    # Use xargs here because the list of files can be to long for 'ls'
    find . -type f -print | xargs jiffy_stooker -o reaper.h5
    mv reaper.h5 reaper_${field_lab}.h5
    cd ../
end

done:
    echo " "
    echo "Finished run_stooker_all_fields"
    echo " "
    exit 0
error:
    echo " "
    echo "ERROR in run_stooker_all_fields"
    echo " "
    exit 1
