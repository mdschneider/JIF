#!/bin/tcsh
#
# run_stooker_all_fields.sh
# 
# Run MBI Stooker on all GREAT3-like 'fields' to thin and re-package JIF Roaster
# outputs for input to Thresher.
# 

cd midsnr/reaper/jif
foreach field (`seq 0 49`)
    set field_lab=`echo $field | awk '{printf "%03d\n", $0;}'`
    cd $field_lab
    echo roaster_${field_lab}_seg*.h5
    jiffy_stooker roaster_${field_lab}_seg*.h5 -o reaper_${field_lab}.h5
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
