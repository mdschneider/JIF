#!/bin/tcsh
#
# run_jif_all_fields.sh
#
# Run JIF Roaster on all GREAT3-like 'fields'
#
if ($#argv != 1) then
    echo "Usage: run_jif_all_fields n_gals"
    goto done
endif

set n_gals=$1

foreach field (`seq 0 49`)
    set field_lab=`echo $field | awk '{printf "%03d\n", $0;}'`
    ./run_jif_cgc_test.sh $field_lab $n_gals
end

done:
    echo " "
    echo "Finished run_jif_all_fields"
    echo " "
    exit 0
error:
    echo " "
    echo "ERROR in run_jif_all_fields"
    echo " "
    exit 1
