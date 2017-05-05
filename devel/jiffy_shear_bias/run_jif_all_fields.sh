#!/bin/tcsh
#
# run_jif_all_fields.sh
#
# Run JIF Roaster on all GREAT3-like 'fields'
#

foreach field (`seq 3 49`)
    set field_lab=`echo $field | awk '{printf "%03d\n", $0;}'`
    ./run_jif_cgc_test.sh $field_lab
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
