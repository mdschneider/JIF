#!/bin/csh

foreach field (`seq 20 199`)
    set field_lab=`echo $field | awk '{printf "%03d\n", $0;}'`
    echo $field_lab
    msub batch/roast_great3_field_lc.sh $field_lab
end
