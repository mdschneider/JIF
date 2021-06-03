#!/bin/csh
#
# For instructions on running multiple jobs with SLURM, see:
# https://computing.llnl.gov/tutorials/bgq/index.html#UQ
# and,
# https://slurm.schedmd.com/job_array.html
#
# Job name
#MSUB -N roaster
#MSUB -A darkbia
#
# Combine stdout and stderr
#MSUB -j oe
#MSUB -o /p/lscratchd/mdschnei/ldrd2016/JIF/logs/roaster_test.log
#MSUB -m bea
#
#MSUB -q pdebug
#MSUB -l nodes=1
#MSUB -l partition=cab
#MSUB -l walltime=00:02:00
#MSUB -V
echo "---------------------------"
date
echo "Job id = $SLURM_JOBID"
echo "Proc id = $SLURM_PROCID"
hostname
#sinfo
#squeue
#

set n_gals=10000
set workdir=/p/lscratchd/mdschnei/ldrd2016/JIF/cgc1

set python=/g/g20/mdschnei/ldrd2016/JIF/env/mbi/bin/python
set roaster_wrapper=/g/g20/mdschnei/ldrd2016/JIF/devel/jiffy_shear_bias/roaster_wrapper.py
set config_file_template=/g/g20/mdschnei/ldrd2016/JIF/devel/jiffy_shear_bias/jiffy_cgc1.yaml


set field=$1
set logdir=/p/lscratchd/mdschnei/ldrd2016/JIF/logs/$field
set config_file=/g/g20/mdschnei/ldrd2016/JIF/devel/jiffy_shear_bias/jiffy_cgc1_${field}.yaml

mkdir -p $logdir
cp $config_file_template $config_file

rm -rf update_config_${field}.py

cat >>update_config_${field}.py <<EOF
#!/usr/bin/env python
import yaml

config = yaml.load(open("${config_file}"))
config['io']['infile'] = '${workdir}/control/ground/constant/segments/seg_${field}.h5'
config['io']['roaster_outfile'] = '${workdir}/reaper/jif/${field}/roaster_${field}'

with open("${config_file}", "w") as f:
    yaml.dump(config, f, indent=4, default_flow_style=False)
EOF

$python update_config_${field}.py


#srun -N1 -n1 -o $logdir/roaster%A_%a.out $python $roaster_wrapper --config_file $config_file \
#--footprint_number $SLURM_ARRAY_TASK_ID --field $field --workdir $workdir &

foreach segnum (`seq 0 $n_gals`)
	#echo $segnum "::" $python $roaster_wrapper --config_file $config_file --footprint_number $segnum --field $field --workdir $workdir
	srun -N1 -n1 -o $logdir/roaster.%J $python $roaster_wrapper --config_file $config_file --footprint_number $segnum --field $field --workdir $workdir &
	# Avoid possible timing problems with the batch scheduler
	sleep 1
end
wait

#srun -N1 -n1 $python $roaster_wrapper --config_file $config_file --footprint_number 0 --field $field --workdir $workdir

date
echo "All done!"
