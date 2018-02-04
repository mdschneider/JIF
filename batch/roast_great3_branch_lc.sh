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
#MSUB -o /p/lscratchd/mdschnei/ldrd2016/JIF/logs/roaster.log
#MSUB -m bea
#
#MSUB -q pbatch
#MSUB -l nodes=2
#MSUB -l partition=cab
#MSUB -l walltime=00:10:00
#MSUB -V

set field=$1
set n_gals=16
set workdir=/p/lscratchd/mdschnei/ldrd2016/JIF/cgc1

set python=/g/g20/mdschnei/ldrd2016/JIF/env/mbi/bin/python
set roaster_wrapper=/g/g20/mdschnei/ldrd2016/JIF/devel/jiffy_shear_bias/roaster_wrapper.py

set config_file=jiffy_cgc1.yaml
# set params_file=jiffy_cgc1_params.cfg

echo "---------------------------"
date
echo "Job id = $SLURM_JOBID"
echo "Proc id = $SLURM_PROCID"
hostname
#sinfo
#squeue
#

rm -rf update_config.py

cat >>update_config.py <<EOF
#!/usr/bin/env python
import yaml

config = yaml.load(open("${config_file}"))
config['io']['infile'] = '${workdir}/control/ground/constant/segments/seg_${field}.h5'
config['io']['roaster_outfile'] = '${workdir}/reaper/jif/${field}/roaster_${field}'

with open("${config_file}", "w") as f:
    yaml.dump(config, f, indent=4, default_flow_style=False)
EOF

$python update_config.py

foreach segnum (`seq 0 $n_gals`)
	srun -N1 -n1 $python $roaster_wrapper --config_file $config_file --footprint_number $segnum --field $field --workdir $workdir &
date
echo "All done!"
