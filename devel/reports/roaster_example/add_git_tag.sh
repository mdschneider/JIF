#!/bin/bash
#
# Run a computation and add a header file with the git commit number,
# machine name, and directory so the exact run can be reproduced at a
# later date.
#

outfile='git_tag.tex'
rm -f $outfile

tag=`git show | grep "commit " | awk '{print $2}'`
host=`hostname`
account=`whoami`
dir=`pwd`
time=`date`

# header="# ${time}\n# git commit: ${tag}\n# machine: ${account}@${host}\n# directory: ${dir}"
header="git commit: ${tag}"

exec 3<> $outfile && awk -v TEXT="$header" 'BEGIN {print TEXT}{print}' $outfile >&3
