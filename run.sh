#!/bin/bash
#SBATCH --job-name=r_3_np
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
##SBATCH --mem=96G
#SBATCH --partition=matrix2
#SBATCH --workdir=./
#SBATCH --output=./_log_sbatch/%j-%N.out
#SBATCH --error=./_log_sbatch/%j-%N.err

##SBATCH --begin=..
##SBATCH --deadline=21:00

#SBATCH --mail-type=end
#SBATCH --mail-user=wangzhaowei@momenta.ai

module load basic

echo -e "         
********************************************************************
Job Name:$SLURM_JOB_NAME,Job ID:$SLURM_JOBID,Allocate Nodes:$SLURM_JOB_NODELIST
********************************************************************\n\n"

export GLOG_logtostderr=1
export GLOG_log_dir=./log/


DATE=`date +%Y%m%d-%H.%M.%S`
LOGFILE="./log/${DATE}.train.log"


# cmd="python test_net.py wzw --dataset pascal_voc --net detnet59 --checksession 1 --checkepoch 19  --cuda --cascade"
# cmd="python main3.py --gpu_id 0 1 2 3 4 5 6 7 --batch_size 4"
cmd="python main.py --gpu_id 0 1 2 3 4 5 6 7 --batch_size 4"
# cmd="python main3.py --gpu_id 0 1 --batch_size 4"
# cmd="python main.py --gpu_id 0 1 --batch_size 4"
${cmd} 2>&1 |tee ${LOGFILE}
                                                                 
