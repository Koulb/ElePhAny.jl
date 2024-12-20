#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="si_kcw_2"
#SBATCH --get-user-env
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=23:30:00

#SBATCH --account=mr33
#SBATCH --partition=normal
#SBATCH --constraint=mc

module load cray/23.12
module load cray-python
module switch PrgEnv-cray PrgEnv-intel
module load cpeIntel libxc
module unload cray-libsci

export OMP_NUM_THREADS="1"
export PATH=/users/apoliukh/soft/koopmans/quantum_espresso/q-e-kcw/bin:$PATH
export QE_PATH=/users/apoliukh/soft/koopmans/quantum_espresso/q-e-kcw/bin
export NMPI=8
export NPOOL=8
export PARA_PREFIX="srun -n $NMPI"
export PARA_POSTFIX="-npool 8 -pd .true."

koopmans si.json | tee si.out

rm -rf ./TMP/kcw