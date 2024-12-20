#!/bin/bash
#SBATCH --no-requeue
#SBATCH --job-name="si_supercell"
#SBATCH --get-user-env
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=23:30:00

#SBATCH --account=mr33
#SBATCH --partition=normal
#SBATCH --constraint=mc

module load cray/23.12
module switch PrgEnv-cray PrgEnv-intel
module load cpeIntel libxc
module unload cray-libsci

export OMP_NUM_THREADS="1"
export PATH=/users/apoliukh/soft/q-e_eiger/bin/:$PATH
export QE_PATH=/users/apoliukh/soft/q-e_eiger/bin
export NMPI=128
export NPOOL=128
export PARA_PREFIX="srun"

$PARA_PREFIX -n $NMPI $QE_PATH/pw.x -in scf.in > scf.out