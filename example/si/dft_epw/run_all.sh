#!/bin/bash

export QE_PATH="/home/poliukhin/Soft/sourse/q-e/bin/"
export NMPI=8
export NPOOL=8
export PARA_PREFIX="mpirun"
export ELEPHANY_PATH="/home/poliukhin/Development/ElectronPhonon/"

copy data ....
cp -r ./displacements/scf_0/tmp/scf.save ./si.save
cp -r ./displacements/scf_0/scf.out ./
echo "0, copy finished finished"

$PARA_PREFIX -n $NMPI $QE_PATH/pw.x -npool $NPOOL -in nscf.in > nscf.out
echo "2, nscf finished"   

$PARA_PREFIX -n $NMPI $QE_PATH/ph.x -npool $NPOOL -in ph.in > ph.out
echo "1, ph finished"

python3 /home/poliukhin/Soft/q-e/EPW/bin/pp.py << EOF
si
EOF
echo "2, pp.py finished"

epw.x -in epw0.in  > epw0.out
echo "3, epw0 finished"

$QE_PATH/epw.x -in epw1.in  > epw1.out
echo "4, epw1 finished"

cp -r si.save/ si_dft.save/ 
cp si.epb1 si_dft.save/

python $ELEPHANY_PATH/epw/parse_epb.py
python $ELEPHANY_PATH/epw/fake2nscf.py

$QE_PATH/epw.x < epw2.in  > epw2.out
echo "4, epw2 finished"