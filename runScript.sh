#!/bin/bash
echo '----------------- nvcc Info: -------------------'
nvcc --version
echo '----------------- Compiling --------------------'
SCRIPT=$1
nvcc $SCRIPT -o exe --ptxas-options=-v
echo '----------------- Executing --------------------'
shift
srun -v --exclusive -p ClsParSystems --time=1:00:00 --gres=gpu:TitanX:1 ./exe "$@"
