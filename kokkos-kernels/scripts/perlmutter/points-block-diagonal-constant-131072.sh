#!/bin/bash
#SBATCH -A m3918_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 0:30:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=1
#SBATCH --output=%x.%j.o
#SBATCH --error=%x.%j.e

export SLURM_CPU_BIND="cores"

export ROOT=$HOME/repos/spmvs
export STATIC=/global/cfs/cdirs/m3918/pearson/sisu
export METHOD=kokkos-kernels
export OUT_DIR=$ROOT/scripts/perlmutter

. $ROOT/$METHOD/load-env.sh

set -eou pipefail
shopt -s extglob

export KOKKOS_NUM_DEVICES=1
export CUDA_LAUNCH_BLOCKING=0

function SRUN () {
srun \
"$@"
}

function F1 () {
    "$@" | cut -d"," -f1 | tr -d '\n'
}

function F2-5 () {
    "$@" | cut -d"," --fields=2,3,4,5 | tr -d '\n'
}

crs_exes=\
"
kk-crs-spmv-cusparse-fp16-fp16 \
kk-crs-spmv-cusparse-fp64-fp64 \
kk-crs-spmv-native-fp16-fp16 \
kk-crs-spmv-native-fp64-fp64 \
"

hybrid_exes=\
"
kk-hybrid-spmv-tc-cusparse-fp16-fp16 \
kk-hybrid-spmv-tc-cusparse-fp64-fp64 \
kk-hybrid-spmv-tc-native-fp16-fp16 \
kk-hybrid-spmv-tc-native-fp64-fp64 \
"

# dont match fill (full blocks)
# dont match fade 1.0 (full blocks)
mats=\
"
$STATIC/block-diagonal-constant_131072_!(1.0)_0_*_bs16.mtx \
$STATIC/block-diagonal-constant_131072_!(1.0)_1000_*_bs16.mtx \
"

date

echo -n "mat"

# column headers for matrix statistics
echo -n ",nnz,dense nnz (real),dense nnz (fill),sparse nnz"

# column header for each method
for exe in $crs_exes; do
    echo -n ","$exe
done
for exe in $hybrid_exes; do
    echo -n ","$exe
done
echo ""

for mat in $mats; do
    # print matrix name
    echo -n `basename $mat`

    # print matrix statistics
    echo -n ","
    F2-5 SRUN $ROOT/$METHOD/build/kk-hybrid-spmv-tc-cusparse-fp16-fp16 16 0.5 $mat

    # print performance 
    for exe in $crs_exes; do
        echo -n ","
        SRUN $ROOT/$METHOD/build/$exe $mat
    done
    for exe in $hybrid_exes; do
        echo -n ","
        F1 SRUN $ROOT/$METHOD/build/$exe 16 0.5 $mat
    done
    echo ""
done

date
