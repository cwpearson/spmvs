#!/bin/bash
#SBATCH -A m3918_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 1:00:00
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
export TMP_DIR=/vscratch1/cwpears
export OUT_DIR=$ROOT/scripts/vortex

. $ROOT/$METHOD/load-env.sh

set -eou pipefail

date

export KOKKOS_NUM_DEVICES=1
export CUDA_LAUNCH_BLOCKING=0

function SRUN () {
srun \
"$@"
}

function F1 () {
    "$@" | cut -d"," -f1 | tr -d '\n'
}

function F2 () {
    "$@" | cut -d"," -f2 | tr -d '\n'
}

function F3 () {
    "$@" | cut -d"," -f3 | tr -d '\n'
}

function F4 () {
    "$@" | cut -d"," -f4 | tr -d '\n'
}

function F5 () {
    "$@" | cut -d"," -f5 | tr -d '\n'
}

bsr_exes=\
"
kk-bsr-spmv-cusparse-fp64-fp64 \
kk-bsr-spmv-native-fp16-fp16 \
kk-bsr-spmv-native-fp64-fp64 \
kk-bsr-spmv-tc-fp16-fp16 \
kk-bsr-spmv-tc-fp64-fp64 \
"

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

# any matrix with the same block structure should be the same
# don't need faded with fill because it's the same as non-fade
block_mats=\
"
$STATIC/block-constant_131072_*_1.0_*_0_bs16.mtx \
$STATIC/block-diagonal-constant_131072_1.0_0_0_bs16.mtx \
$STATIC/block-diagonal-variable_131072_*_1.0_*_0_pad16_fill16.mtx \
$STATIC/block-variable_131072_*_*_1.0_*_0_pad16_fill16.mtx \
"

date

echo -n "mat"
for exe in $bsr_exes; do
    echo -n ","$exe
done
for exe in $crs_exes; do
    echo -n ","$exe
done
for exe in $hybrid_exes; do
    echo -n ","$exe
done
echo ""

for mat in $block_mats; do
    echo -n `basename $mat`
    for exe in $bsr_exes; do
        echo -n ","
        SRUN $ROOT/$METHOD/build/$exe 16 $mat
    done
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
