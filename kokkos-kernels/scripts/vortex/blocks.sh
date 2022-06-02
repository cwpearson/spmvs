#!/bin/bash
#BSUB -J blocks
#BSUB -o blocks.o%J
#BSUB -e blocks.e%J
#BSUB -W 04:00
#BSUB -nnodes 1

export ROOT=$HOME/repos/spmvs
export METHOD=kokkos-kernels
export TMP_DIR=/vscratch1/cwpears
export OUT_DIR=$ROOT/scripts/vortex

. $ROOT/$METHOD/load-env.sh

set -eou pipefail

date

export KOKKOS_NUM_DEVICES=1
export CUDA_LAUNCH_BLOCKING=0

function JSRUN () {
jsrun \
--smpiargs="-disable_gpu_hooks" \
-n 1 \
-r 1 \
-a 1 \
-g 1 \
-c 2 \
-b rs \
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
block_mats=\
"
$ROOT/static/block-constant_1024_*_1.0_*_0_bs16.mtx \
$ROOT/static/block-diagonal-constant_1024_1.0_0.0_0_bs16.mtx \
$ROOT/static/block-diagonal-variable_1024_*_1.0_*_0_pad16_fill16.mtx \
$ROOT/static/block-variable_1024_*_*_1.0_*_0_pad16_fill16.mtx \
$ROOT/static/block-constant_16384_*_1.0_*_0_bs16.mtx \
$ROOT/static/block-diagonal-constant_16384_1.0_0.0_0_bs16.mtx \
$ROOT/static/block-diagonal-variable_16384_*_1.0_*_0_pad16_fill16.mtx \
$ROOT/static/block-variable_16384_*_*_1.0_*_0_pad16_fill16.mtx \
$ROOT/static/block-constant_131072_*_1.0_*_0_bs16.mtx \
$ROOT/static/block-diagonal-constant_131072_1.0_0.0_0_bs16.mtx \
$ROOT/static/block-diagonal-variable_131072_*_1.0_*_0_pad16_fill16.mtx \
$ROOT/static/block-variable_131072_*_*_1.0_*_0_pad16_fill16.mtx \
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
        JSRUN $ROOT/$METHOD/build/$exe 16 $mat
    done
    for exe in $crs_exes; do
        echo -n ","
        JSRUN $ROOT/$METHOD/build/$exe $mat
    done
    for exe in $hybrid_exes; do
        echo -n ","
        F1 JSRUN $ROOT/$METHOD/build/$exe 16 0.5 $mat
    done
    echo ""
done

date
