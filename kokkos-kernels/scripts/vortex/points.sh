#!/bin/bash
#BSUB -J points
#BSUB -o points.o%J
#BSUB -e points.e%J
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

mats=\
"
$ROOT/static/block-constant_1024_*_*_0.0_0_bs16.mtx \
$ROOT/static/block-constant_16384_*_*_0.0_0_bs16.mtx \
$ROOT/static/block-constant_131072_*_*_0.0_0_bs16.mtx \
$ROOT/static/block-diagonal-constant_1024_*_*_0_bs16.mtx \
$ROOT/static/block-diagonal-constant_16384_*_*_0_bs16.mtx \
$ROOT/static/block-diagonal-constant_131072_*_*_0_bs16.mtx \
$ROOT/static/block-diagonal-variable_1024_*_*_*_0.mtx \
$ROOT/static/block-diagonal-variable_1024_*_*_*_*_pad16.mtx \
$ROOT/static/block-diagonal-variable_16384_*_*_*_0.mtx \
$ROOT/static/block-diagonal-variable_16384_*_*_*_0_pad16.mtx \
$ROOT/static/block-diagonal-variable_131072_*_*_*_0.mtx \
$ROOT/static/block-diagonal-variable_131072_*_*_*_0_pad16.mtx \
$ROOT/static/block-variable_1024_*_*_*_*_0.mtx \
$ROOT/static/block-variable_1024_*_*_*_*_0_pad16.mtx \
$ROOT/static/block-variable_16384_*_*_*_*_0.mtx \
$ROOT/static/block-variable_16384_*_*_*_*_0_pad16.mtx \
$ROOT/static/block-variable_131072_*_*_*_*_0.mtx \
$ROOT/static/block-variable_131072_*_*_*_*_0_pad16.mtx \
$HOME/suitesparse/Fault_639/Fault_639.mtx \
$HOME/suitesparse/Bump_2911/Bump_2911.mtx \
"

date

echo -n "mat"
for exe in $crs_exes; do
    echo -n ","$exe
done
for exe in $hybrid_exes; do
    echo -n ","$exe
done
echo ""

for mat in $mats; do
    echo -n `basename $mat`
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
