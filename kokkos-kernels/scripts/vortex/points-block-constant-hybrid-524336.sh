#!/bin/bash
#BSUB -J points-block-constant-hybrid-524336
#BSUB -o points-block-constant-hybrid-524336.o%J
#BSUB -e points-block-constant-hybrid-524336.e%J
#BSUB -W 04:00
#BSUB -nnodes 1

export ROOT=$HOME/repos/spmvs
export METHOD=kokkos-kernels
export TMP_DIR=/vscratch1/cwpears
export OUT_DIR=$ROOT/scripts/vortex

. $ROOT/$METHOD/load-env.sh

set -eou pipefail
shopt -s extglob

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
-l gpu-cpu \
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
kk-hybrid-spmv-cusparse-cusparse-fp16-fp16 \
kk-hybrid-spmv-cusparse-cusparse-fp64-fp64 \
kk-hybrid-spmv-tc-cusparse-fp16-fp16 \
kk-hybrid-spmv-tc-cusparse-fp64-fp64 \
kk-hybrid-spmv-tc-native-fp16-fp16 \
kk-hybrid-spmv-tc-native-fp64-fp64 \
"



mats=\
"
$ROOT/static/block-constant-hybrid_524336_0.0001952972424029373_0.5_0.1_0_bs16.mtx
$ROOT/static/block-constant-hybrid_524336_0.0001952972424029373_0.5_0.5_0_bs16.mtx
$ROOT/static/block-constant-hybrid_524336_0.0001952972424029373_0.5_1.0_0_bs16.mtx
$ROOT/static/block-constant-hybrid_524336_0.0001952972424029373_0.5_2.0_0_bs16.mtx
$ROOT/static/block-constant-hybrid_524336_0.0001952972424029373_0.7_0.1_0_bs16.mtx
$ROOT/static/block-constant-hybrid_524336_0.0001952972424029373_0.7_0.5_0_bs16.mtx
$ROOT/static/block-constant-hybrid_524336_0.0001952972424029373_0.7_1.0_0_bs16.mtx
$ROOT/static/block-constant-hybrid_524336_0.0001952972424029373_0.7_2.0_0_bs16.mtx
$ROOT/static/block-constant-hybrid_524336_0.0001952972424029373_1.0_0.1_0_bs16.mtx
$ROOT/static/block-constant-hybrid_524336_0.0001952972424029373_1.0_0.5_0_bs16.mtx
$ROOT/static/block-constant-hybrid_524336_0.0001952972424029373_1.0_1.0_0_bs16.mtx
$ROOT/static/block-constant-hybrid_524336_0.0001952972424029373_1.0_2.0_0_bs16.mtx
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
    F2-5 JSRUN $ROOT/$METHOD/build/kk-hybrid-spmv-tc-cusparse-fp16-fp16 16 0.5 $mat

    # print performance 
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
