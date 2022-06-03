#!/bin/bash
#BSUB -J points-block-constant-524288
#BSUB -o points-block-constant-524288.o%J
#BSUB -e points-block-constant-524288.e%J
#BSUB -W 01:00
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
kk-hybrid-spmv-tc-cusparse-fp16-fp16 \
kk-hybrid-spmv-tc-cusparse-fp64-fp64 \
kk-hybrid-spmv-tc-native-fp16-fp16 \
kk-hybrid-spmv-tc-native-fp64-fp64 \
"


# don't match fade 1.0 (full blocks)
# don't match fill (full blocks)
mats=\
"
$ROOT/static/block-constant_524288_*_!(1.0)_0.0_*_bs16.mtx \
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
