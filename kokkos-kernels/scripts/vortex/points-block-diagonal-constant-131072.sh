#!/bin/bash
#BSUB -J points-block-diagonal-constant-131072
#BSUB -o points-block-diagonal-constant-131072.o%J
#BSUB -e points-block-diagonal-constant-131072.e%J
#BSUB -W 00:30
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

. $ROOT/$METHOD/scripts/vortex/common.sh

# dont match fill (full blocks)
# dont match fade 1.0 (full blocks)
mats=\
"
$ROOT/static/block-diagonal-constant_131072_!(1.0)_0_*_bs16.mtx \
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
    echo -n ",hybrid-"$exe
    echo -n ",rem-"$exe
    echo -n ",dense-"$exe
done
echo ""

for mat in $mats; do
    # print matrix name
    echo -n `basename $mat`

    # print matrix statistics
    echo -n ","
    F4-7 JSRUN $ROOT/$METHOD/build/kk-hybrid-spmv-tc-cusparse-fp16-fp16 16 0.5 $mat

    # print performance 
    for exe in $crs_exes; do
        echo -n ","
        JSRUN $ROOT/$METHOD/build/$exe $mat
    done
    for exe in $hybrid_exes; do
        echo -n ","
        F1-3 JSRUN $ROOT/$METHOD/build/$exe 16 0.3 $mat
    done
    echo ""
done

date
