#!/bin/bash
#BSUB -J points-block-constant-131072
#BSUB -o points-block-constant-131072.o%J
#BSUB -e points-block-constant-131072.e%J
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

. $ROOT/$METHOD/scripts/vortex/common.sh

# don't match fade 1.0 (full blocks)
# don't match fill (full blocks)
mats=\
"
$ROOT/static/block-constant_131072_*_!(1.0)_0_*_bs16.mtx \
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
        F1 JSRUN $ROOT/$METHOD/build/$exe 16 0.3 $mat
    done
    echo ""
done

date
