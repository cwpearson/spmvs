#!/bin/bash
#BSUB -J points-block-diagonal-variable-524288-0
#BSUB -o points-block-diagonal-variable-524288-0.o%J
#BSUB -e points-block-diagonal-variable-524288-0.e%J
#BSUB -W 02:00
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
# separate out sprinkes since performance is quite different
mats=\
"
$ROOT/static/block-diagonal-variable_524288_*_*_0_0.mtx \
$ROOT/static/block-diagonal-variable_524288_*_*_0_0_pad16.mtx \
$ROOT/static/block-diagonal-variable_524288_*_*_0_0.mtx_v-cycle.mtx \
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
