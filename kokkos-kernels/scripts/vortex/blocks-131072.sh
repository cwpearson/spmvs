#!/bin/bash
#BSUB -J blocks-131072
#BSUB -o blocks-131072.o%J
#BSUB -e blocks-131072.e%J
#BSUB -W 04:00
#BSUB -nnodes 1

export ROOT=$HOME/repos/spmvs
export METHOD=kokkos-kernels
export TMP_DIR=/vscratch1/cwpears
export OUT_DIR=$ROOT/scripts/vortex

. $ROOT/$METHOD/load-env.sh

set -eou pipefail

export KOKKOS_NUM_DEVICES=1
export CUDA_LAUNCH_BLOCKING=0

. $ROOT/$METHOD/scripts/vortex/common.sh

# any matrix with the same block structure should be the same
# don't need faded with fill because it's the same as non-fade
block_mats=\
"
$ROOT/static/block-constant_131072_*_1.0_*_0_bs16.mtx \
$ROOT/static/block-constant-hybrid_131072_*_*_0.0_0_bs16.mtx \
$ROOT/static/block-diagonal-constant_131072_1.0_0_0_bs16.mtx \
"

date

echo -n "mat,nnz"
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

    # print matrix nnz
    echo -n ","
    F4 JSRUN $ROOT/$METHOD/build/kk-hybrid-spmv-tc-cusparse-fp16-fp16 16 0.3 $mat

    for exe in $bsr_exes; do
        echo -n ","
        JSRUN $ROOT/$METHOD/build/$exe 16 $mat
    done
    for exe in $crs_exes; do
        echo -n ","
        JSRUN $ROOT/$METHOD/build/$exe $mat
    done
    # Times: hybrid, remainder, dense
    for exe in $hybrid_exes; do
        echo -n ","
        F1-3 JSRUN $ROOT/$METHOD/build/$exe 16 0.3 $mat
    done
    echo ""
done

date
