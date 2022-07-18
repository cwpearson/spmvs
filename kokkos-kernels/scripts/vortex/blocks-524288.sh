#!/bin/bash
#BSUB -J blocks-524288
#BSUB -o blocks-524288.o%J
#BSUB -e blocks-524288.e%J
#BSUB -W 04:00
#BSUB -nnodes 1
#BSUB -x
##BSUB -j_exclusive=yes

export ROOT=$HOME/repos/spmvs
export METHOD=kokkos-kernels
export TMP_DIR=/vscratch1/cwpears
export OUT_DIR=$ROOT/scripts/vortex

. $ROOT/$METHOD/load-env.sh

set -eou pipefail

date

export KOKKOS_NUM_DEVICES=1
export CUDA_LAUNCH_BLOCKING=0

. $ROOT/$METHOD/scripts/vortex/common.sh

# any matrix with the same block structure should be the same
# don't need faded with fill because it's the same as non-fade
block_mats=\
"
$ROOT/static/block-constant_524288_*_1.0_*_0_bs16.mtx \
$ROOT/static/block-diagonal-constant_524288_1.0_0_0_bs16.mtx \
$ROOT/static/block-diagonal-variable_524288_*_1.0_*_0_pad16_fill16.mtx \
$ROOT/static/block-variable_524288_*_*_1.0_*_0_pad16_fill16.mtx \
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
