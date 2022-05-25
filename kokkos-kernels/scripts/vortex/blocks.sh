#!/bin/bash
#BSUB -J Fault_639_overlap_2n_8r
#BSUB -o Fault_639_overlap_2n_8r.o%J
#BSUB -e Fault_639_overlap_2n_8r.e%J
#BSUB -W 00:20
#BSUB -nnodes 1

export PFX=Fault_639_overlap_vortex_8n_32r
export ROOT=$HOME/repos/spmvs
export METHOD=kokkos-kernels
export TMP_DIR=/vscratch1/cwpears
export OUT_DIR=$ROOT/scripts/vortex

. $ROOT/$METHOD/load-env.sh

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
"kk-bsr-spmv-cusparse-fp64-fp64 \
kk-bsr-spmv-native-fp16-fp16 \
kk-bsr-spmv-native-fp64-fp64 \
kk-bsr-spmv-tc-fp16-fp16 \
kk-bsr-spmv-tc-fp64-fp64"

crs_exes=\
"kk-crs-spmv-native-fp16-fp16 \
kk-crs-spmv-native-fp64-fp64"

hybrid_exes=\
"kk-hybrid-spmv-tc-native-fp16-fp16 \
kk-hybrid-spmv-tc-native-fp64-fp64"

block_mats=\
"bs16_block-constant_128_1.0_0.00_0.mtx \
bs16_block-constant_128_1.0_0.00_1.mtx \
bs16_block-constant_128_1.0_0.00_2.mtx \
bs16_block-constant_128_1.0_0.00_3.mtx \
bs16_block-constant_128_1.0_0.00_4.mtx"


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
        JSRUN $ROOT/$METHOD/build/$exe 16 $ROOT/static/blocks/$mat
    done
    for exe in $crs_exes; do
        echo -n ","
        JSRUN $ROOT/$METHOD/build/$exe $ROOT/static/blocks/$mat
    done
    for exe in $hybrid_exes; do
        echo -n ","
        F1 JSRUN $ROOT/$METHOD/build/$exe 16 0.5 $ROOT/static/blocks/$mat
    done
    echo ""
done

