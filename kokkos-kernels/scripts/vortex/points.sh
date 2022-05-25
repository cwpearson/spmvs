#!/bin/bash
#BSUB -J points
#BSUB -o points.o%J
#BSUB -e points.e%J
#BSUB -W 04:00
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

crs_exes=\
"kk-crs-spmv-native-fp16-fp16 \
kk-crs-spmv-native-fp64-fp64"

hybrid_exes=\
"kk-hybrid-spmv-tc-native-fp16-fp16 \
kk-hybrid-spmv-tc-native-fp64-fp64"

mats=\
"$HOME/suitesparse/Fault_639/Fault_639.mtx \
$HOME/suitesparse/Bump_2911/Bump_2911.mtx"

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

