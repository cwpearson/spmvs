function JSRUN () {
jsrun \
--smpiargs="-disable_gpu_hooks" \
-n 1 \
-r 1 \
-a 1 \
-g 4 \
-c 32 \
-b rs \
-l gpu-cpu \
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

function F1-3 () {
    "$@" | cut -d"," --fields=1,2,3 | tr -d '\n'
}

function F4-7 () {
    "$@" | cut -d"," --fields=2,3,4,5 | tr -d '\n'
}

bsr_exes=\
"
kk-bsr-spmv-cusparse-fp64-fp64 \
kk-bsr-spmv-native-fp16-fp16 \
kk-bsr-spmv-native-fp64-fp64 \
kk-bsr-spmv-tc-fp16-fp16 \
kk-bsr-spmv-tc-fp64-fp64 \
"

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
