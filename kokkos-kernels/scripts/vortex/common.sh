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
