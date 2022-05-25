project(spmv CXX CUDA)
cmake_minimum_required(VERSION 3.16) # same as kokkos

add_subdirectory(kokkos)
add_subdirectory(kokkos-kernels)

function(target_link_with_kokkos tgt)
target_link_libraries(${tgt} PRIVATE Kokkos::kokkos)
target_link_libraries(${tgt} PRIVATE Kokkos::kokkoskernels)
target_compile_options(${tgt} PRIVATE ${KOKKOS_CUDA_OPTIONS})
target_link_options(${tgt} PRIVATE ${KOKKOS_LINK_OPTIONS})
endfunction()


add_executable(kk-bsr-spmv-cusparse-fp64-fp64 kk_bsr_spmv_cusparse_fp64_fp64.cpp)
target_link_with_kokkos(kk-bsr-spmv-cusparse-fp64-fp64)

add_executable(kk-bsr-spmv-native-fp16-fp16 kk_bsr_spmv_native_fp16_fp16.cpp)
target_link_with_kokkos(kk-bsr-spmv-native-fp16-fp16)

add_executable(kk-bsr-spmv-native-fp64-fp64 kk_bsr_spmv_native_fp64_fp64.cpp)
target_link_with_kokkos(kk-bsr-spmv-native-fp64-fp64)

add_executable(kk-bsr-spmv-tc-fp16-fp16 kk_bsr_spmv_tc_fp16_fp16.cpp)
target_link_with_kokkos(kk-bsr-spmv-tc-fp16-fp16)

add_executable(kk-bsr-spmv-tc-fp64-fp64 kk_bsr_spmv_tc_fp64_fp64.cpp)
target_link_with_kokkos(kk-bsr-spmv-tc-fp64-fp64)


add_executable(kk-crs-spmv-native-fp16-fp16 kk_crs_spmv_native_fp16_fp16.cpp)
target_link_with_kokkos(kk-crs-spmv-native-fp16-fp16)

add_executable(kk-crs-spmv-native-fp64-fp64 kk_crs_spmv_native_fp64_fp64.cpp)
target_link_with_kokkos(kk-crs-spmv-native-fp64-fp64)


add_executable(kk-hybrid-spmv-tc-native-fp16-fp16 kk_hybrid_spmv_tc_native_fp16_fp16.cpp)
target_link_with_kokkos(kk-hybrid-spmv-tc-native-fp16-fp16)

add_executable(kk-hybrid-spmv-tc-native-fp64-fp64 kk_hybrid_spmv_tc_native_fp64_fp64.cpp)
target_link_with_kokkos(kk-hybrid-spmv-tc-native-fp64-fp64)