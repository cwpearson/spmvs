# spmvs
Various SpMV implementations

## kokkos-kernels

**Build with CUSPARSE enabled**.
Not all comparisons are possible, because:
* BSR cuSparse only supports `int` offsets and ordinals.
* BSR cuSparse only supports `LayoutLeft` for multivector X and Y
* BSR cuSparse does not support half precision


| Matrix    | Arithmatic   | Impl.        | Binary |
|-|-|-|-|
| CrsMatrix | fp64 += fp64 | cuSparse     | `kk-crs-spmv-cusparse-fp64-fp64` |
| ...       | ...          | native       | `kk-crs-spmv-native-fp64-fp64`   |
| ...       | ...          | tensor cores | **N/A**                          | 
| ...       | fp16 += fp16 | cuSparse     | `kk-crs-spmv-cusparse-fp64-fp64` |
| ...       | ...          | native       | `kk-crs-spmv-native-fp16-fp16`   |
| ...       | ...          | tensor cores | **N/A**                          | 
| BsrMatrix | fp64 += fp64 | cuSparse     | `kk-bsr-spmv-cusparse-fp64-fp64` |
| ...       | ...          | native       | `kk-bsr-spmv-native-fp64-fp64`   |
| ...       | ...          | tensor cores | `kk-bsr-spmv-tc-fp64-fp64`       | 
| ...       | fp16 += fp16 | cuSparse     | **N/A**                          |
| ...       | ...          | native       | `kk-bsr-spmv-native-fp16-fp16`   |
| ...       |              | tensor cores | `kk-bsr-spmv-tc-fp16-fp16`       |
| Hybrid    | fp64 += fp64 | tensor cores (blocks) + cuSparse (remainder) | `kk-hybrid-spmv-tc-cusparse-fp64-fp64` |
| ...       | ...          | tensor cores (blocks) + native (remainder)   | `kk-hybrid-spmv-tc-native-fp64-fp64`   |
| ...       | fp16 += fp16 | tensor cores (blocks) + cuSparse (remainder) | `kk-hybrid-spmv-tc-cusparse-fp16-fp16` |
| ...       | ...          | tensor cores (blocks) + native (remainder)   | `kk-hybrid-spmv-tc-native-fp16-fp16`   |

## Static Data

* `static/blocks`: block-sparse matrices with block sizes in file names
* `static/points`: point matrices

## Running

* vortex
    * `jsrun --smpiargs="-disable_gpu_hooks" -n 1 -g 1 ./kk-bsr-spmv-base-fp16-fp16 16 ../../static/bs16_block-constant_128_1.0_0.00_0.mtx`
* weaver
    * `bsub -Is -n 40 bash`

## Example Builds

baseline options on Vortex
```
source ../load-env.sh
cmake .. \
-DCMAKE_CXX_COMPILER=${NVCC_WRAPPER} \
-DCMAKE_BUILD_TYPE=Release \
-DKokkos_ENABLE_CUDA=On \
-DKokkos_ENABLE_CUDA_LAMBDA=On \
-DKokkos_ENABLE_HWLOC=Off \
-DKokkosKernels_INST_COMPLEX_FLOAT=OFF \
-DKokkosKernels_INST_DOUBLE=ON \
-DKokkosKernels_INST_FLOAT=OFF \
-DKokkosKernels_INST_HALF=ON \
-DKokkosKernels_INST_OFFSET_INT=ON \
-DKokkosKernels_INST_OFFSET_SIZE_T=OFF \
-DKokkosKernels_INST_LAYOUTRIGHT=OFF \
-DKokkosKernels_INST_MEMSPACE_CUDAUVMSPACE=OFF \
-DKokkosKernels_ENABLE_TPL_CUSPARSE=ON \
-DCMAKE_CXX_FLAGS="-lineinfo" \
-DKokkosKernels_ENABLE_TESTS=OFF 
```

## Additional options



* CUDA on volta: 
```
-DKokkos_ARCH_VOLTA70=On \
```

bsub -W 6:00 -nnodes 1 --shared-launch -Is bash


cmake .. \
-DCMAKE_CXX_COMPILER=${NVCC_WRAPPER} \
-DCMAKE_BUILD_TYPE=Release \
-DKokkos_ENABLE_HWLOC=Off \
-DKokkosKernels_INST_COMPLEX_FLOAT=OFF \
-DKokkosKernels_INST_COMPLEX_DOUBLE=OFF \
-DKokkosKernels_INST_DOUBLE=ON \
-DKokkosKernels_INST_FLOAT=OFF \
-DKokkosKernels_INST_HALF=ON \
-DKokkosKernels_INST_OFFSET_INT=ON \
-DKokkosKernels_INST_OFFSET_SIZE_T=OFF \
-DKokkosKernels_INST_LAYOUTLEFT=ON \
-DKokkosKernels_INST_LAYOUTRIGHT=OFF \
-DCMAKE_CXX_FLAGS="-lineinfo" \
-DKokkosKernels_ENABLE_TESTS=OFF \
-DKokkos_ENABLE_CUDA=On \
-DKokkos_ARCH_VOLTA70=On \
-DKokkos_ENABLE_CUDA_LAMBDA=On \
-DKokkosKernels_ENABLE_TPL_CUSPARSE=ON \
-DKokkosKernels_INST_MEMSPACE_CUDAUVMSPACE=OFF