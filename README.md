# spmvs
Various SpMV implementations

## kokkos-kernels

Build with CUSPARSE on, and then use controls to opt-out in some of the binaries

| Matrix    | Arithmatic   | Impl.        | Binary |
|-|-|-|-|
| CrsMatrix | fp64 += fp64 | cuSparse     | |
| ...       | ...          | native       | |
| ...       | ...          | tensor cores | **N/A** | 
| ...       | fp16 += fp16 | cuSparse     | |
| ...       | ...          | native       | |
| ...       | ...          | tensor cores | **N/A** | 
| BsrMatrix | fp64 += fp64 | cuSparse     | |
| ...       | ...          | native       | |
| ...       | ...          | tensor cores |  | 
| ...       | fp16 += fp16 | cuSparse     | |
| ...       | ...          | native       |
| ...       |              | tensor cores |  | 

## Example Builds

baseline options on Vortex
```
source ../load-env.sh
cmake .. \
-DCMAKE_CXX_COMPILER=${NVCC_WRAPPER} \
-DCMAKE_BUILD_TYPE=Release \
-DKokkos_ENABLE_HWLOC=Off \
-DKokkosKernels_INST_COMPLEX_FLOAT=OFF \
-DKokkosKernels_INST_DOUBLE=ON \
-DKokkosKernels_INST_FLOAT=ON \
-DKokkosKernels_INST_HALF=ON \
-DKokkosKernels_INST_OFFSET_INT=OFF \
-DCMAKE_CXX_FLAGS="-lineinfo" \
-DKokkosKernels_ENABLE_TESTS=ON
```

## Additional options

* "threads" backend:
```
-DKokkos_ENABLE_THREADS=On \
-DKokkos_ENABLE_OPENMP=Off \
```

* OpenMP backend:
```
-DKokkos_ENABLE_OPENMP=On \
-DKokkos_ENABLE_THREADS=Off \
```

* CUDA on volta: 
```
-DKokkos_ENABLE_CUDA=On \
-DKokkos_ARCH_VOLTA70=On \
-DKokkos_ENABLE_CUDA_LAMBDA=On \
-DKokkosKernels_INST_MEMSPACE_CUDAUVMSPACE=OFF \
```

bsub -W 6:00 -nnodes 1 --shared-launch -Is bash

cmake .. \
-DCMAKE_CXX_COMPILER=${NVCC_WRAPPER} \
-DCMAKE_BUILD_TYPE=Release \
-DKokkos_ENABLE_HWLOC=Off \
-DKokkosKernels_INST_COMPLEX_FLOAT=OFF \
-DKokkosKernels_INST_DOUBLE=ON \
-DKokkosKernels_INST_FLOAT=ON \
-DKokkosKernels_INST_HALF=ON \
-DKokkosKernels_INST_OFFSET_INT=OFF \
-DCMAKE_CXX_FLAGS="-lineinfo" \
-DKokkosKernels_ENABLE_TESTS=OFF \
-DKokkos_ENABLE_CUDA=On \
-DKokkos_ARCH_VOLTA70=On \
-DKokkos_ENABLE_CUDA_LAMBDA=On \
-DKokkosKernels_ENABLE_TPL_CUSPARSE=ON \
-DKokkosKernels_INST_MEMSPACE_CUDAUVMSPACE=OFF