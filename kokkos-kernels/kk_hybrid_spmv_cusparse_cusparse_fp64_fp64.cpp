/*! \file
    Run a hybrid SpMV

    Output:
    time per nnz (s)
    nnz in read matrix
    real nnz in dense part
    actual nnz in dense part
    nnz in sparse remainder
*/

#include <chrono>
#include <iostream>

#include "base.hpp"
#include "split.hpp"

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;

int main(int argc, char **argv) {

  Kokkos::initialize(argc, argv);
  {

    if (argc < 4) {
      std::cerr << "usage: <exe> block-size block-threshold matrix\n";
      return 1;
    }

    typedef int Ordinal;
    typedef int Offset;
    typedef double AScalar;
    typedef double XScalar;
    typedef double YScalar;
    typedef Kokkos::Cuda Device;

    typedef KokkosSparse::CrsMatrix<AScalar, Ordinal, Device, void, Offset> CrsType;
    typedef KokkosSparse::Experimental::BsrMatrix<AScalar, Ordinal, Device, void, Offset> BsrType;

    typedef Kokkos::View<XScalar **, Kokkos::LayoutLeft, Device> XViewType;
    typedef Kokkos::View<YScalar **, Kokkos::LayoutLeft, Device> YViewType;

    const int blockSize = std::atoi(argv[1]);
    const float blockThresh = std::atof(argv[2]);
    CrsType a = read_crs<CrsType>(argv[3]);

    // split matrix into BsrMatrix with dense parts and CrsMatrix with remainder
    SplitResult<CrsType> split = split_matrix(a, blockSize, blockThresh);
    BsrType dense = BsrType(split.dense, blockSize);
    CrsType remainder = split.sparse;

    // the dense matrix will be >= the size of the remainder (since it is padded out to blockSize)
    // the multivectors should therefore be that large and the remainder will operate on a subview

    // make multivectors
    const Ordinal K = 16;
    const Ordinal M = dense.numRows() * dense.blockDim();
    const Ordinal N = dense.numCols() * dense.blockDim();

    XViewType x("x", N, K);
    YViewType y("y", M, K);

    // sparse remainder multivectors.
    // both dense and sparse accumulate into the same y.
    // the dense matrix may be padded, so sparse is just a subview of that
    auto x_sp =
        Kokkos::subview(x, Kokkos::make_pair(0, remainder.numCols()), Kokkos::make_pair(0, K));
    auto y_sp =
        Kokkos::subview(y, Kokkos::make_pair(0, remainder.numRows()), Kokkos::make_pair(0, K));

    const AScalar alpha = 0.1;
    const YScalar beta = -0.1;

    // invoke hybrid SpMV
    //  original spmv is y = by + aAx
    //  we do y = (alpha)(sparse)(x) + (beta)(y)
    //        y = (alpha)(dense) (x) + (1)y
    const int niters = 1000;
    const int nwarmup = 5;
    KokkosKernels::Experimental::Controls denseCtls, remCtls;
    Kokkos::fence();
    for (int i = 0; i < nwarmup; ++i) {
      KokkosSparse::spmv(remCtls, KokkosSparse::NoTranspose, alpha, remainder, x_sp, beta, y_sp);
      KokkosSparse::spmv(denseCtls, KokkosSparse::NoTranspose, alpha, dense, x, YScalar(1), y);
    }
    Kokkos::fence();
    auto start = Clock::now();
    for (int i = 0; i < niters; ++i) {
      KokkosSparse::spmv(remCtls, KokkosSparse::NoTranspose, alpha, remainder, x_sp, beta, y_sp);
      KokkosSparse::spmv(denseCtls, KokkosSparse::NoTranspose, alpha, dense, x, YScalar(1), y);
    }
    Kokkos::fence();
    Duration hybridElapsed = Clock::now() - start;

    // individual SpMVs
    remMicros = bench_single(niters, nwarmup, remCtls, KokkosSparse::NoTranspose, alpha, remainder, x_sp, beta, y_sp);
    denseMicros = bench_single(niters, nwarmup, denseCtls, KokkosSparse::NoTranspose, alpha, dense, x, YScalar(1), y);

    // clang-format off
    std::cout << hybridElapsed.count() / niters * 1e6 
    << "," << remMicros
    << "," << denseMicros
    << "," << a.nnz() 
    << "," << split.denseNnz 
    << "," << dense.nnz() * dense.blockDim() * dense.blockDim() 
    << "," << remainder.nnz();
    // clang-format on
  }
  Kokkos::finalize();
  return 0;
}
