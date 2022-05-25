#include <chrono>
#include <iostream>

#include "base.hpp"

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;

int main(int argc, char **argv) {

  Kokkos::initialize(argc, argv);
  {

    if (argc < 3) {
      std::cerr << "usage: <exe> block-size matrix\n";
      return 1;
    }

    typedef int Ordinal;
    typedef int Offset; // cuSPARSE bsr matrix needs int
    typedef double AScalar;
    typedef double XScalar;
    typedef double YScalar;
    typedef Kokkos::Cuda Device;

    typedef KokkosSparse::Experimental::BsrMatrix<AScalar, Ordinal, Device, void, Offset>
        MatrixType;

    typedef Kokkos::View<XScalar **, Kokkos::LayoutLeft, Device> XViewType;
    typedef Kokkos::View<YScalar **, Kokkos::LayoutLeft, Device> YViewType;

    const int blockSize = std::atoi(argv[1]);

    MatrixType a = read_bsr<MatrixType>(argv[2], blockSize);

    // make multivectors
    const Ordinal K = 16;
    const Ordinal M = a.numRows() * blockSize;
    const Ordinal N = a.numCols() * blockSize;

    XViewType x("x", N, K);
    YViewType y("y", M, K);

    const AScalar alpha = 0.1;
    const YScalar beta = -0.1;

    const int niters = 1000;
    KokkosKernels::Experimental::Controls controls;
    controls.setParameter("algorithm", "native");
    Kokkos::fence();
    auto start = Clock::now();
    for (int i = 0; i < niters; ++i) {
      KokkosSparse::spmv(controls, KokkosSparse::NoTranspose, alpha, a, x, beta, y);
    }
    Kokkos::fence();
    Duration elapsed = Clock::now() - start;

    std::cout << elapsed.count() / niters << "\n";
  }
  Kokkos::finalize();
  return 0;
}
