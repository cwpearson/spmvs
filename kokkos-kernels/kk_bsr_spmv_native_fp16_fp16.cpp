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
    typedef Kokkos::Experimental::half_t AScalar;
    typedef Kokkos::Experimental::half_t XScalar;
    typedef Kokkos::Experimental::half_t YScalar;
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
    const int nwarmup = 5;
    KokkosKernels::Experimental::Controls controls;
    controls.setParameter("algorithm", "native");
    double secsPerSpmv =
        bench_single(niters, nwarmup, controls, KokkosSparse::NoTranspose, alpha, a, x, beta, y);

    std::cout << secsPerSpmv;
    std::cout << std::flush;
  }
  Kokkos::finalize();
  return 0;
}
