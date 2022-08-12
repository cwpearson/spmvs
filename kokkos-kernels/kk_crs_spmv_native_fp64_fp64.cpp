#include <chrono>
#include <iostream>

#include "base.hpp"

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;

int main(int argc, char **argv) {

  Kokkos::initialize(argc, argv);
  {

    if (argc < 2) {
      std::cerr << "usage: <exe> matrix\n";
      return 1;
    }

    typedef int Ordinal;
    typedef int Offset;
    typedef double AScalar;
    typedef double XScalar;
    typedef double YScalar;
    typedef Kokkos::Cuda Device;

    typedef KokkosSparse::CrsMatrix<AScalar, Ordinal, Device, void, Offset> MatrixType;

    typedef Kokkos::View<XScalar **, Kokkos::LayoutLeft, Device> XViewType;
    typedef Kokkos::View<YScalar **, Kokkos::LayoutLeft, Device> YViewType;

    MatrixType a = read_crs<MatrixType>(argv[1]);

    // make multivectors
    const Ordinal K = 16;
    const Ordinal M = a.numRows();
    const Ordinal N = a.numCols();

    XViewType x("x", N, K);
    YViewType y("y", M, K);

    const AScalar alpha = 0.1;
    const YScalar beta = -0.1;

    const int niters = 1000;
    const int nwarmup = 5;
    KokkosKernels::Experimental::Controls controls;
    controls.setParameter("algorithm", "native");
    double microsPerSpmv =
        bench_single(niters, nwarmup, controls, KokkosSparse::NoTranspose, alpha, a, x, beta, y);

    std::cout << microsPerSpmv;
    std::cout << std::flush;
  }
  Kokkos::finalize();
  return 0;
}
