#include "KokkosSparse_IOUtils.hpp"
#include "KokkosSparse_spmv.hpp"

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;

template <typename MatrixType> MatrixType read_crs(const std::string &path) {
  std::cerr << __FILE__ << ":" << __LINE__ << ": read " << path << " ...\n";
  return KokkosSparse::Impl::read_kokkos_crst_matrix<MatrixType>(path.c_str());
}

template <typename MatrixType> MatrixType read_bsr(const std::string &path, const int blockSize) {
  typedef KokkosSparse::CrsMatrix<typename MatrixType::non_const_value_type,
                                  typename MatrixType::non_const_ordinal_type,
                                  typename MatrixType::device_type>
      CrsType;
  std::cerr << __FILE__ << ":" << __LINE__ << ": read " << path << " ...\n";
  CrsType crs = KokkosSparse::Impl::read_kokkos_crst_matrix<CrsType>(path.c_str());
  return MatrixType(crs, blockSize);
}

inline std::string get_basename(const std::string &path) {
  return path.substr(path.find_last_of("/\\") + 1);
}

template <typename YMatrix, typename XMatrix, typename AMatrix>
double bench_single(const int nIters, const int nWarmup,
                    KokkosKernels::Experimental::Controls &controls, const char mode[],
                    const typename AMatrix::non_const_value_type &alpha, AMatrix &a, XMatrix &x,
                    const typename AMatrix::non_const_value_type &beta, YMatrix &y) {

  Kokkos::fence();
  for (int i = 0; i < nWarmup; ++i) {
    KokkosSparse::spmv(controls, KokkosSparse::NoTranspose, alpha, a, x, beta, y);
  }
  Kokkos::fence();
  auto start = Clock::now();
  for (int i = 0; i < nIters; ++i) {
    KokkosSparse::spmv(controls, KokkosSparse::NoTranspose, alpha, a, x, beta, y);
  }
  Kokkos::fence();
  Duration elapsed = Clock::now() - start;
  return elapsed.count() / nIters * 1e6;
}
