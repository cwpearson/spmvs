#include "KokkosKernels_IOUtils.hpp"
#include "KokkosSparse_spmv.hpp"

template <typename MatrixType> MatrixType read_crs(const std::string &path) {
  std::cerr << __FILE__ << ":" << __LINE__ << ": read " << path << " ...\n";
  return KokkosKernels::Impl::read_kokkos_crst_matrix<MatrixType>(path.c_str());
}

template <typename MatrixType> MatrixType read_bsr(const std::string &path, const int blockSize) {
  typedef KokkosSparse::CrsMatrix<typename MatrixType::non_const_value_type,
                                  typename MatrixType::non_const_ordinal_type,
                                  typename MatrixType::device_type>
      CrsType;
  std::cerr << __FILE__ << ":" << __LINE__ << ": read " << path << " ...\n";
  CrsType crs = KokkosKernels::Impl::read_kokkos_crst_matrix<CrsType>(path.c_str());
  return MatrixType(crs, blockSize);
}

inline std::string get_basename(const std::string &path) {
    return path.substr(path.find_last_of("/\\") + 1);
}

template <typename YScalar, typename XScalar, typename AScalar> void bench() {}