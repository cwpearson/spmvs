#include "KokkosSparse_spmv.hpp"
#include "KokkosKernels_IOUtils.hpp"

template<typename MatrixType>
MatrixType read_crs(const std::string &path) {
    return KokkosKernels::Impl::read_kokkos_crst_matrix<MatrixType>(path.c_str());
}

template<typename MatrixType>
MatrixType read_bsr(const std::string &path, const int blockSize) {
    typedef KokkosSparse::CrsMatrix<
        typename MatrixType::non_const_value_type,
        typename MatrixType::non_const_ordinal_type,
        typename MatrixType::device_type
    > CrsType;
    CrsType crs = KokkosKernels::Impl::read_kokkos_crst_matrix<CrsType>(path.c_str());
    return MatrixType(crs, blockSize);
}

template <typename YScalar, typename XScalar, typename AScalar>
void bench() {

}