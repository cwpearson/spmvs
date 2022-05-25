/*! \file
  Split a CrsMatrix into dense blocks + sparse remainder
*/

#pragma once

#pragma GCC diagnostic ignored "-Wswitch-enum"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include "KokkosSparse_BsrMatrix.hpp"
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop

// represents the position of a block in a matrix
template<typename Ordinal>
struct Block {
  Ordinal i;
  Ordinal j;
  Block(Ordinal _i, Ordinal _j) : i(_i), j(_j) {}
  bool operator<(const Block &rhs) const {
    if (i < rhs.i) {
      return true;
    } else if (i == rhs.i) {
      return j < rhs.j;
    } else {
      return false;
    }
  }
};

// coordinate for COO form of a sparse matrix
template <typename Ordinal>
struct Coordinate {
    typedef Ordinal ordinal_type;
    Ordinal i;
    Ordinal j;

    static bool by_ij(const Coordinate &a, const Coordinate &b) noexcept {
        if (a.i < b.i) { // sort by lower i
        return true;
        } else if (a.i > b.i) {
        return false;
        } else {
            return a.j < b.j;
        }
    }

  Coordinate(Ordinal _i, Ordinal _j) : i(_i), j(_j) {}
};

// entry in a COO matrix
template<typename Ordinal, typename Scalar>
struct Entry {
  Ordinal i;
  Ordinal j;
  Scalar e;
  Entry() = default;
  Entry(Ordinal _i, Ordinal _j, Scalar _e) : i(_i), j(_j), e(_e) {}
  static bool by_ij(const Entry &a, const Entry &b) {
    if (a.i < b.i) { // sort by lower i
      return true;
    } else if (a.i > b.i) {
      return false;
    } else {
      if (a.j < b.j) {
        return true;
      } else if (a.j > b.j) {
        return false;
      } else {
        if (a.e != 0) {
          return true;
        } else {
          return false;
        }
      }
    }
  }

  // true if two entries are the same coordinate 
  static bool same_ij(const Entry &a, const Entry &b) noexcept {
    return a.i == b.i && a.j == b.j;
  }
};

// a split matrix with a dense part and a sparse remainder
template<typename Matrix>
struct SplitResult {
  Matrix dense;
  Matrix sparse;
  size_t denseNnz; // how many nnz in dense were present in original matrix
};

/* split a matrix into a BlockCrsMatrix and a CrsMatrix
*/
template <typename CrsMatrix>
SplitResult<CrsMatrix> split_matrix(CrsMatrix m, int blockSize, float density) {

  typedef typename CrsMatrix::ordinal_type Ordinal;
  typedef typename CrsMatrix::value_type Scalar;
  typedef typename CrsMatrix::non_const_size_type Offset;

  typedef Kokkos::View<const Offset*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> UnmanagedRowmap;
  typedef Kokkos::View<const Ordinal*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> UnmanagedEntries;
  typedef Kokkos::View<const Scalar*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> UnmanagedValues;

  typedef Block<Ordinal> Block;
  typedef Entry<Ordinal, Scalar> Entry;

  SplitResult<CrsMatrix> result;

  std::cerr << "copy CSR data to host\n";
  typename CrsMatrix::row_map_type::HostMirror row_map_h = Kokkos::create_mirror_view(m.graph.row_map);
  typename CrsMatrix::index_type::HostMirror entries_h = Kokkos::create_mirror_view(m.graph.entries);
  typename CrsMatrix::values_type::HostMirror values_h = Kokkos::create_mirror_view(m.values);
  Kokkos::deep_copy(row_map_h, m.graph.row_map);
  Kokkos::deep_copy(entries_h, m.graph.entries);
  Kokkos::deep_copy(values_h, m.values);
  std::cerr << "row 0 nnz: " << row_map_h(1) - row_map_h(0) << "\n";
  std::cerr << "row 1 nnz: " << row_map_h(2) - row_map_h(1) << "\n";
  std::cerr << "row 2 nnz: " << row_map_h(3) - row_map_h(2) << "\n";

  std::cerr << "determine block densities\n";
  std::map<Block, size_t> blocks;
  for (size_t row = 0; row < row_map_h.size() - 1; ++row) {
      for (Offset ci = row_map_h(row); ci < row_map_h(row+1); ++ci) {
          Ordinal col = entries_h(ci);
          Block key(row / blockSize * blockSize, col / blockSize * blockSize);
          auto p = blocks.insert(std::make_pair(key, 0));
          p.first->second += 1;
      }
  }
  std::cerr << blocks.size() << " blocks\n";

  std::cerr << "remove low-density blocks\n";
  {
    std::vector<Block> toErase;
    for (auto kv : blocks) {
      if (float(kv.second) / (blockSize * blockSize) < density) {
        toErase.push_back(kv.first);
      }
    }
    for (Block key : toErase) {
      blocks.erase(key);
    }
  }
  std::cerr << blocks.size() << " blocks\n";

  std::cerr << "build dense CrsMatrix\n";
  {
    result.denseNnz = 0;
    // dense num rows / num cols
    Ordinal dnr = (m.numRows() + blockSize - 1) / blockSize * blockSize;
    Ordinal dnc = (m.numCols() + blockSize - 1) / blockSize * blockSize;

    std::vector<Entry> coo;

    {
      // for each explicit entry, we'll have the scalar and column index
      size_t csrEntryBytes = blocks.size() * blockSize * blockSize * (sizeof(Scalar)+sizeof(Ordinal));
      std::cerr << "CSR entries will require " << csrEntryBytes << " B\n";
      if (csrEntryBytes >= 12ull * 1024ull * 1024ull * 1024ull) {
        std::cerr << "ERR: skip: explicit values consume more than 13GiB\n";
        Kokkos::Impl::throw_runtime_exception("explicit values too large");
      }
      size_t nEntries = blocks.size() * blockSize * blockSize + m.nnz();
      std::cerr << "reserve: " << nEntries << " entries (" << nEntries * sizeof(Entry) << " B)\n";
      coo.reserve(nEntries);
    }
    std::cerr << "add blocks zeros\n";
    for (const auto &kv : blocks) {
      Block b = kv.first;
      for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
          coo.push_back(Entry(b.i+i, b.j+j, 0));
        }
      }
    }
    if (blocks.size() * blockSize * blockSize != coo.size()) {
      std::cerr << "unexpected nnz = " << coo.size() << "\n";
      Kokkos::Impl::throw_runtime_exception("unexpected nnz for blocks");
    }

    std::cerr << "add original non-zeros\n";
    result.denseNnz = 0;
    for (size_t row = 0; row < row_map_h.size() - 1; ++row) {
        for (Offset ci = row_map_h(row); ci < row_map_h(row+1); ++ci) {
            if (ci >= entries_h.size()) {
              Kokkos::Impl::throw_runtime_exception("bad access into entries_h");
            }
            if (ci >= values_h.size()) {
              Kokkos::Impl::throw_runtime_exception("bad access into values_h");
            }
            Ordinal col = entries_h(ci);
            if (col < 0 || col > m.numCols()) {
              Kokkos::Impl::throw_runtime_exception("column wrong");
            }
            Scalar val = values_h(ci);

            Block key(row / blockSize * blockSize, col / blockSize * blockSize);
            if (0 != blocks.count(key)) { // this is a dense block
              if (Kokkos::Details::ArithTraits<Scalar>::isNan(val)) {
                std::cerr << __FILE__ << ":" << __LINE__ << "NaN in original matrix\n";
              }
              coo.push_back(Entry(row, col, val));
              result.denseNnz += 1;
            }
        }
    }

    // sort by i,j, and remove zeros if a non-zero exists 
    std::cerr << "sort non-zeros for " << coo.size() <<" entries\n";
    std::sort(coo.begin(), coo.end(), Entry::by_ij);

    std::cerr << "remove duplicate entries\n";
    auto it = std::unique(coo.begin(), coo.end(), Entry::same_ij);
    coo.resize(it - coo.begin());

    if (blocks.size() * blockSize * blockSize != coo.size()) {
      std::cerr << "unexpected nnz. got " << coo.size() << " expected " << blocks.size() * blockSize * blockSize << "\n";
      Kokkos::Impl::throw_runtime_exception("unexpected nnz for blocks after resize");
    }

    std::cerr << "build up CRS format for " << coo.size() <<" entries\n";

    std::vector<Offset> rowMap;
    std::vector<Ordinal> colInd;
    std::vector<Scalar> val;
    {
      std::cerr << "reserve " << coo.size() * sizeof(Ordinal) << "\n";
      colInd.reserve(coo.size());
      std::cerr << "reserve " << coo.size() * sizeof(Scalar) << "\n";
      val.reserve(coo.size());
    }

    for (Entry &e : coo) {
        while(rowMap.size() < size_t(e.i+1)) { // catch empty rows
            rowMap.push_back(colInd.size());
        }
        colInd.push_back(e.j);
        val.push_back(e.e);
    }
    // possibly empty rows at end of matrix
    while(rowMap.size() <= size_t(dnr)) {
        rowMap.push_back(colInd.size());
    }

    std::cerr << "copy CSR data to device\n";
    typename CrsMatrix::row_map_type::non_const_type sparseRowMap("", rowMap.size());
    Kokkos::deep_copy(sparseRowMap, UnmanagedRowmap(rowMap.data(), rowMap.size()));
    typename CrsMatrix::index_type::non_const_type sparseCols("", colInd.size());
    Kokkos::deep_copy(sparseCols, UnmanagedEntries(colInd.data(), colInd.size()));
    typename CrsMatrix::values_type::non_const_type sparseVals("", val.size());
    Kokkos::deep_copy(sparseVals, UnmanagedValues(val.data(), val.size()));

    std::cerr << "convert dense to CrsMatrix\n";
    CrsMatrix dense("dense", dnr, dnc, sparseVals.size(), sparseVals, sparseRowMap, sparseCols);
    result.dense = dense;
  }

  std::cerr << "Build sparse matrix\n";
  {
    std::vector<Entry> coo;

    std::cerr << "add original non-zeros\n";
    for (size_t row = 0; row < row_map_h.size() - 1; ++row) {
        for (Offset ci = row_map_h(row); ci < row_map_h(row+1); ++ci) {
            Ordinal col = entries_h(ci);
            Scalar val = values_h(ci);
            Block key(row / blockSize * blockSize, col / blockSize * blockSize);
            if (0 == blocks.count(key)) {
              coo.push_back(Entry(row, col, val));
              if (Kokkos::Details::ArithTraits<Scalar>::isNan(val)) {
                std::cerr << __FILE__ << ":" << __LINE__ << ": " <<  "NaN in original matrix\n";
              }
              if (ci < 5) {
                std::cerr << __FILE__ << ":" << __LINE__ << ": " << ci << "th entry is " << val << "\n";
              }
            }
        }
    }

    std::cerr << "sort by ij\n";
    std::sort(coo.begin(), coo.end(), Entry::by_ij);

    std::cerr << "build up CRS format\n";
    std::vector<Offset> rowMap;
    std::vector<Ordinal> colInd;
    std::vector<Scalar> val;
    for (Entry &e : coo) {
        while(rowMap.size() < size_t(e.i+1)) { // catch empty rows
            rowMap.push_back(colInd.size());
        }
        if (e.j >= m.numCols()) {
          Kokkos::Impl::throw_runtime_exception("column too large");
        }
        if (e.j < 0) {
          Kokkos::Impl::throw_runtime_exception("column too small");
        }
        colInd.push_back(e.j);
        val.push_back(e.e);
    }
    // possibly empty rows at end of matrix
    while(rowMap.size() <= size_t(m.numRows())) {
        rowMap.push_back(colInd.size());
    }

    std::cerr << "copy CSR data to device\n";
    typename CrsMatrix::row_map_type::non_const_type sparseRowMap("", rowMap.size());
    Kokkos::deep_copy(sparseRowMap, UnmanagedRowmap(rowMap.data(), rowMap.size()));
    typename CrsMatrix::index_type::non_const_type sparseCols("", colInd.size());
    Kokkos::deep_copy(sparseCols, UnmanagedEntries(colInd.data(), colInd.size()));
    typename CrsMatrix::values_type::non_const_type sparseVals("", val.size());
    Kokkos::deep_copy(sparseVals, UnmanagedValues(val.data(), val.size()));

    std::cerr << "convert to CrsMatrix\n";
    CrsMatrix sparse("sparse", m.numRows(), m.numCols(), sparseVals.size(), sparseVals, sparseRowMap, sparseCols);
    result.sparse = sparse;    
  }

  return result;
}

