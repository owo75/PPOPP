#ifndef PPOPP_H
#define PPOPP_H

#include <cuda_runtime.h>
#include <curand.h>
#include <type_traits>
#include <thrust/device_vector.h>
#include <cstddef>

template <typename T>
__global__ void scale_columns_kernel(
    T* M_out,
    const T* M_in,
    const T* vec,
    size_t rows,
    size_t cols,
    size_t ld_out,
    size_t ld_in)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        T scale_factor = vec[col];
        M_out[col * ld_out + row] = M_in[col * ld_in + row] * scale_factor;
    }
}

template <typename T>
void generateNormalMatrix(curandGenerator_t& gen, thrust::device_vector<T>& d_vec, size_t m, size_t n)
{
    T* raw_ptr = thrust::raw_pointer_cast(d_vec.data());

    if constexpr (std::is_same_v<T, float>) {
        curandGenerateNormal(gen, raw_ptr, m * n, 0.0f, 1.0f);
    } else if constexpr (std::is_same_v<T, double>) {
        curandGenerateNormalDouble(gen, raw_ptr, m * n, 0.0, 1.0);
    }
}

template <typename T>
__global__ void extractDiagonal(T* diag_vector, const T* R_block, size_t ld_R, size_t ost, size_t block_width)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < block_width) {
        diag_vector[i] = R_block[(i + ost) * ld_R + (i + ost)];
    }
}

template <typename T>
__global__ void add_identity_diagonal_kernel(
    T* matrix,
    long int rows,
    long int cols,
    long int lda,
    T alpha = static_cast<T>(1.0))
{
    long int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && i < cols) {
        matrix[i * lda + i] += alpha;
    }
}

template <typename T>
__global__ void extract_L_factor_kernel(T* L_out,
                                        long int lda_out,
                                        const T* LU_in,
                                        long int lda_in,
                                        long int rows,
                                        long int cols)
{
    long int c = blockIdx.x * blockDim.x + threadIdx.x;
    long int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < rows && c < cols) {
        long int out_idx = c * lda_out + r;

        if (c >= rows) {
            L_out[out_idx] = (r == c) ? static_cast<T>(1.0) : static_cast<T>(0.0);
            return;
        }

        if (r < c) {
            L_out[out_idx] = static_cast<T>(0.0);
        } else if (r == c) {
            L_out[out_idx] = static_cast<T>(1.0);
        } else {
            long int in_idx = c * lda_in + r;
            L_out[out_idx] = LU_in[in_idx];
        }
    }
}

template <typename T>
__global__ void select_columns_kernel(T* dest,
                                      const int ld_dest,
                                      const T* src,
                                      const int ld_src,
                                      const int* indices,
                                      const int rows,
                                      const int num_new_cols)
{
    int dest_col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && dest_col < num_new_cols) {
        int src_col = indices[dest_col];

        size_t src_idx = (size_t)row + (size_t)src_col * ld_src;
        size_t dest_idx = (size_t)row + (size_t)dest_col * ld_dest;

        dest[dest_idx] = src[src_idx];
    }
}

template <typename T>
__global__ void select_rows_kernel(
    T* dest,
    const int ld_dest,
    const T* src,
    const int ld_src,
    const int* indices,
    const int rows_keep,
    const int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_keep && col < cols) {
        int src_row = indices[row];
        size_t src_idx = (size_t)col * ld_src + src_row;
        size_t dest_idx = (size_t)col * ld_dest + row;
        dest[dest_idx] = src[src_idx];
    }
}

template <typename T>
__global__ void extract_upper_triangle_kernel(
    T* dest,
    const int ld_dest,
    const T* src,
    const int ld_src,
    const int rows,
    const int cols)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        size_t src_idx = (size_t)i + (size_t)j * ld_src;
        size_t dest_idx = (size_t)i + (size_t)j * ld_dest;

        if (i <= j) {
            dest[dest_idx] = src[src_idx];
        } else {
            dest[dest_idx] = static_cast<T>(0.0);
        }
    }
}

#endif
