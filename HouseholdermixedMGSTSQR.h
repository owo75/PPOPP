#pragma once
#include <cublas_api.h>
#include <cublas_v2.h>

#include <cassert>
#include <cmath>
#include <type_traits>
#include <cuda_runtime.h>

#include "TSQRCommon.h"


template <typename T>
__global__ void tsqr_householder_block_kernel(int m, int n, T *A, int lda, T *R,
                                              int ldr) {
    shared_memory<T> shared;
    T *shared_A = shared.get_pointer();  

    const int ldsa = TSQR_BLOCK_SIZE;    
    const int tidx = threadIdx.x;        
    const int tidy = threadIdx.y;       
    const int bidx = blockIdx.x;        

    const int block_size = min(TSQR_BLOCK_SIZE, m - bidx * TSQR_BLOCK_SIZE);

    A += bidx * TSQR_BLOCK_SIZE;
    R += bidx * n;

    const int num_data_col = (n + TSQR_BLOCK_DIM_Y - 1) / TSQR_BLOCK_DIM_Y;

    T acc[TSQR_NUM_DATA_ROW];
    T q[TSQR_NUM_DATA_ROW];

#pragma unroll
    for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
        int row_idx = tidx + k * TSQR_BLOCK_DIM_X;  
        if (row_idx < block_size) {
            for (int h = 0; h < num_data_col; ++h) {
                int col_idx = tidy + h * TSQR_BLOCK_DIM_Y;  
                if (col_idx < n) {
                    shared_A[row_idx + col_idx * ldsa] = A[row_idx + col_idx * lda];
                }
            }
        }
    }

    __syncthreads();

    for (int cols = 0; cols < n; ++cols) {
        T nu = 0.0;

        if (tidy == cols % TSQR_BLOCK_DIM_Y) {
#pragma unroll
            for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
                acc[k] = 0.0;
                int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
                if (row_idx >= cols && row_idx < block_size) {
                    q[k] = shared_A[row_idx + cols * ldsa];
                    acc[k] = q[k] * q[k];
                }
                nu += acc[k];
            }
            T norm_x_sq = warpAllReduceSum(nu);
            T norm_x = sqrt(norm_x_sq);
            T scale = (norm_x > 0) ? static_cast<T>(1.0) / norm_x : static_cast<T>(0.0);
#pragma unroll
            for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
                int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
                if (row_idx >= cols && row_idx < block_size) {
                    q[k] *= scale;
                }
            }

            int owner_lane = cols % TSQR_BLOCK_DIM_X;
            int owner_off = cols / TSQR_BLOCK_DIM_X;
            T u1 = 0;
            if (tidx == owner_lane) {
                q[owner_off] += (q[owner_off] >= 0) ? 1 : -1;
                u1 = q[owner_off];
                R[cols + cols * ldr] = (u1 >= 0) ? -norm_x : norm_x;
            }
            u1 = __shfl_sync(0xFFFFFFFF, u1, owner_lane);

            scale = (u1 != 0) ? static_cast<T>(1.0) / sqrt(fabs(u1))
                              : static_cast<T>(0.0);
#pragma unroll
            for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
                int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
                if (row_idx >= cols && row_idx < block_size) {
                    shared_A[row_idx + cols * ldsa] = q[k] * scale;
                }
            }
        }

        __syncthreads();

        for (int h = 0; h < num_data_col; ++h) {
            int opCols = tidy + h * TSQR_BLOCK_DIM_Y;
            if (cols < opCols && opCols < n) {
                nu = 0.0;
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
                    acc[k] = 0.0;
                    int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
                    if (row_idx >= cols && row_idx < block_size) {
                        q[k] = shared_A[row_idx + cols * ldsa];
                        acc[k] = q[k] * shared_A[row_idx + opCols * ldsa];
                    }
                    nu += acc[k];
                }
                T utx = warpAllReduceSum(nu);
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
                    int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
                    if (row_idx >= cols && row_idx < block_size) {
                        shared_A[row_idx + opCols * ldsa] -= utx * q[k];
                    }
                }
            }
        }

        __syncthreads();
    }

    const int rRowDataNum = (n + (TSQR_BLOCK_DIM_X - 1)) / TSQR_BLOCK_DIM_X;
    for (int h = 0; h < num_data_col; h++) {
        int opCols = tidy + h * TSQR_BLOCK_DIM_Y;
        if (opCols >= n) continue;

#pragma unroll
        for (int k = 0; k < rRowDataNum; k++) {
            int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
            if (row_idx < block_size && row_idx < opCols) {
                R[row_idx + opCols * ldr] = shared_A[row_idx + opCols * ldsa];
            } else if (row_idx == opCols && row_idx < n) {
            } else if (row_idx > opCols && row_idx < n) {
                R[row_idx + opCols * ldr] = static_cast<T>(0.0);
            }
        }
    }

    __syncthreads();

    for (int h = 0; h < num_data_col; h++) {
        int opCols = tidy + h * TSQR_BLOCK_DIM_Y;
        if (opCols >= n) continue;

#pragma unroll
        for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
            int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
            q[k] = (row_idx == opCols) ? static_cast<T>(1.0) : static_cast<T>(0.0);
        }

        __syncwarp();

        for (int cols = n - 1; cols >= 0; cols--) {
            if (opCols >= cols) {
                T nu = 0.0;
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    acc[k] = 0.0;
                    int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
                    if (row_idx >= cols && row_idx < block_size) {
                        acc[k] = shared_A[row_idx + cols * ldsa] * q[k];
                        nu += acc[k];
                    }
                }
                T utq = warpAllReduceSum(nu);
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
                    if (row_idx >= cols && row_idx < block_size) {
                        q[k] -= utq * shared_A[row_idx + cols * ldsa];
                    }
                }

                __syncwarp();
            }
        }

#pragma unroll
        for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
            int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
            if (row_idx < block_size) {
                A[row_idx + opCols * lda] = q[k];
            }
        }
    }
}


template __global__ void tsqr_householder_block_kernel<float>(int m, int n, float *A,
                                                              int lda, float *R, int ldr);
template __global__ void tsqr_householder_block_kernel<double>(int m, int n, double *A,
                                                               int lda, double *R, int ldr);


template <typename T>
__global__ void tsqr_mgs_block_kernel(int m, int n, T *A, int lda, T *R, int ldr) {
    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T *Qblk = reinterpret_cast<T *>(smem);               
    T *Rblk = Qblk + TSQR_BLOCK_SIZE * n;                

    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int bidx = blockIdx.x;
    const int ldq = TSQR_BLOCK_SIZE;

    const int block_size = min(TSQR_BLOCK_SIZE, m - bidx * TSQR_BLOCK_SIZE);


    A += bidx * TSQR_BLOCK_SIZE;
    R += bidx * n;

    const int num_data_col = (n + TSQR_BLOCK_DIM_Y - 1) / TSQR_BLOCK_DIM_Y;
    T acc[TSQR_NUM_DATA_ROW];

#pragma unroll
    for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
        int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
        if (row_idx < block_size) {
            for (int h = 0; h < num_data_col; ++h) {
                int col_idx = tidy + h * TSQR_BLOCK_DIM_Y;
                if (col_idx < n) {
                    Qblk[row_idx + col_idx * ldq] = A[row_idx + col_idx * lda];
                }
            }
        }
    }

    if (tidx < n && tidy == 0) {
        for (int j = 0; j < n; ++j) Rblk[tidx + j * n] = 0.0;
    }
    __syncthreads();

    for (int cols = 0; cols < n; ++cols) {
        __shared__ int column_zero;
        if (tidx == 0 && tidy == 0) column_zero = 0;

        if (tidy == cols % TSQR_BLOCK_DIM_Y) {
            T acc_norm = 0.0;
#pragma unroll
            for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
                int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
                if (row_idx < block_size) {
                    T val = Qblk[row_idx + cols * ldq];
                    acc_norm += val * val;
                }
            }
            T norm_sq = warpAllReduceSum(acc_norm);
            T norm = sqrt(norm_sq);
            T eps = static_cast<T>(0.0);
            int is_zero = norm <= eps;
            T inv_norm = is_zero ? static_cast<T>(0.0) : static_cast<T>(1.0) / norm;
            if (tidx == 0) {
                column_zero = is_zero;
                for (int j = cols; j < n; ++j) {
                    Rblk[cols + j * n] = is_zero ? static_cast<T>(0.0) : Rblk[cols + j * n];
                }
                Rblk[cols + cols * n] = is_zero ? static_cast<T>(0.0) : norm;
            }
#pragma unroll
            for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
                int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
                if (row_idx < block_size) {
                    Qblk[row_idx + cols * ldq] *= inv_norm;
                }
            }
        }

        __syncthreads();

        for (int h = 0; h < num_data_col; ++h) {
            int opCols = tidy + h * TSQR_BLOCK_DIM_Y;
            if (cols < opCols && opCols < n) {
                if (column_zero) {
                    if (tidx == 0) Rblk[cols + opCols * n] = 0.0;
                    continue;
                }
                T accp = 0.0;
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
                    int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
                    if (row_idx < block_size) {
                        accp += Qblk[row_idx + cols * ldq] *
                                Qblk[row_idx + opCols * ldq];
                    }
                }
                T r_val = warpAllReduceSum(accp);
                if (tidx == 0) Rblk[cols + opCols * n] = r_val;
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
                    int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
                    if (row_idx < block_size) {
                        Qblk[row_idx + opCols * ldq] -=
                            r_val * Qblk[row_idx + cols * ldq];
                    }
                }

                __syncthreads();

                accp = 0.0;
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
                    int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
                    if (row_idx < block_size) {
                        accp += Qblk[row_idx + cols * ldq] *
                                Qblk[row_idx + opCols * ldq];
                    }
                }
                T r_val2 = warpAllReduceSum(accp);
                if (tidx == 0) Rblk[cols + opCols * n] += r_val2;
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
                    int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
                    if (row_idx < block_size) {
                        Qblk[row_idx + opCols * ldq] -=
                            r_val2 * Qblk[row_idx + cols * ldq];
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
        int row_idx = tidx + k * TSQR_BLOCK_DIM_X;
        if (row_idx < block_size) {
            for (int h = 0; h < num_data_col; ++h) {
                int col_idx = tidy + h * TSQR_BLOCK_DIM_Y;
                if (col_idx < n) {
                    A[row_idx + col_idx * lda] = Qblk[row_idx + col_idx * ldq];
                }
            }
        }
    }
    if (tidx < n && tidy == 0) {
        int i = tidx;
        for (int j = 0; j < n; ++j) {
            R[i + j * ldr] = (i <= j) ? Rblk[i + j * n] : static_cast<T>(0.0);
        }
    }
}

template __global__ void tsqr_mgs_block_kernel<float>(int m, int n, float *A, int lda,
                                                      float *R, int ldr);
template __global__ void tsqr_mgs_block_kernel<double>(int m, int n, double *A, int lda,
                                                       double *R, int ldr);

template <typename T>
void tsqr_mgs_func(cublasHandle_t cublas_handle, cudaDataType_t cuda_data_type,
                   cublasComputeType_t cublas_compute_type, int share_memory_size,
                   int m, int n, T *A, int lda, T *R, int ldr, T *workQ,
                   int ldworkQ, T *workR, int ldworkR) {
    dim3 blockDim(TSQR_BLOCK_DIM_X, TSQR_BLOCK_DIM_Y);


    if (m <= TSQR_BLOCK_SIZE) {
        tsqr_mgs_block_kernel<T>
            <<<1, blockDim, share_memory_size>>>(m, n, A, lda, R, ldr);
        cudaDeviceSynchronize();
        return;
    }

    const int blockNum = (m + TSQR_BLOCK_SIZE - 1) / TSQR_BLOCK_SIZE;

    tsqr_mgs_block_kernel<T>
        <<<blockNum, blockDim, share_memory_size>>>(m, n, A, lda, workR, ldworkR);

    tsqr_mgs_func<T>(cublas_handle, cuda_data_type, cublas_compute_type,
                     share_memory_size, blockNum * n, n, workR, ldworkR, R, ldr,
                     workQ + n * ldworkQ, ldworkQ, workR + n * ldworkR, ldworkR);

    T tone = 1.0, tzero = 0.0;
    const long long strideA = static_cast<long long>(TSQR_BLOCK_SIZE);
    const long long strideB = static_cast<long long>(n);  
    cublasGemmStridedBatchedEx(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, TSQR_BLOCK_SIZE, n, n, &tone,
        A, cuda_data_type, lda, strideA, workR, cuda_data_type, ldworkR, strideB,
        &tzero, A, cuda_data_type, lda, strideA, m / TSQR_BLOCK_SIZE,
        cublas_compute_type, CUBLAS_GEMM_DEFAULT);

    int mm = m % TSQR_BLOCK_SIZE;
    if (mm > 0) {
        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, mm, n, n, &tone,
                     A + (m - mm), cuda_data_type, lda,
                     workR + (m / TSQR_BLOCK_SIZE * n), cuda_data_type, ldworkR,
                     &tzero, A + (m - mm), cuda_data_type, lda, cublas_compute_type,
                     CUBLAS_GEMM_DEFAULT);
    }
}

template void tsqr_mgs_func<float>(cublasHandle_t cublas_handle,
                                   cudaDataType_t cuda_data_type,
                                   cublasComputeType_t cublas_compute_type,
                                   int share_memory_size, int m, int n, float *A,
                                   int lda, float *R, int ldr, float *workQ,
                                   int ldworkQ, float *workR, int ldworkR);
template void tsqr_mgs_func<double>(cublasHandle_t cublas_handle,
                                    cudaDataType_t cuda_data_type,
                                    cublasComputeType_t cublas_compute_type,
                                    int share_memory_size, int m, int n, double *A,
                                    int lda, double *R, int ldr, double *workQ,
                                    int ldworkQ, double *workR, int ldworkR);

template <typename T>
__global__ void tsqr_mgs_stack_kernel(int m, int n, T *A, int lda, T *R, int ldr) {

    int tid = threadIdx.x;
    extern __shared__ unsigned char smem[];
    T *sdata = reinterpret_cast<T *>(smem);  

    const T eps = (sizeof(T) == sizeof(float)) ? static_cast<T>(1e-4)
                                               : static_cast<T>(1e-13);

    for (int j = tid; j < n * n; j += blockDim.x) {
        R[j] = static_cast<T>(0);
    }
    __syncthreads();

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < j; ++i) {
            T local_sum = static_cast<T>(0);
            for (int row = tid; row < m; row += blockDim.x) {
                T qi = A[row + i * lda];  
                T vj = A[row + j * lda];  
                local_sum += qi * vj;
            }
            sdata[tid] = local_sum;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    sdata[tid] += sdata[tid + stride];
                }
                __syncthreads();
            }
            T dot = sdata[0];  


            if (tid == 0) {
                R[i + j * ldr] = dot;
            }


            for (int row = tid; row < m; row += blockDim.x) {
                T qi = A[row + i * lda];
                A[row + j * lda] -= dot * qi;
            }
            __syncthreads();
        }
        T local_norm_sq = static_cast<T>(0);
        for (int row = tid; row < m; row += blockDim.x) {
            T vj = A[row + j * lda];
            local_norm_sq += vj * vj;
        }
        sdata[tid] = local_norm_sq;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }
        T norm_sq = sdata[0];
        T norm = sqrt(norm_sq);
        __syncthreads();

        if (norm < eps) {
            if (tid == 0) {
                R[j + j * ldr] = static_cast<T>(0);
            }
            __syncthreads();
            for (int row = tid; row < m; row += blockDim.x) {
                A[row + j * lda] = static_cast<T>(0);
            }
            __syncthreads();
        } else {
            if (tid == 0) {
                R[j + j * ldr] = norm;
            }
            __syncthreads();
            T inv = static_cast<T>(1) / norm;
            for (int row = tid; row < m; row += blockDim.x) {
                A[row + j * lda] *= inv;
            }
            __syncthreads();
        }
    }
}

template __global__ void tsqr_mgs_stack_kernel<float>(int m, int n, float *A, int lda,
                                                      float *R, int ldr);
template __global__ void tsqr_mgs_stack_kernel<double>(int m, int n, double *A, int lda,
                                                       double *R, int ldr);


template <typename T>
void tsqr_hh_mgs(cublasHandle_t cublas_handle, int m, int n, T *A, int lda, T *R,
                 int ldr, T *work, int ldwork) {
    assert(m >= n);
    assert(ldwork >= n);
    static_assert(TSQR_BLOCK_SIZE % TSQR_BLOCK_DIM_X == 0);
    static_assert(TSQR_BLOCK_DIM_X * TSQR_NUM_DATA_ROW == TSQR_BLOCK_SIZE);

    cudaDataType_t cuda_data_type;
    cublasComputeType_t cublas_compute_type;
    if (std::is_same<T, double>::value) {
        cuda_data_type = CUDA_R_64F;
        cublas_compute_type = CUBLAS_COMPUTE_64F;
    } else if (std::is_same<T, float>::value) {
        cuda_data_type = CUDA_R_32F;
        cublas_compute_type = CUBLAS_COMPUTE_32F;
    } else {
        cuda_data_type = CUDA_R_16F;
        cublas_compute_type = CUBLAS_COMPUTE_16F;
    }

    int hh_share_memory_size = TSQR_BLOCK_SIZE * n * sizeof(T);
    cudaFuncSetAttribute(tsqr_householder_block_kernel<T>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         hh_share_memory_size);

    int mgs_share_memory_size = (TSQR_BLOCK_SIZE * n + n * n) * sizeof(T);
    cudaFuncSetAttribute(tsqr_mgs_block_kernel<T>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         mgs_share_memory_size);

    dim3 blockDim(TSQR_BLOCK_DIM_X, TSQR_BLOCK_DIM_Y);
    const int blockNum = (m + TSQR_BLOCK_SIZE - 1) / TSQR_BLOCK_SIZE;
    const int ld_stack = blockNum * n;
    assert(ldwork >= ld_stack);

    T *bufR = work;

    tsqr_householder_block_kernel<T>
        <<<blockNum, blockDim, hh_share_memory_size>>>(m, n, A, lda, bufR, ld_stack);
    cudaDeviceSynchronize();

    {
        const int m_top = blockNum * n;    
        const int n_top = n;
        const int lda_stack = ld_stack;   
        const int ldr_top = ldr;          

       
        int threads = 256;
        size_t shmem_size = threads * sizeof(T);

        tsqr_mgs_stack_kernel<T>
            <<<1, threads, shmem_size>>>(m_top, n_top, bufR, lda_stack, R, ldr_top);
        cudaDeviceSynchronize();
    }

    const T tone = 1.0, tzero = 0.0;
    int full_blocks = m / TSQR_BLOCK_SIZE;
    if (full_blocks > 0) {
        const long long strideA = TSQR_BLOCK_SIZE;  
        const long long strideB = n;               
        cublasGemmStridedBatchedEx(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, TSQR_BLOCK_SIZE, n, n, &tone,
            A, cuda_data_type, lda, strideA,
            bufR, cuda_data_type, ld_stack, strideB,
            &tzero,
            A, cuda_data_type, lda, strideA, full_blocks,
            cublas_compute_type, CUBLAS_GEMM_DEFAULT);
    }

    int mm = m % TSQR_BLOCK_SIZE;
    if (mm > 0) {
        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, mm, n, n, &tone,
                     A + (m - mm), cuda_data_type, lda,
                     bufR + full_blocks * n, cuda_data_type, ld_stack,
                     &tzero,
                     A + (m - mm), cuda_data_type, lda,
                     cublas_compute_type, CUBLAS_GEMM_DEFAULT);
    }
}

template void tsqr_hh_mgs<float>(cublasHandle_t cublas_handle, int m, int n,
                                 float *A, int lda, float *R, int ldr, float *work,
                                 int ldwork);
template void tsqr_hh_mgs<double>(cublasHandle_t cublas_handle, int m, int n,
                                  double *A, int lda, double *R, int ldr,
                                  double *work, int ldwork);
