#include <iostream> 
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <cusolverDn.h>
#include <type_traits>
#include <cmath>
#include <vector>
#include <string>
#include "ppopp.h"
#include <algorithm>
#include <thrust/transform.h>



struct ToFloat {
    __host__ __device__ float operator()(const double& v) const { return static_cast<float>(v); }
};

struct EigenToSigma {
    __host__ __device__ double operator()(const double& l) const {
        double v = (l > 0.0) ? l : 0.0;
        return sqrt(v);
    }
};

__global__ void scatter_columns_kernel(double* dest,
                                       const int ld_dest,
                                       const double* src,
                                       const int ld_src,
                                       const int* indices,
                                       const int rows,
                                       const int num_cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < num_cols) {
        int dst_col = indices[col];
        size_t src_idx = (size_t)row + (size_t)col * ld_src;
        size_t dst_idx = (size_t)row + (size_t)dst_col * ld_dest;
        dest[dst_idx] = src[src_idx];
    }
}

long int m, n, k, rank;
double eps;
std::string spectrum_type; // "geometric", "uniform", "cluster0", "cluster1"，"arithmetic", "normal"
double s_max;
double s_min;
bool debug_mode = true;

int parseArguments(int argc,char *argv[])
{
    if(argc < 9)
    {
        printf("Usage: %s <m> <n> <k> <rank> <eps> <spectrum_type> <s_max> <s_min>\n", argv[0]);
        printf("  <spectrum_type>: 'geometric', 'geometric_zero', 'uniform', 'cluster0', 'cluster1', 'arithmetic', 'normal'\n");
        printf("  <s_max>: Maximum singular value (e.g., 1.0)\n");
        printf("  <s_min>: Minimum singular value (e.g., 1e-10)\n");
        return -1;
    }
    m = atol(argv[1]);
    n = atol(argv[2]);
    k = atol(argv[3]);
    rank = atol(argv[4]);
    eps = atof(argv[5]);
    spectrum_type = argv[6];
    s_max = atof(argv[7]);
    s_min = atof(argv[8]);
    
    printf("Parsed arguments:\n");
    printf("  m = %ld, n = %ld, k = %ld, rank = %ld. eps = %e\n", m, n, k, rank, eps);
    printf("  Spectrum Type: %s\n", spectrum_type.c_str());
    printf("  Singular Value Range: [%e, %e]\n\n", s_max, s_min);

    if (spectrum_type != "geometric" && spectrum_type != "geometric_zero" && spectrum_type != "uniform" &&
        spectrum_type != "cluster0" && spectrum_type != "cluster1" &&
        spectrum_type != "arithmetic" && spectrum_type != "normal") {
        printf("Error: Invalid spectrum_type '%s'.\n", spectrum_type.c_str());
        return -1;
    }

    return 0;
}

void generate_singular_values(
    const std::string& type,
    double max_val,
    double min_val,
    std::vector<double>& s_values,
    size_t rank)
{
    s_values.resize(rank);
    if (rank == 0) return;

    if (type == "geometric") {
        printf("Generating GEOMETRICALLY distributed singular values...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            double log_max = log(max_val);
            double log_min = log(min_val);
            double step = (log_min - log_max) / (double)(rank - 1);
            for (size_t i = 0; i < rank; ++i) {
                s_values[i] = exp(log_max + (double)i * step);
            }
        }
    }
    else if (type == "geometric_zero") {
        printf("Generating GEOMETRICALLY distributed singular values with an internal ZERO block...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            size_t zero_count = std::max<size_t>(1, rank / 6);
            size_t zero_start = (rank > zero_count) ? (rank - zero_count) / 2 : 0;
            size_t zero_end = std::min(rank, zero_start + zero_count);
            size_t nonzero_count = rank - (zero_end - zero_start);

            double min_positive = (min_val > 0.0) ? min_val : max_val * 1e-6;
            double log_max = log(max_val);
            double log_min = log(min_positive);
            double step = (nonzero_count > 1) ? (log_min - log_max) / (double)(nonzero_count - 1) : 0.0;

            size_t nz_pos = 0;
            for (size_t i = 0; i < rank; ++i) {
                if (i >= zero_start && i < zero_end) {
                    s_values[i] = 0.0;
                } else {
                    double val = (nonzero_count == 1) ? max_val : exp(log_max + (double)nz_pos * step);
                    s_values[i] = val;
                    ++nz_pos;
                }
            }
        }
    }
    else if (type == "uniform") {
        printf("Generating UNIFORMLY (linearly) distributed singular values...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            double step = (max_val - min_val) / (double)(rank - 1);
            for (size_t i = 0; i < rank; ++i) {
                s_values[i] = max_val - (double)i * step;
            }
        }
    }
    else if (type == "cluster0") {
        printf("Generating 'Cluster0' singular values (sharp drop)...\n");
        size_t cutoff_rank = rank / 4;
        if (cutoff_rank == 0 && rank > 0) cutoff_rank = 1;
        
        double high_end_val = max_val * 0.9;
        double step = (cutoff_rank > 1) ? (max_val - high_end_val) / (double)(cutoff_rank - 1) : 0.0;
        
        for (size_t i = 0; i < rank; ++i) {
            if (i < cutoff_rank) {
                s_values[i] = max_val - (double)i * step;
            } else {
                s_values[i] = min_val;
            }
        }
    }
    else if (type == "cluster1") {
        printf("Generating 'Cluster1' singular values (staircase)...\n");
        size_t cutoff1 = rank / 3;
        size_t cutoff2 = 2 * rank / 3;
        double mid_val = (max_val + min_val) / 2.0;

        for (size_t i = 0; i < rank; ++i) {
            if (i < cutoff1) {
                s_values[i] = max_val;
            } else if (i < cutoff2) {
                s_values[i] = mid_val;
            } else {
                s_values[i] = min_val;
            }
        }
    }
    else if (type == "arithmetic") {
        printf("Generating ARITHMETIC progression singular values...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            double step = (max_val - min_val) / (double)(rank - 1);
            for (size_t i = 0; i < rank; ++i) {
                s_values[i] = max_val - (double)i * step;
            }
        }
    }
    else if (type == "normal") {
        printf("Generating NORMAL (Gaussian-like) distributed singular values...\n");
        double mean = (double)rank / 2.0;
        double sigma = (double)rank / 6.0; 
        
        for (size_t i = 0; i < rank; ++i) {
            double x = (double)i;
            double gaussian_weight = exp(-0.5 * pow((x - mean) / sigma, 2.0));
            s_values[i] = min_val + (max_val - min_val) * gaussian_weight;
        }
    }

}

cudaEvent_t begin_evt, end_evt;

void startTimer()
{
    cudaEventCreate(&begin_evt);
    cudaEventRecord(begin_evt);
    cudaEventCreate(&end_evt);
}

float stopTimer()
{
    cudaEventRecord(end_evt);
    cudaEventSynchronize(end_evt);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, begin_evt, end_evt);
    cudaEventDestroy(begin_evt);
    cudaEventDestroy(end_evt);
    return milliseconds;
}

template<typename T>
void print_device_matrix(const char* filename, size_t rows, size_t cols, const T* d_matrix_ptr, size_t ld)
{
    size_t num_elements_to_copy = rows * cols;
    std::vector<T> h_matrix_buffer(num_elements_to_copy);
    cudaError_t err = cudaMemcpy2D(
        h_matrix_buffer.data(),     
        rows * sizeof(T),           
        d_matrix_ptr,                
        ld * sizeof(T),            
        rows * sizeof(T),           
        cols,                        
        cudaMemcpyDeviceToHost
    );

    if (err != cudaSuccess) {
        printf("Failed to copy matrix from device to host: %s\n", cudaGetErrorString(err));
        return;
    }

    FILE *f = fopen(filename, "w");
    if (f == NULL) {
        printf("Error opening file %s!\n", filename);
        return;
    }

    printf("Printing matrix (%zu x %zu) to %s...\n", rows, cols, filename);

    for(size_t i = 0; i < rows; i++) {
        for(size_t j = 0; j < cols; j++) {
            fprintf(f, "%.6f", (double)h_matrix_buffer[j * rows + i]);
            if (j == cols - 1) {
                fprintf(f, "\n");
            } else {
                fprintf(f, ",");
            }
        }
    }

    fclose(f);
    printf("Done printing to %s.\n", filename);
}


int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;

    bool stop_condition_met = false;
    const size_t mn = std::min(m, n);
    std::cout << "Constant mn: " << mn << std::endl;
    const size_t double_rank = rank; 
    int current_m = m; 
    printf("Creating a large low-rank matrix of size %ld x %ld with rank %ld.\n", m, n, double_rank);

    thrust::device_vector<double> d_Y_final(m * double_rank);
    double* d_Y_final_ptr = thrust::raw_pointer_cast(d_Y_final.data());
    cudaMemset(d_Y_final_ptr, 0, m * double_rank * sizeof(double));

    thrust::device_vector<double> d_W_final(m * double_rank);
    double* d_W_final_ptr = thrust::raw_pointer_cast(d_W_final.data());
    cudaMemset(d_W_final_ptr, 0, m * double_rank * sizeof(double));

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);

    double one = 1.0;
    double zero = 0.0;
    double minus_one = -1.0;
    size_t accumulated_rank = 0;

    curandGenerator_t main_gen;
    curandCreateGenerator(&main_gen, CURAND_RNG_PSEUDO_DEFAULT);

    thrust::device_vector<double> d_A(m * n);
    std::vector<double> h_S_expected(double_rank);
    { 
        thrust::device_vector<double> factor1(m * double_rank);
        thrust::device_vector<double> factor2(n * double_rank);
        
        curandSetPseudoRandomGeneratorSeed(main_gen, 1234ULL);
        generateNormalMatrix(main_gen, factor1, m, double_rank); 
        curandSetPseudoRandomGeneratorSeed(main_gen, 9999ULL);
        generateNormalMatrix(main_gen, factor2, n, double_rank); 

        int lwork_geqrf1 = 0;
        int lwork_orgqr1 = 0;
        cusolverDnDgeqrf_bufferSize(cusolverHandle, m, double_rank,
                                    thrust::raw_pointer_cast(factor1.data()), m, &lwork_geqrf1);
        cusolverDnDorgqr_bufferSize(cusolverHandle, m, double_rank, double_rank,
                                    thrust::raw_pointer_cast(factor1.data()), m,
                                    thrust::raw_pointer_cast(factor1.data()), &lwork_orgqr1);
        int lwork1 = std::max(lwork_geqrf1, lwork_orgqr1);
        printf("Workspace for factor1 (m=%ld): geqrf=%d, orgqr=%d. Using max size: %d\n", m, lwork_geqrf1, lwork_orgqr1, lwork1);
        
        thrust::device_vector<double> work1(lwork1);
        thrust::device_vector<double> tau1(double_rank);
        thrust::device_vector<int> devInfo(1);

        cusolverDnDgeqrf(cusolverHandle, m, double_rank,
                         thrust::raw_pointer_cast(factor1.data()), m,
                         thrust::raw_pointer_cast(tau1.data()),
                         thrust::raw_pointer_cast(work1.data()), lwork1,
                         thrust::raw_pointer_cast(devInfo.data()));
        cusolverDnDorgqr(cusolverHandle, m, double_rank, double_rank,
                         thrust::raw_pointer_cast(factor1.data()), m,
                         thrust::raw_pointer_cast(tau1.data()),
                         thrust::raw_pointer_cast(work1.data()), lwork1,
                         thrust::raw_pointer_cast(devInfo.data()));
        const thrust::device_vector<double>& Q1 = factor1;
        
        int lwork_geqrf2 = 0;
        int lwork_orgqr2 = 0;
        cusolverDnDgeqrf_bufferSize(cusolverHandle, n, double_rank,
                                    thrust::raw_pointer_cast(factor2.data()), n, &lwork_geqrf2);
        cusolverDnDorgqr_bufferSize(cusolverHandle, n, double_rank, double_rank,
                                    thrust::raw_pointer_cast(factor2.data()), n,
                                    thrust::raw_pointer_cast(factor2.data()), &lwork_orgqr2);
        int lwork2 = std::max(lwork_geqrf2, lwork_orgqr2);
        printf("Workspace for factor2 (m=%ld): geqrf=%d, orgqr=%d. Using max size: %d\n", n, lwork_geqrf2, lwork_orgqr2, lwork2);
        thrust::device_vector<double> work2(lwork2);
        thrust::device_vector<double> tau2(double_rank);

        cusolverDnDgeqrf(cusolverHandle, n, double_rank,
                         thrust::raw_pointer_cast(factor2.data()), n,
                         thrust::raw_pointer_cast(tau2.data()),
                         thrust::raw_pointer_cast(work2.data()), lwork2,
                         thrust::raw_pointer_cast(devInfo.data()));
        cusolverDnDorgqr(cusolverHandle, n, double_rank, double_rank,
                         thrust::raw_pointer_cast(factor2.data()), n,
                         thrust::raw_pointer_cast(tau2.data()),
                         thrust::raw_pointer_cast(work2.data()), lwork2,
                         thrust::raw_pointer_cast(devInfo.data()));
        const thrust::device_vector<double>& Q2 = factor2;

        printf("Step 3: Defining a spectrum of singular values...\n");
        generate_singular_values(spectrum_type, s_max, s_min, h_S_expected, double_rank);
        thrust::device_vector<double> d_S = h_S_expected;
        printf("Constructed singular values (top 64):\n");
        for(int i=0; i<std::min((size_t)64, double_rank); ++i)
            printf("  S_expected[%d] = %f\n", i, h_S_expected[i]);

        printf("Step 4: Assembling A = factor1 * diag(S) * factor2^T...\n");
        thrust::device_vector<double> d_Temp(m * double_rank);
        dim3 threads(16, 16);
        dim3 blocks((double_rank + 15) / 16, (m + 15) / 16);
        scale_columns_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(d_Temp.data()),
                                                  thrust::raw_pointer_cast(Q1.data()),
                                                  thrust::raw_pointer_cast(d_S.data()),
                                                  m, double_rank, m, m);

        cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                    m, n, double_rank,
                    &one,
                    thrust::raw_pointer_cast(d_Temp.data()), m,
                    thrust::raw_pointer_cast(Q2.data()), n,
                    &zero,
                    thrust::raw_pointer_cast(d_A.data()), m);
        cudaDeviceSynchronize();
        printf("Matrix A constructed successfully using cuSolver.\n\n");
    }
    

    thrust::device_vector<double> d_A_backup(d_A.size());
    thrust::copy(d_A.begin(), d_A.end(), d_A_backup.begin());
    double* d_A_ptr = thrust::raw_pointer_cast(d_A.data());
    double* d_A_backup_ptr = thrust::raw_pointer_cast(d_A_backup.data());
    size_t final_rank = mn;

    printf("Calling QB_g algorithm...\n");
    const size_t num_blocks = (n + k - 1) / k;


    thrust::device_vector<double> d_A_filtered(m * double_rank);
    double* d_A_filtered_ptr = thrust::raw_pointer_cast(d_A_filtered.data());
    cudaMemset(d_A_filtered_ptr, 0, m * double_rank * sizeof(double));
    std::vector<int> h_keep_cols_global;
    h_keep_cols_global.reserve(double_rank);
    
    startTimer();
    for (size_t j = 0; j < num_blocks; ++j)
    {
        long int col_start = j * k;
        long int row_start = j * k;
        long int panel_height = m - row_start;
        long int panel_width = std::min(k, n - col_start);
        panel_width = std::min(panel_width, (long int)(double_rank - accumulated_rank));
        if (panel_height <= 0 || panel_width <= 0) break;

        thrust::device_vector<double> d_R_tsqr(panel_width * panel_width);
        double* d_R_tsqr_ptr = thrust::raw_pointer_cast(d_R_tsqr.data());
        double* d_panel_ptr = d_A_ptr + col_start * m + row_start;
        thrust::device_vector<double> d_tau(panel_width);
        double* d_tau_ptr = thrust::raw_pointer_cast(d_tau.data());
        thrust::device_vector<int> d_devInfo(1);
        int* d_devInfo_ptr = thrust::raw_pointer_cast(d_devInfo.data());

        int lwork_geqrf = 0;
        int lwork_orgqr = 0;
        cusolverDnDgeqrf_bufferSize(cusolverHandle, panel_height, panel_width,
                                    d_panel_ptr, m, &lwork_geqrf);
        cusolverDnDorgqr_bufferSize(cusolverHandle, panel_height, panel_width, panel_width,
                                    d_panel_ptr, m, d_tau_ptr, &lwork_orgqr);
        int lwork_qr = std::max(lwork_geqrf, lwork_orgqr);
        thrust::device_vector<double> d_qr_work(lwork_qr);
        double* d_qr_work_ptr = thrust::raw_pointer_cast(d_qr_work.data());

        cusolverDnDgeqrf(cusolverHandle, panel_height, panel_width,
                         d_panel_ptr, m,
                         d_tau_ptr,
                         d_qr_work_ptr, lwork_qr,
                         d_devInfo_ptr);
        dim3 threads_R_extract(16, 16);
        dim3 blocks_R_extract((panel_width + 15) / 16, (panel_width + 15) / 16);
        extract_upper_triangle_kernel<<<blocks_R_extract, threads_R_extract>>>(d_R_tsqr_ptr, panel_width,
                                                                               d_panel_ptr, m,
                                                                               panel_width, panel_width);
        cusolverDnDorgqr(cusolverHandle, panel_height, panel_width, panel_width,
                         d_panel_ptr, m,
                         d_tau_ptr,
                         d_qr_work_ptr, lwork_qr,
                         d_devInfo_ptr);
       
        thrust::device_vector<double> diag_elements_gpu(panel_width);
        extractDiagonal<<<(panel_width + 255) / 256, 256>>>(thrust::raw_pointer_cast(diag_elements_gpu.data()),
                                                            d_R_tsqr_ptr,
                                                            panel_width,
                                                            0, panel_width);

        std::vector<double> diag_elements_cpu(panel_width);
        thrust::copy(diag_elements_gpu.begin(), diag_elements_gpu.end(), diag_elements_cpu.begin());
        
        std::vector<int> h_keep_indices;
        h_keep_indices.reserve(panel_width);
        
        double max_diag = 0.0;
        for (int i = 0; i < panel_width; ++i) {
            double abs_val = std::abs(diag_elements_cpu[i]);
            if (abs_val > max_diag) {
                max_diag = abs_val;
            }
        }
        
        double relative_eps = max_diag * eps;
        for (int i = 0; i < panel_width; ++i) {
            if (std::abs(diag_elements_cpu[i]) >= relative_eps) {
                h_keep_indices.push_back(i);
            }
        }

        long int new_panel_width = h_keep_indices.size();
        if (new_panel_width == 0) {
            continue;
        }

        for (int idx : h_keep_indices) {
            h_keep_cols_global.push_back((int)(col_start + idx));
        }

        thrust::device_vector<int> d_keep_indices = h_keep_indices;

        {
            dim3 threads_gather_A(16, 16);
            dim3 blocks_gather_A((new_panel_width + 15) / 16, (m + 15) / 16);
            select_columns_kernel<<<blocks_gather_A, threads_gather_A>>>(
                d_A_filtered_ptr + (size_t)accumulated_rank * m, m,
                d_A_backup_ptr + col_start * m, m,
                thrust::raw_pointer_cast(d_keep_indices.data()),
                m, new_panel_width);
        }

        thrust::device_vector<double> d_Q_filtered(panel_height * new_panel_width);
        double* d_Q_filtered_ptr = thrust::raw_pointer_cast(d_Q_filtered.data());
        dim3 threads_gather(16, 16);
        dim3 blocks_gather((new_panel_width + 15) / 16, (panel_height + 15) / 16);
        select_columns_kernel<<<blocks_gather, threads_gather>>>(d_Q_filtered_ptr, panel_height,
                                                                 d_A_ptr + col_start * m + row_start, m,
                                                                 thrust::raw_pointer_cast(d_keep_indices.data()),
                                                                 panel_height, new_panel_width);



        thrust::device_vector<double> d_R_temp(panel_width * new_panel_width);
        double* d_R_temp_ptr = thrust::raw_pointer_cast(d_R_temp.data());
        dim3 threads_gather_R(16, 16);
        dim3 blocks_gather_R((new_panel_width + 15) / 16, (panel_width + 15) / 16);
        select_columns_kernel<<<blocks_gather_R, threads_gather_R>>>(
                                                                    d_R_temp_ptr, panel_width,
                                                                    d_R_tsqr_ptr, panel_width,
                                                                    thrust::raw_pointer_cast(d_keep_indices.data()),
                                                                    panel_width, new_panel_width);

        thrust::device_vector<double> d_R_filtered(new_panel_width * new_panel_width);
        double* d_R_filtered_ptr = thrust::raw_pointer_cast(d_R_filtered.data());
        dim3 threads_gather_R_rows(16, 16);
        dim3 blocks_gather_R_rows((new_panel_width + 15) / 16, (new_panel_width + 15) / 16);
        select_rows_kernel<<<blocks_gather_R_rows, threads_gather_R_rows>>>(d_R_filtered_ptr, new_panel_width,
                                                                            d_R_temp_ptr, panel_width,
                                                                            thrust::raw_pointer_cast(d_keep_indices.data()),
                                                                            new_panel_width, new_panel_width);


        thrust::device_vector<double> d_B(panel_height * new_panel_width);
        double* d_B_ptr = thrust::raw_pointer_cast(d_B.data());
        cublasDgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                    panel_height, new_panel_width,
                    &minus_one, d_Q_filtered_ptr, panel_height,
                    &zero, d_Q_filtered_ptr, panel_height, 
                    d_B_ptr, panel_height);

        dim3 threads_id(256);
        dim3 blocks_id((std::min(panel_height, new_panel_width) + threads_id.x - 1) / threads_id.x);
        add_identity_diagonal_kernel<<<blocks_id, threads_id>>>(d_B_ptr, panel_height, new_panel_width, panel_height);

        thrust::device_vector<double> d_W(d_B.size());
        thrust::copy(d_B.begin(), d_B.end(), d_W.begin());
        double* d_W_ptr = thrust::raw_pointer_cast(d_W.data());

        thrust::device_vector<double> d_Y(panel_height * new_panel_width);
        double* d_Y_ptr = thrust::raw_pointer_cast(d_Y.data());

        int lwork_getrf = 0;
        cusolverDnDgetrf_bufferSize(cusolverHandle, panel_height, new_panel_width,
                                    d_B_ptr, panel_height, &lwork_getrf);
        thrust::device_vector<double> d_getrf_work(lwork_getrf);
        double* d_getrf_work_ptr = thrust::raw_pointer_cast(d_getrf_work.data());
        thrust::device_vector<int> devInfo_getrf(1);
        int* d_devInfo_getrf_ptr = thrust::raw_pointer_cast(devInfo_getrf.data());

        cusolverDnDgetrf(cusolverHandle, panel_height, new_panel_width,
                         d_B_ptr, panel_height,
                         d_getrf_work_ptr, NULL,
                         d_devInfo_getrf_ptr);

        cudaMemset(d_Y_ptr, 0, panel_height * new_panel_width * sizeof(double));
        dim3 threads_for_L(16, 16); 
        dim3 blocks_for_L((new_panel_width + threads_for_L.x - 1) / threads_for_L.x,
                          (panel_height + threads_for_L.y - 1) / threads_for_L.y);
        extract_L_factor_kernel<<<blocks_for_L, threads_for_L>>>(d_Y_ptr, panel_height,
                                                                 d_B_ptr, panel_height,
                                                                 panel_height, new_panel_width);

        cublasDtrsm(cublasHandle,
                    CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_UNIT, 
                    panel_height, new_panel_width,
                    &one, d_Y_ptr, panel_height, d_W_ptr, panel_height);

        double* d_W_final_dest_ptr = d_W_final_ptr + (size_t)accumulated_rank * m;
        cudaMemcpy2D(d_W_final_dest_ptr + row_start,
                     m * sizeof(double),
                     d_W_ptr,
                     panel_height * sizeof(double),
                     panel_height * sizeof(double),
                     new_panel_width,
                     cudaMemcpyDeviceToDevice);

        double* d_Y_final_dest_ptr = d_Y_final_ptr + (size_t)accumulated_rank * m;
        cudaMemcpy2D(d_Y_final_dest_ptr + row_start,
                     m * sizeof(double),
                     d_Y_ptr,
                     panel_height * sizeof(double),
                     panel_height * sizeof(double),
                     new_panel_width,
                     cudaMemcpyDeviceToDevice);

        long int trailing_matrix_cols = n - (col_start + panel_width);
        if (trailing_matrix_cols > 0) {
            double* d_A_trailing_ptr = d_A_ptr + (size_t)(col_start + panel_width) * m + row_start;
            thrust::device_vector<double> d_Temp1(new_panel_width * trailing_matrix_cols);
            double* d_Temp1_ptr = thrust::raw_pointer_cast(d_Temp1.data());

            cublasDgemm(cublasHandle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        new_panel_width, trailing_matrix_cols, panel_height,
                        &one,
                        d_W_ptr, panel_height,
                        d_A_trailing_ptr, m,
                        &zero,
                        d_Temp1_ptr, new_panel_width);

            cublasDgemm(cublasHandle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        panel_height, trailing_matrix_cols, new_panel_width,
                        &minus_one,
                        d_Y_ptr, panel_height,
                        d_Temp1_ptr, new_panel_width,
                        &one,
                        d_A_trailing_ptr, m);
        }

        accumulated_rank += new_panel_width;
        if (new_panel_width < panel_width) {
            stop_condition_met = true;
            final_rank = accumulated_rank;
        }
        if (accumulated_rank >= double_rank) {
            stop_condition_met = true;
            if (final_rank != accumulated_rank) { 
                final_rank = accumulated_rank;   
            }
            printf("\n         !!! Stopping: Reached or exceeded theoretical rank. Final rank set to: %zu\n", final_rank);
        }
        if (stop_condition_met) {
            break;
        }
    }
    float elapsed_time_QB = stopTimer();
    printf("QB finished successfully in %f ms.\n", elapsed_time_QB);

    {
        int kept_cols = (int)h_keep_cols_global.size();
        thrust::device_vector<double> d_A_hat((size_t)m * (size_t)n);
        double* d_A_hat_ptr = thrust::raw_pointer_cast(d_A_hat.data());
        cudaMemset(d_A_hat_ptr, 0, (size_t)m * (size_t)n * sizeof(double));
        if (kept_cols > 0) {
            thrust::device_vector<int> d_keep_cols_global = h_keep_cols_global;
            dim3 threads(16, 16);
            dim3 blocks((kept_cols + threads.x - 1) / threads.x,
                        (m + threads.y - 1) / threads.y);
            scatter_columns_kernel<<<blocks, threads>>>(
                d_A_hat_ptr, (int)m,
                d_A_filtered_ptr, (int)m,
                thrust::raw_pointer_cast(d_keep_cols_global.data()),
                (int)m, kept_cols);
        }
        double norm_A = 0.0;
        cublasDnrm2(cublasHandle, (int)((size_t)m * (size_t)n), d_A_backup_ptr, 1, &norm_A);
        cublasDgeam(cublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)m, (int)n,
                    &one,
                    d_A_backup_ptr, (int)m,
                    &minus_one,
                    d_A_hat_ptr, (int)m,
                    d_A_hat_ptr, (int)m);
        double resid = 0.0;
        cublasDnrm2(cublasHandle, (int)((size_t)m * (size_t)n), d_A_hat_ptr, 1, &resid);
        double rel = (norm_A > 1e-30) ? (resid / norm_A) : resid;
        printf("Filtered A residual: ||A - A_filtered||_F = %e\n", resid);
        printf("Filtered A relative residual: ||A - A_filtered||_F / ||A||_F = %e\n", rel);
    }


    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);
    curandDestroyGenerator(main_gen);

    return 0;
}
