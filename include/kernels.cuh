#ifndef __KERNELS__
#define __KERNELS__

#include "matrix.cuh"

__global__ void conv1d_1kernel(Matrix *input, Matrix *output, Matrix *weight, Matrix *bias);
__global__ void conv1d_1kernel_shared_mem(Matrix *input, Matrix *output, Matrix *weight, Matrix *bias);
__global__ void linear(Matrix *input, Matrix *output, Matrix *weight, Matrix *bias);
__global__ void linear_shared_mem(Matrix *input, Matrix *output, Matrix *weight, Matrix *bias);

__global__ void add(Matrix *a, Matrix *b, Matrix *c);

__global__ void relu2(Matrix *a);
__global__ void relu3(Matrix *a);

__global__ void maxpool1d_all_by_column(Matrix *input, Matrix *output);

__global__ void maxpool1d_all_by_row(Matrix *input, Matrix *output);

__global__ void batch_norm1d_3d(Matrix *input, Matrix *running_mean, Matrix *running_var, Matrix *gamma, Matrix *beta, ElementType eps);
__global__ void batch_norm1d_2d(Matrix *input, Matrix *running_mean, Matrix *running_var, Matrix *gamma, Matrix *beta, ElementType eps);

__global__ void self_add_i(Matrix *input, unsigned I_size);

__global__ void multifly(Matrix *input, Matrix *output, Matrix *trans);
__global__ void multifly_with_t(Matrix *input, Matrix *output, Matrix *trans);
__global__ void multifly_with_t_shared_mem(Matrix *input, Matrix *output, Matrix *trans);

__global__ void transpose(Matrix *input, Matrix *output);
__global__ void arg_max(Matrix *input, Matrix *output);

#endif