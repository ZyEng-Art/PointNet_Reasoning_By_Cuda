#ifndef __KERNELS__
#define __KERNELS__

#include "matrix.cuh"

__global__ void conv1d_1kernel(Matrix *input, Matrix *output, Matrix *weight, Matrix *bias);

__global__ void add(Matrix *a, Matrix *b, Matrix *c);

#endif