#include "kernels.cuh"
__global__ void conv1d_1kernel(Matrix *input, Matrix *output, Matrix *weight, Matrix *bias) {
    // input:: length x in_channels
    // output:: length x out_channels
    // kernel:: in_channels x out_channels
    // 如果kernel 很小，可以放入constant memory?
    // 不使用shared memory
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= output->height || col >= output->width) {
        return;
    }
    ElementType sum = 0;
    for (unsigned i = 0; i < input->width /*weight->height*/; i++) {
        sum += input->get_element(row, i) * weight->get_element(i, col) + bias->get_element(0, col);
    }
    output->set_element(row, col, sum);
}

__global__ void add(Matrix *a, Matrix *b, Matrix *c) {
    unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= c->height || col >= c->width) {
        return;
    }
    c->set_element(row, col, a->get_element(row, col) + b->get_element(row, col));
}
